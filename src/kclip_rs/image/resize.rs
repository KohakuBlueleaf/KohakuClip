//! Antialiased resize fused with the crop, per plane.
//!
//! The reader's filter is bilinear, the same as
//! `torch.nn.functional.interpolate(mode="bilinear", antialias=True)` (PIL BILINEAR); the writer
//! downscales with Lanczos-3 (PIL LANCZOS). Q14 weights, uint8 intermediates, two separable
//! passes. Chroma planes use the same call; their taps map the smaller plane straight onto the
//! output grid.

use std::cell::RefCell;

use super::simd::{self, transpose};
use super::{Grid, Plane};

/// The resampling filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    /// Triangle, support 1 (the reader).
    Bilinear,
    /// Windowed sinc, support 3 (the writer), when shrinking; bilinear when enlarging.
    Lanczos3,
}

impl Filter {
    fn support(self) -> f64 {
        match self {
            Filter::Bilinear => 1.0,
            Filter::Lanczos3 => 3.0,
        }
    }

    /// The filter's weight at `x` input samples (scaled to the output) from the center.
    fn weight(self, x: f64) -> f64 {
        match self {
            Filter::Bilinear => (1.0 - x.abs()).max(0.0),
            Filter::Lanczos3 => {
                if x.abs() >= 3.0 {
                    0.0
                } else {
                    sinc(x) * sinc(x / 3.0)
                }
            }
        }
    }
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let x = x * std::f64::consts::PI;
    x.sin() / x
}

/// Filter taps of every output sample along one axis.
#[derive(Default)]
struct Taps {
    /// First input sample of each output sample.
    first: Vec<usize>,
    /// Number of input samples of each output sample.
    count: Vec<usize>,
    /// `max` Q14 weights per output sample.
    weights: Vec<i16>,
    max: usize,
}

impl Taps {
    /// Taps for outputs `out0 + [0, outn)` of `input` samples resized to `out`.
    fn build(&mut self, input: usize, out: f64, out0: f64, outn: usize, filter: Filter) {
        let scale = input as f64 / out;
        // Lanczos-3 when shrinking; enlarging (e.g. 4:2:0 chroma onto a 4:4:4 grid) is bilinear
        let filter = if scale < 1.0 {
            Filter::Bilinear
        } else {
            filter
        };
        let stretch = scale.max(1.0);
        let support = filter.support() * stretch;
        let inverse = 1.0 / stretch;
        self.max = support.ceil() as usize * 2 + 1;
        self.first.clear();
        self.count.clear();
        self.weights.clear();
        self.weights.resize(outn * self.max, 0);

        let mut w = vec![0.0f64; self.max];
        for i in 0..outn {
            let center = (out0 + i as f64 + 0.5) * scale;
            let lo = ((center - support + 0.5) as isize).max(0) as usize;
            let hi = ((center + support + 0.5) as usize).min(input);
            let n = (hi - lo).min(self.max);

            let mut total = 0.0;
            for (j, wj) in w.iter_mut().take(n).enumerate() {
                let distance = ((j + lo) as f64 - center + 0.5) * inverse;
                *wj = filter.weight(distance);
                total += *wj;
            }
            let row = &mut self.weights[i * self.max..i * self.max + n];
            for (q, wj) in row.iter_mut().zip(&w) {
                *q = (wj / total * 16384.0).round() as i16;
            }
            self.first.push(lo);
            self.count.push(n);
        }
    }

    fn of(&self, i: usize) -> (usize, &[i16]) {
        let n = self.count[i];
        (self.first[i], &self.weights[i * self.max..i * self.max + n])
    }
}

/// dst[x] = sum_j weights[j] * rows[j][x] (Q14, rounded, clamped): SIMD where the CPU has it,
/// the plain loop for the rest.
fn weighted_rows(rows: &[&[u8]], weights: &[i16], acc: &mut Vec<i32>, dst: &mut [u8]) {
    let done = simd::weighted_rows(rows, weights, dst);
    let rest: Vec<&[u8]> = rows.iter().map(|r| &r[done..]).collect();
    weighted_rows_plain(&rest, weights, acc, &mut dst[done..]);
}

fn weighted_rows_plain(rows: &[&[u8]], weights: &[i16], acc: &mut Vec<i32>, dst: &mut [u8]) {
    let len = dst.len();
    acc.clear();
    acc.resize(len, 1 << 13);
    for (row, &w) in rows.iter().zip(weights) {
        let w = w as i32;
        for (a, &v) in acc.iter_mut().zip(&row[..len]) {
            *a += w * v as i32;
        }
    }
    for (d, &a) in dst.iter_mut().zip(acc.iter()) {
        *d = (a >> 14).clamp(0, 255) as u8;
    }
}

/// Per-thread buffers (taps and intermediates are rebuilt per call, allocations are kept).
#[derive(Default)]
struct Scratch {
    tx: Taps,
    ty: Taps,
    vertical: Vec<u8>,
    columns: Vec<u8>,
    horizontal: Vec<u8>,
    acc: Vec<i32>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// Resize and crop `src` as `grid` says into `dst` (oh * ow bytes) with the bilinear filter.
pub fn resize_plane(src: Plane, grid: &Grid, dst: &mut [u8]) {
    resize_plane_with(src, grid, Filter::Bilinear, dst);
}

/// Resize and crop `src` as `grid` says into `dst` (oh * ow bytes): a vertical pass over the
/// needed rows (only the window the crop needs), then the horizontal pass as a vertical pass on
/// the transposed result.
pub fn resize_plane_with(src: Plane, grid: &Grid, filter: Filter, dst: &mut [u8]) {
    // a plane already at the grid's resolution, at a whole-sample offset: a copy of the window
    // (what the filter's (1, 0) taps give, bit for bit)
    let whole = |x: f64| x.fract() == 0.0 && x >= 0.0;
    let same_size = grid.nh == src.h as f64 && grid.nw == src.w as f64;
    if same_size && whole(grid.top) && whole(grid.left) {
        let (top, left) = (grid.top as usize, grid.left as usize);
        super::copy_window(src, top, left, grid.oh, grid.ow, dst);
        return;
    }
    SCRATCH.with_borrow_mut(|s| {
        let (oh, ow) = (grid.oh, grid.ow);
        s.tx.build(src.w, grid.nw, grid.left, ow, filter);
        s.ty.build(src.h, grid.nh, grid.top, oh, filter);
        let x0 = s.tx.first[0];
        let width = s.tx.first[ow - 1] + s.tx.count[ow - 1] - x0;

        let mut rows: Vec<&[u8]> = Vec::with_capacity(s.ty.max.max(s.tx.max));
        s.vertical.resize(oh * width, 0);
        for y in 0..oh {
            let (first, weights) = s.ty.of(y);
            rows.clear();
            for j in 0..weights.len() {
                let at = (first + j) * src.stride + x0;
                rows.push(&src.data[at..at + width]);
            }
            let out = &mut s.vertical[y * width..(y + 1) * width];
            weighted_rows(&rows, weights, &mut s.acc, out);
        }

        s.columns.resize(width * oh, 0);
        transpose(&s.vertical, oh, width, &mut s.columns);

        s.horizontal.resize(ow * oh, 0);
        for x in 0..ow {
            let (first, weights) = s.tx.of(x);
            rows.clear();
            for j in 0..weights.len() {
                let at = (first + j - x0) * oh;
                rows.push(&s.columns[at..at + oh]);
            }
            let out = &mut s.horizontal[x * oh..(x + 1) * oh];
            weighted_rows(&rows, weights, &mut s.acc, out);
        }
        transpose(&s.horizontal, ow, oh, dst);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lanczos-3 keeps a flat plane flat (the weights, negative lobes included, sum to one).
    #[test]
    fn lanczos_keeps_flat_planes() {
        let (h, w) = (64usize, 90usize);
        let flat = vec![137u8; h * w];
        let plane = Plane {
            data: &flat,
            stride: w,
            h,
            w,
        };
        let grid = Grid {
            nh: 32.0,
            nw: 45.0,
            top: 0.0,
            left: 0.0,
            oh: 32,
            ow: 45,
        };
        let mut out = vec![0u8; 32 * 45];
        resize_plane_with(plane, &grid, Filter::Lanczos3, &mut out);
        assert!(out.iter().all(|&v| v == 137));
    }

    /// A 1:1 grid (copied) gives what the filter gives at that scale.
    #[test]
    fn identity_grid_is_a_copy() {
        let (h, w) = (40usize, 70usize);
        let data: Vec<u8> = (0..h * w).map(|i| (i * 29 % 251) as u8).collect();
        let plane = Plane {
            data: &data,
            stride: w,
            h,
            w,
        };
        let grid = Grid {
            nh: h as f64,
            nw: w as f64,
            top: 3.0,
            left: 5.0,
            oh: 32,
            ow: 48,
        };
        let mut copied = vec![0u8; 32 * 48];
        resize_plane(plane, &grid, &mut copied);

        // the same grid nudged off the shortcut by a sub-ulp left offset goes through the filter
        let nudged = Grid {
            left: 5.0 + 1e-12,
            ..grid
        };
        let mut filtered = vec![0u8; 32 * 48];
        resize_plane(plane, &nudged, &mut filtered);
        assert_eq!(copied, filtered);
    }
}

#[cfg(test)]
mod bench {
    use std::time::Instant;

    use super::*;
    use crate::image::{Matrix, to_rgb};

    /// `cargo test --release -- --ignored --nocapture resize_speed`: time per stage for a 910x512
    /// luma plane (and 455x256 chroma) to a 256 crop.
    #[test]
    #[ignore]
    fn resize_speed() {
        let (h, w) = (512usize, 910usize);
        let luma: Vec<u8> = (0..h * w).map(|i| (i * 31 % 251) as u8).collect();
        let chroma: Vec<u8> = (0..h * w / 4).map(|i| (i * 17 % 241) as u8).collect();
        let grid = Grid {
            nh: 256.0,
            nw: 455.0,
            top: 0.0,
            left: 100.0,
            oh: 256,
            ow: 256,
        };
        let y = Plane {
            data: &luma,
            stride: w,
            h,
            w,
        };
        let c = Plane {
            data: &chroma,
            stride: w / 2,
            h: h / 2,
            w: w / 2,
        };
        let mut out = vec![0u8; 3 * 256 * 256];
        let iters = 2000;

        let t = Instant::now();
        for _ in 0..iters {
            let (a, rest) = out.split_at_mut(256 * 256);
            let (b, d) = rest.split_at_mut(256 * 256);
            resize_plane(y, &grid, a);
            resize_plane(c, &grid, b);
            resize_plane(c, &grid, d);
        }
        let resize = t.elapsed().as_secs_f64() / iters as f64 * 1e6;

        let planes = out.clone();
        let matrix = Matrix::new(true, false);
        let t = Instant::now();
        for _ in 0..iters {
            let (py, rest) = planes.split_at(256 * 256);
            let (pu, pv) = rest.split_at(256 * 256);
            to_rgb(py, pu, pv, &matrix, 256, 256, false, false, &mut out);
        }
        let convert = t.elapsed().as_secs_f64() / iters as f64 * 1e6;

        // the parts of one luma resize
        let mut s = Scratch::default();
        s.tx.build(w, grid.nw, grid.left, 256, Filter::Bilinear);
        s.ty.build(h, grid.nh, grid.top, 256, Filter::Bilinear);
        let x0 = s.tx.first[0];
        let width = s.tx.first[255] + s.tx.count[255] - x0;
        s.vertical.resize(256 * width, 0);
        s.columns.resize(256 * width, 0);
        let t = Instant::now();
        for _ in 0..iters {
            for row in 0..256 {
                let (first, weights) = s.ty.of(row);
                let rows: Vec<&[u8]> = (0..weights.len())
                    .map(|j| &luma[(first + j) * w + x0..(first + j) * w + x0 + width])
                    .collect();
                let dst = &mut s.vertical[row * width..(row + 1) * width];
                weighted_rows(&rows, weights, &mut s.acc, dst);
            }
        }
        let vertical = t.elapsed().as_secs_f64() / iters as f64 * 1e6;
        let t = Instant::now();
        for _ in 0..iters {
            transpose(&s.vertical, 256, width, &mut s.columns);
        }
        let transposed = t.elapsed().as_secs_f64() / iters as f64 * 1e6;

        println!(
            "per frame (3 planes): resize {resize:.1} us, convert {convert:.1} us; \
             luma parts: vertical {vertical:.1} us, transpose {transposed:.1} us (x{width})"
        );
    }
}
