//! Planning one image: where its bytes are, and (once its size is known, in the decode thread)
//! its crop, resize and flips.
//!
//! The per-image randomness is a seed drawn at planning, so a batch is a pure function of the
//! call's seed whether the stored sizes come from the index or from each image's header.

use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

use super::plan::Mode;
use crate::codec::jpeg;
use crate::decode::Output;
use crate::image::Grid;
use crate::storage::image_shard::ImageEntry;

/// How the output square is cut from an image.
#[derive(Clone, Copy, Debug)]
pub enum Crop {
    /// Short side resized to the output size, then a square crop, random or centered.
    Short { random: bool },
    /// torchvision's RandomResizedCrop: a random area fraction in `scale` with a random aspect
    /// ratio in `ratio` (log-uniform), resized to the output square.
    Resized {
        scale: (f64, f64),
        ratio: (f64, f64),
    },
}

/// Output settings shared by every image of a reader.
#[derive(Clone, Copy, Debug)]
pub struct ImageSettings {
    pub size: u32,
    pub mode: Mode,
    pub crop: Crop,
    /// Probability of a horizontal / vertical flip.
    pub hflip: f64,
    pub vflip: f64,
    /// Let the JPEG decoder shrink by 1/2, 1/4 or 1/8 when the output needs no more pixels.
    pub dct_scale: bool,
}

pub struct ImagePlan {
    pub entry: ImageEntry,
    /// Seeds this image's crop and flips.
    pub seed: u64,
    /// "yuv" mode: the side of the stored-resolution window (shared by the batch).
    pub side: u32,
}

pub fn plan(entry: ImageEntry, side: u32, rng: &mut impl Rng) -> ImagePlan {
    ImagePlan {
        entry,
        seed: rng.random(),
        side,
    }
}

/// The flips of an image: the first draws of its seed, so they are known without its size.
fn draw_flips(settings: &ImageSettings, rng: &mut impl Rng) -> (bool, bool) {
    let hflip = rng.random::<f64>() < settings.hflip;
    let vflip = rng.random::<f64>() < settings.vflip;
    (hflip, vflip)
}

/// The flips of a planned image (horizontal, vertical), as its decode applies them.
pub fn flips(settings: &ImageSettings, plan: &ImagePlan) -> (bool, bool) {
    draw_flips(settings, &mut StdRng::seed_from_u64(plan.seed))
}

/// The output grid and flips of a stored (h, w) image.
pub fn layout(h: u32, w: u32, settings: &ImageSettings, plan: &ImagePlan) -> Output {
    let mut rng = StdRng::seed_from_u64(plan.seed);
    let (hflip, vflip) = draw_flips(settings, &mut rng);
    let size = settings.size;
    let grid = if settings.mode == Mode::Yuv {
        stored_window(h, w, plan.side, settings.crop, &mut rng)
    } else {
        match settings.crop {
            Crop::Short { random } => short_side(h, w, size, random, &mut rng),
            Crop::Resized { scale, ratio } => resized_crop(h, w, size, scale, ratio, &mut rng),
        }
    };
    Output {
        grid,
        hflip,
        vflip,
        mode: settings.mode,
    }
}

/// Short side resized to `size` (as the video reader), then the square at a random or centered
/// position.
fn short_side(h: u32, w: u32, size: u32, random: bool, rng: &mut impl Rng) -> Grid {
    let scale = size as f64 / h.min(w) as f64;
    let resized = |x: u32| ((x as f64 * scale).round_ties_even() as u32).max(size);
    let (nh, nw) = (resized(h), resized(w));
    let (top, left) = if random {
        (
            rng.random_range(0..=nh - size),
            rng.random_range(0..=nw - size),
        )
    } else {
        ((nh - size) / 2, (nw - size) / 2)
    };
    Grid {
        nh: nh as f64,
        nw: nw as f64,
        top: top as f64,
        left: left as f64,
        oh: size as usize,
        ow: size as usize,
    }
}

/// The yuv mode: a `side` x `side` window of the stored planes at an even position.
fn stored_window(h: u32, w: u32, side: u32, crop: Crop, rng: &mut impl Rng) -> Grid {
    let (top, left) = match crop {
        Crop::Short { random: false } => ((h - side) / 2, (w - side) / 2),
        _ => (
            rng.random_range(0..=h - side),
            rng.random_range(0..=w - side),
        ),
    };
    Grid {
        nh: h as f64,
        nw: w as f64,
        top: (top & !1) as f64,
        left: (left & !1) as f64,
        oh: side as usize,
        ow: side as usize,
    }
}

/// RandomResizedCrop (torchvision's algorithm: up to 10 draws of area and aspect ratio, then a
/// centered crop of the image's own ratio clamped to `ratio`), as a grid: the whole image
/// resized so that the crop becomes `size` x `size`, and the crop's position in it.
fn resized_crop(
    h: u32,
    w: u32,
    size: u32,
    scale: (f64, f64),
    ratio: (f64, f64),
    rng: &mut impl Rng,
) -> Grid {
    let (hf, wf) = (h as f64, w as f64);
    let area = hf * wf;
    let log_ratio = (ratio.0.ln(), ratio.1.ln());

    let mut crop = None;
    for _ in 0..10 {
        let target = area * rng.random_range(scale.0..=scale.1);
        let aspect = rng.random_range(log_ratio.0..=log_ratio.1).exp();
        let cw = (target * aspect).sqrt().round();
        let ch = (target / aspect).sqrt().round();
        if cw > 0.0 && cw <= wf && ch > 0.0 && ch <= hf {
            let top = rng.random_range(0..=(hf - ch) as u32) as f64;
            let left = rng.random_range(0..=(wf - cw) as u32) as f64;
            crop = Some((top, left, ch, cw));
            break;
        }
    }
    let (top, left, ch, cw) = crop.unwrap_or_else(|| {
        let image_ratio = wf / hf;
        let (cw, ch) = if image_ratio < ratio.0 {
            (wf, (wf / ratio.0).round())
        } else if image_ratio > ratio.1 {
            ((hf * ratio.1).round(), hf)
        } else {
            (wf, hf)
        };
        (((hf - ch) / 2.0).floor(), ((wf - cw) / 2.0).floor(), ch, cw)
    });

    let (sy, sx) = (size as f64 / ch, size as f64 / cw);
    Grid {
        nh: hf * sy,
        nw: wf * sx,
        top: top * sy,
        left: left * sx,
        oh: size as usize,
        ow: size as usize,
    }
}

/// How far the JPEG decoder may shrink an (h, w) image for `grid` (1 / 2^shift): the shrunk
/// planes keep at least the grid's resolution on both axes.
pub fn dct_shift(h: u32, w: u32, grid: &Grid) -> u32 {
    jpeg::max_shift(h, w, grid.nh, grid.nw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resized_crops_stay_inside() {
        let mut rng = StdRng::seed_from_u64(0);
        for (h, w) in [(512, 683), (384, 1600), (1000, 400)] {
            for _ in 0..200 {
                let g = resized_crop(h, w, 256, (0.08, 1.0), (0.75, 4.0 / 3.0), &mut rng);
                assert!(g.top >= -1e-9 && g.left >= -1e-9, "{g:?}");
                assert!(
                    g.top + 256.0 <= g.nh + 1e-6 && g.left + 256.0 <= g.nw + 1e-6,
                    "{g:?}"
                );
            }
        }
    }

    #[test]
    fn dct_shift_keeps_resolution() {
        let grid = |nh: f64, nw: f64| Grid {
            nh,
            nw,
            top: 0.0,
            left: 0.0,
            oh: 256,
            ow: 256,
        };
        assert_eq!(dct_shift(512, 683, &grid(256.0, 342.0)), 1);
        assert_eq!(dct_shift(512, 683, &grid(300.0, 400.0)), 0);
        assert_eq!(dct_shift(2048, 2048, &grid(256.0, 256.0)), 3);
        assert_eq!(dct_shift(400, 400, &grid(256.0, 256.0)), 0);
    }
}
