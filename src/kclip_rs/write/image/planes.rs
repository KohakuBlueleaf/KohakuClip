//! The writer's working pictures: owned planes, resized to the stored size, converted to what
//! the storage codec takes, turned upright.

use crate::codec::jpeg::Chroma;
use crate::image::{
    Color, Filter, Grid, Matrix, Picture, Plane, resize_plane_with, rgb_to_yuv, to_rgb,
};

/// One owned plane (rows of `w` bytes).
#[derive(Clone)]
pub struct Owned {
    pub data: Vec<u8>,
    pub h: usize,
    pub w: usize,
}

impl Owned {
    fn new(h: usize, w: usize) -> Self {
        Self {
            data: vec![0; h * w],
            h,
            w,
        }
    }

    pub fn view(&self) -> Plane<'_> {
        Plane {
            data: &self.data,
            stride: self.w,
            h: self.h,
            w: self.w,
        }
    }
}

/// What a storage codec takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// Full-range BT.601 Y, Cb, Cr with this chroma subsampling (JPEG, AV1, WebP); gray stays
    /// gray.
    Yuv(Chroma),
    /// R, G, B planes (JPEG XL).
    Rgb,
}

/// A picture in the target's form: Y, Cb, Cr / R, G, B, or luma alone (`gray`).
pub struct Planes {
    pub planes: Vec<Owned>,
    pub gray: bool,
}

impl Planes {
    /// Height and width of the full-resolution plane.
    pub fn size(&self) -> (usize, usize) {
        (self.planes[0].h, self.planes[0].w)
    }
}

/// `plane` resized to (h, w) with Lanczos-3.
fn resized(plane: Plane, h: usize, w: usize) -> Owned {
    resized_to(plane, (h as f64, w as f64), h, w)
}

/// `plane` conceptually resized to `full` = (nh, nw), then its top-left (h, w) window (a
/// subsampled plane whose samples cover more than the picture: odd-sized 4:2:0).
fn resized_to(plane: Plane, full: (f64, f64), h: usize, w: usize) -> Owned {
    let mut out = Owned::new(h, w);
    let grid = Grid {
        nh: full.0,
        nw: full.1,
        top: 0.0,
        left: 0.0,
        oh: h,
        ow: w,
    };
    resize_plane_with(plane, &grid, Filter::Lanczos3, &mut out.data);
    out
}

/// A chroma plane of a (luma_h, luma_w) picture onto the chroma grid (ch, cw) of the stored
/// (h, w) picture. Each plane's subsampling factor (1 or 2) is its luma size over its own
/// size, rounded; samples stay aligned (a 4:2:0 chroma row covers two luma rows, also the
/// last one of an odd-sized picture).
fn chroma_resized(
    plane: Plane,
    (luma_h, luma_w): (usize, usize),
    (h, w): (usize, usize),
    (ch, cw): (usize, usize),
) -> Owned {
    let factor = |luma: usize, chroma: usize| (luma as f64 / chroma as f64).round().max(1.0);
    let (sy, sx) = (factor(luma_h, plane.h), factor(luma_w, plane.w));
    let (ty, tx) = (factor(h, ch), factor(w, cw));
    let nh = plane.h as f64 * sy * (h as f64 / luma_h as f64) / ty;
    let nw = plane.w as f64 * sx * (w as f64 / luma_w as f64) / tx;
    resized_to(plane, (nh, nw), ch, cw)
}

/// `source` resized to (h, w) and converted for `target`.
pub fn convert(source: &Picture, h: usize, w: usize, target: Target) -> Planes {
    let [p0, p1, p2] = source.planes;
    match (source.color, target) {
        (Color::Gray, Target::Yuv(_)) => Planes {
            planes: vec![resized(p0, h, w)],
            gray: true,
        },
        (Color::Gray, Target::Rgb) => {
            let luma = resized(p0, h, w);
            Planes {
                planes: vec![luma.clone(), luma.clone(), luma],
                gray: false,
            }
        }
        // the JPEG decoder's planes: full-range BT.601 already, only resized (chroma straight
        // from its own subsampling to the target's)
        (
            Color::Yuv {
                bt709: false,
                full_range: true,
            },
            Target::Yuv(chroma),
        ) => {
            let (ch, cw) = chroma_size(chroma, h, w);
            let luma = (p0.h, p0.w);
            Planes {
                planes: vec![
                    resized(p0, h, w),
                    chroma_resized(p1, luma, (h, w), (ch, cw)),
                    chroma_resized(p2, luma, (h, w), (ch, cw)),
                ],
                gray: false,
            }
        }
        (Color::Yuv { bt709, full_range }, _) => {
            // any other YUV: to RGB at the stored size first
            let luma = (p0.h, p0.w);
            let (y, u, v) = (
                resized(p0, h, w),
                chroma_resized(p1, luma, (h, w), (h, w)),
                chroma_resized(p2, luma, (h, w), (h, w)),
            );
            let mut rgb = vec![0u8; 3 * h * w];
            let matrix = Matrix::new(bt709, full_range);
            to_rgb(
                &y.data, &u.data, &v.data, &matrix, h, w, false, false, &mut rgb,
            );
            let planes: Vec<Owned> = rgb
                .chunks_exact(h * w)
                .map(|c| Owned {
                    data: c.to_vec(),
                    h,
                    w,
                })
                .collect();
            from_rgb(planes, target)
        }
        (Color::Rgb, _) => {
            let planes = vec![resized(p0, h, w), resized(p1, h, w), resized(p2, h, w)];
            from_rgb(planes, target)
        }
    }
}

/// The chroma plane size of an (h, w) picture.
fn chroma_size(chroma: Chroma, h: usize, w: usize) -> (usize, usize) {
    let (cw, ch) = chroma.chroma_size(w as u32, h as u32);
    (ch, cw)
}

/// R, G, B planes at the stored size -> the target's planes.
fn from_rgb(rgb: Vec<Owned>, target: Target) -> Planes {
    let Target::Yuv(chroma) = target else {
        return Planes {
            planes: rgb,
            gray: false,
        };
    };
    let (h, w) = (rgb[0].h, rgb[0].w);
    let (mut y, mut u, mut v) = (Owned::new(h, w), Owned::new(h, w), Owned::new(h, w));
    rgb_to_yuv(
        &rgb[0].data,
        &rgb[1].data,
        &rgb[2].data,
        &mut y.data,
        &mut u.data,
        &mut v.data,
    );
    let (ch, cw) = chroma_size(chroma, h, w);
    if (ch, cw) != (h, w) {
        u = chroma_resized(u.view(), (h, w), (h, w), (ch, cw));
        v = chroma_resized(v.view(), (h, w), (h, w), (ch, cw));
    }
    Planes {
        planes: vec![y, u, v],
        gray: false,
    }
}

/// Turn the planes upright for an EXIF orientation (1-8): flips and transposes per plane.
pub fn orient(planes: &mut Planes, orientation: u8) {
    let (transpose, hflip, vflip) = match orientation {
        2 => (false, true, false),
        3 => (false, true, true),
        4 => (false, false, true),
        5 => (true, false, false),
        6 => (true, true, false),
        7 => (true, true, true),
        8 => (true, false, true),
        _ => (false, false, false),
    };
    for plane in &mut planes.planes {
        if transpose {
            let mut out = Owned::new(plane.w, plane.h);
            crate::image::transpose(&plane.data, plane.h, plane.w, &mut out.data);
            *plane = out;
        }
        if vflip {
            let rows: Vec<&[u8]> = plane.data.chunks_exact(plane.w).rev().collect();
            plane.data = rows.concat();
        }
        if hflip {
            for row in plane.data.chunks_exact_mut(plane.w) {
                row.reverse();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Orientations 1-8 map a labeled 2 x 3 picture to what the EXIF spec shows upright.
    #[test]
    fn orientations() {
        let stored = || Planes {
            planes: vec![Owned {
                data: vec![1, 2, 3, 4, 5, 6],
                h: 2,
                w: 3,
            }],
            gray: true,
        };
        let cases: [(u8, &[u8], usize); 8] = [
            (1, &[1, 2, 3, 4, 5, 6], 3),
            (2, &[3, 2, 1, 6, 5, 4], 3),
            (3, &[6, 5, 4, 3, 2, 1], 3),
            (4, &[4, 5, 6, 1, 2, 3], 3),
            (5, &[1, 4, 2, 5, 3, 6], 2),
            (6, &[4, 1, 5, 2, 6, 3], 2),
            (7, &[6, 3, 5, 2, 4, 1], 2),
            (8, &[3, 6, 2, 5, 1, 4], 2),
        ];
        for (orientation, want, w) in cases {
            let mut planes = stored();
            orient(&mut planes, orientation);
            assert_eq!(planes.planes[0].data, want, "orientation {orientation}");
            assert_eq!(planes.planes[0].w, w);
        }
    }
}
