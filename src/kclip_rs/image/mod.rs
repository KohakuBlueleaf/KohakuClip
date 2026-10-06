//! Pixels: decoded pictures, antialiased resize fused with the crop, YUV -> RGB conversion,
//! window copies.

mod color;
mod picture;
mod resize;
mod simd;

pub use color::{Matrix, rgb_to_yuv, to_rgb};
pub use picture::{Color, Picture, split_packed};
pub use resize::{Filter, resize_plane, resize_plane_with};
pub use simd::transpose;

/// A plane of a decoded picture.
#[derive(Clone, Copy)]
pub struct Plane<'a> {
    pub data: &'a [u8],
    pub stride: usize,
    pub h: usize,
    pub w: usize,
}

/// The plane conceptually resized to (nh, nw), then the (oh, ow) window at (top, left).
/// Fractional values describe a chroma grid of an odd-sized or odd-offset luma grid.
#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub nh: f64,
    pub nw: f64,
    pub top: f64,
    pub left: f64,
    pub oh: usize,
    pub ow: usize,
}

impl Grid {
    /// The same grid at half resolution (4:2:0 chroma of an output grid).
    pub fn half(&self) -> Grid {
        Grid {
            nh: self.nh / 2.0,
            nw: self.nw / 2.0,
            top: self.top / 2.0,
            left: self.left / 2.0,
            oh: self.oh / 2,
            ow: self.ow / 2,
        }
    }
}

/// Copy the (oh, ow) window at (top, left) of a plane into `dst`.
pub fn copy_window(src: Plane, top: usize, left: usize, oh: usize, ow: usize, dst: &mut [u8]) {
    for (y, out) in dst.chunks_exact_mut(ow).take(oh).enumerate() {
        let at = (top + y) * src.stride + left;
        out.copy_from_slice(&src.data[at..at + ow]);
    }
}
