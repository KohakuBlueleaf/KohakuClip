//! The output pixels of one decoded picture (a video frame or a still image), per output mode.

use super::profile::{Stage, timed};
use crate::image::{Color, Grid, Matrix, Picture, copy_window, resize_plane, to_rgb};
use crate::read::plan::Mode;

/// Where a picture's output pixels come from and how they are written.
#[derive(Clone, Copy, Debug)]
pub struct Output {
    /// The picture conceptually resized to (nh, nw), then the (oh, ow) window.
    pub grid: Grid,
    pub hflip: bool,
    pub vflip: bool,
    pub mode: Mode,
}

/// Write the output pixels of `picture` into `dst` (one output slot). `scratch` holds the
/// resized planes of the rgb mode between the resize and the conversion.
pub fn emit(
    picture: &Picture,
    out: &Output,
    dst: &mut [u8],
    scratch: &mut Vec<u8>,
) -> Result<(), String> {
    match out.mode {
        Mode::Yuv => timed(Stage::Convert, || stored_window(picture, &out.grid, dst)),
        Mode::YuvResized => timed(Stage::Resize, || resized_planes(picture, &out.grid, dst)),
        Mode::Rgb => rgb(picture, out, dst, scratch),
    }
}

/// "yuv": the (oh, ow) window at (top, left) of the stored planes as 4:2:0: copied as they are
/// (4:2:0 pictures), or with the chroma resampled (other subsamplings).
fn stored_window(picture: &Picture, grid: &Grid, dst: &mut [u8]) -> Result<(), String> {
    let (oh, ow) = (grid.oh, grid.ow);
    let (top, left) = (grid.top as usize, grid.left as usize);
    let n = oh * ow;
    let (luma, chroma) = dst.split_at_mut(n);
    let (cb, cr) = chroma.split_at_mut(n / 4);
    let [y, u, v] = picture.planes;

    match picture.color {
        Color::Yuv { .. } => {
            copy_window(y, top, left, oh, ow, luma);
            let (h, w) = picture.size();
            if u.h == h.div_ceil(2) && u.w == w.div_ceil(2) {
                copy_window(u, top / 2, left / 2, oh / 2, ow / 2, cb);
                copy_window(v, top / 2, left / 2, oh / 2, ow / 2, cr);
            } else {
                // other subsamplings (4:4:4 JPEGs): chroma resampled onto the 4:2:0 grid
                resize_plane(u, &grid.half(), cb);
                resize_plane(v, &grid.half(), cr);
            }
        }
        Color::Gray => {
            copy_window(y, top, left, oh, ow, luma);
            cb.fill(128);
            cr.fill(128);
        }
        Color::Rgb => return Err("yuv modes need YUV-coded pictures".into()),
    }
    Ok(())
}

/// "yuv_resized": luma resized to the output grid, chroma to the same grid at half resolution.
fn resized_planes(picture: &Picture, grid: &Grid, dst: &mut [u8]) -> Result<(), String> {
    let n = grid.oh * grid.ow;
    let (luma, chroma) = dst.split_at_mut(n);
    let (cb, cr) = chroma.split_at_mut(n / 4);
    let [y, u, v] = picture.planes;

    match picture.color {
        Color::Yuv { .. } => {
            resize_plane(y, grid, luma);
            resize_plane(u, &grid.half(), cb);
            resize_plane(v, &grid.half(), cr);
        }
        Color::Gray => {
            resize_plane(y, grid, luma);
            cb.fill(128);
            cr.fill(128);
        }
        Color::Rgb => return Err("yuv modes need YUV-coded pictures".into()),
    }
    Ok(())
}

/// "rgb": every plane resized onto the output grid, then converted (YUV) or copied (RGB, gray)
/// into R, G, B with the flips.
fn rgb(
    picture: &Picture,
    out: &Output,
    dst: &mut [u8],
    scratch: &mut Vec<u8>,
) -> Result<(), String> {
    let grid = &out.grid;
    let (oh, ow) = (grid.oh, grid.ow);
    let n = oh * ow;
    let [y, u, v] = picture.planes;

    match picture.color {
        Color::Yuv { bt709, full_range } => {
            scratch.resize(3 * n, 0);
            let (py, rest) = scratch.split_at_mut(n);
            let (pu, pv) = rest.split_at_mut(n);
            timed(Stage::Resize, || {
                resize_plane(y, grid, py);
                resize_plane(u, grid, pu);
                resize_plane(v, grid, pv);
            });
            let matrix = Matrix::new(bt709, full_range);
            timed(Stage::Convert, || {
                to_rgb(py, pu, pv, &matrix, oh, ow, out.hflip, out.vflip, dst);
            });
        }
        Color::Rgb => {
            timed(Stage::Resize, || {
                for (plane, channel) in picture.planes.iter().zip(dst.chunks_exact_mut(n)) {
                    resize_plane(*plane, grid, channel);
                }
            });
            timed(Stage::Convert, || {
                flip_channels(dst, oh, ow, out.hflip, out.vflip)
            });
        }
        Color::Gray => {
            timed(Stage::Resize, || resize_plane(y, grid, &mut dst[..n]));
            timed(Stage::Convert, || {
                dst.copy_within(..n, n);
                dst.copy_within(..n, 2 * n);
                flip_channels(dst, oh, ow, out.hflip, out.vflip);
            });
        }
    }
    Ok(())
}

/// Flip each (oh, ow) channel of `dst` in place.
fn flip_channels(dst: &mut [u8], oh: usize, ow: usize, hflip: bool, vflip: bool) {
    for channel in dst.chunks_exact_mut(oh * ow) {
        if vflip {
            for row in 0..oh / 2 {
                let (upper, lower) = channel.split_at_mut((oh - 1 - row) * ow);
                upper[row * ow..(row + 1) * ow].swap_with_slice(&mut lower[..ow]);
            }
        }
        if hflip {
            for row in channel.chunks_exact_mut(ow) {
                row.reverse();
            }
        }
    }
}
