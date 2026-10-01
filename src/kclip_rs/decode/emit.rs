//! The output pixels of one decoded frame, per output mode.

use rsmpeg::avutil::AVFrame;
use rsmpeg::ffi;

use super::profile::{Stage, timed};
use crate::image::{Grid, Matrix, Plane, copy_window, resize_plane, to_rgb};
use crate::read::plan::{ClipPlan, Mode};

/// The conversion applied for a frame's matrix coefficients: BT.601 only when tagged so;
/// untagged video is read as BT.709 (the writer tags untagged SD sources as BT.601 explicitly).
pub fn colorspace_name(matrix: ffi::AVColorSpace) -> &'static str {
    match matrix {
        ffi::AVCOL_SPC_BT470BG | ffi::AVCOL_SPC_SMPTE170M | ffi::AVCOL_SPC_FCC => "bt601",
        _ => "bt709",
    }
}

/// Plane `p` of a decoded frame, with its subsampled size.
fn plane(frame: &AVFrame, p: usize, chroma_shift: (u32, u32)) -> Plane<'_> {
    let (h, w) = (frame.height as usize, frame.width as usize);
    let (h, w) = if p == 0 {
        (h, w)
    } else {
        (
            h.div_ceil(1 << chroma_shift.0),
            w.div_ceil(1 << chroma_shift.1),
        )
    };
    let stride = frame.linesize[p] as usize;
    // SAFETY: an 8-bit planar frame holds h rows of `stride` bytes per plane
    let data = unsafe { std::slice::from_raw_parts(frame.data[p], stride * (h - 1) + w) };
    Plane { data, stride, h, w }
}

/// Write the output pixels of `frame` into `dst` (one output slot).
pub fn emit(
    plan: &ClipPlan,
    frame: &AVFrame,
    dst: &mut [u8],
    planes: &mut Vec<u8>,
) -> Result<(), String> {
    // SAFETY: the pixel format of a decoded frame always has a descriptor
    let desc = unsafe { &*ffi::av_pix_fmt_desc_get(frame.format) };
    if desc.comp[0].depth != 8 || desc.nb_components < 3 {
        return Err("only 8-bit YUV video is supported".into());
    }
    let shift = (desc.log2_chroma_h as u32, desc.log2_chroma_w as u32);
    let (y, u, v) = (
        plane(frame, 0, shift),
        plane(frame, 1, shift),
        plane(frame, 2, shift),
    );

    let g = plan.geometry;
    let grid = Grid {
        nh: g.nh as f64,
        nw: g.nw as f64,
        top: g.top as f64,
        left: g.left as f64,
        oh: g.oh as usize,
        ow: g.ow as usize,
    };
    let (oh, ow) = (grid.oh, grid.ow);
    let n = oh * ow;

    match plan.mode {
        Mode::Yuv => timed(Stage::Convert, || {
            let (top, left) = (g.top as usize, g.left as usize);
            let (luma, chroma) = dst.split_at_mut(n);
            let (cb, cr) = chroma.split_at_mut(n / 4);
            copy_window(y, top, left, oh, ow, luma);
            copy_window(u, top / 2, left / 2, oh / 2, ow / 2, cb);
            copy_window(v, top / 2, left / 2, oh / 2, ow / 2, cr);
        }),
        Mode::YuvResized => timed(Stage::Resize, || {
            let (luma, chroma) = dst.split_at_mut(n);
            let (cb, cr) = chroma.split_at_mut(n / 4);
            resize_plane(y, &grid, luma);
            resize_plane(u, &grid.half(), cb);
            resize_plane(v, &grid.half(), cr);
        }),
        Mode::Rgb => {
            planes.resize(3 * n, 0);
            let (py, rest) = planes.split_at_mut(n);
            let (pu, pv) = rest.split_at_mut(n);
            timed(Stage::Resize, || {
                resize_plane(y, &grid, py);
                resize_plane(u, &grid, pu);
                resize_plane(v, &grid, pv);
            });
            let bt709 = colorspace_name(frame.colorspace) == "bt709";
            let full_range = frame.color_range == ffi::AVCOL_RANGE_JPEG;
            let matrix = Matrix::new(bt709, full_range);
            timed(Stage::Convert, || {
                to_rgb(py, pu, pv, &matrix, oh, ow, g.hflip, g.vflip, dst);
            });
        }
    }
    Ok(())
}
