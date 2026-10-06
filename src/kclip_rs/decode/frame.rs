//! A decoded FFmpeg frame as a `Picture`: 8-bit planar YUV in place, packed RGB(A) split into
//! planes (into a per-thread buffer).

use rsmpeg::avutil::AVFrame;
use rsmpeg::ffi;

use crate::image::{Color, Picture, Plane, split_packed};

/// Whether a frame's matrix coefficients are read as BT.709: everything but the BT.601 tags
/// (untagged video included; the writer tags untagged SD sources as BT.601 explicitly).
fn is_bt709(matrix: ffi::AVColorSpace) -> bool {
    !matches!(
        matrix,
        ffi::AVCOL_SPC_BT470BG | ffi::AVCOL_SPC_SMPTE170M | ffi::AVCOL_SPC_FCC
    )
}

/// The colorspace name of a frame's matrix coefficients: "bt709" or "bt601".
pub fn colorspace_name(matrix: ffi::AVColorSpace) -> &'static str {
    if is_bt709(matrix) { "bt709" } else { "bt601" }
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

/// A frame as a picture: 8-bit planar YUV (any subsampling) or gray read in place; packed
/// RGB24 / RGBA (still-image decoders) split into R, G, B planes in `planes` (alpha dropped).
pub fn picture<'a>(frame: &'a AVFrame, planes: &'a mut Vec<u8>) -> Result<Picture<'a>, String> {
    let format = frame.format;
    if format == ffi::AV_PIX_FMT_RGB24 || format == ffi::AV_PIX_FMT_RGBA {
        return Ok(split_rgb(frame, planes));
    }

    // SAFETY: the pixel format of a decoded frame always has a descriptor
    let desc = unsafe { &*ffi::av_pix_fmt_desc_get(format) };
    let is_rgb = desc.flags & ffi::AV_PIX_FMT_FLAG_RGB as u64 != 0;
    // Y, U, V each alone in its own plane (not semi-planar like NV12), one byte per sample
    let three_planes = desc.nb_components >= 3
        && (0..3).all(|c| desc.comp[c].plane == c as i32 && desc.comp[c].step == 1);
    let gray = desc.nb_components == 1;
    if desc.comp[0].depth != 8 || is_rgb || !(three_planes || gray) {
        return Err("only 8-bit planar YUV, gray or packed RGB pictures are supported".into());
    }
    let full_range =
        frame.color_range == ffi::AVCOL_RANGE_JPEG || format == ffi::AV_PIX_FMT_YUVJ420P;
    if desc.nb_components == 1 {
        let y = plane(frame, 0, (0, 0));
        return Ok(Picture {
            planes: [y, y, y],
            color: Color::Gray,
        });
    }

    let shift = (desc.log2_chroma_h as u32, desc.log2_chroma_w as u32);
    Ok(Picture {
        planes: [
            plane(frame, 0, shift),
            plane(frame, 1, shift),
            plane(frame, 2, shift),
        ],
        color: Color::Yuv {
            bt709: is_bt709(frame.colorspace),
            full_range,
        },
    })
}

/// Packed RGB24 / RGBA -> three planes of `buffer`.
fn split_rgb<'a>(frame: &AVFrame, buffer: &'a mut Vec<u8>) -> Picture<'a> {
    let (h, w) = (frame.height as usize, frame.width as usize);
    let channels = if frame.format == ffi::AV_PIX_FMT_RGBA {
        4
    } else {
        3
    };
    let stride = frame.linesize[0] as usize;
    // SAFETY: a packed frame holds h rows of `stride` bytes
    let src = unsafe { std::slice::from_raw_parts(frame.data[0], stride * (h - 1) + w * channels) };
    split_packed(src, stride, (h, w), channels, buffer)
}
