//! Encoding the stored picture: JPEG through mozjpeg or libjpeg-turbo; AV1 (intra, raw OBUs),
//! JPEG XL and WebP through the linked FFmpeg's encoders (libaom, libjxl, libwebp).
//!
//! AV1 and JPEG store full-range BT.601 YUV, WebP limited-range BT.601 4:2:0 (its format),
//! JPEG XL sRGB.

use std::cell::RefCell;
use std::ffi::CString;

use rsmpeg::avcodec::{AVCodec, AVCodecContext};
use rsmpeg::avutil::{AVDictionary, AVFrame};
use rsmpeg::ffi;

use super::planes::Planes;
use super::{ImageEncoding, JpegEncoder};
use crate::codec::ImageCodec;
use crate::codec::jpeg::{self, Chroma};
use crate::write::open_encoder;

thread_local! {
    /// The last JPEG encoder of this thread and its (quality, chroma).
    static JPEG: RefCell<Option<((u32, Chroma), jpeg::Encoder)>> = const { RefCell::new(None) };
}

pub fn encode(planes: &Planes, enc: &ImageEncoding) -> Result<Vec<u8>, String> {
    match enc.codec {
        ImageCodec::Jpeg => encode_jpeg(planes, enc),
        ImageCodec::Av1 | ImageCodec::Jxl | ImageCodec::Webp => encode_ffmpeg(planes, enc),
        ImageCodec::Png => Err("png is a source format, not a storage codec".into()),
    }
}

fn encode_jpeg(planes: &Planes, enc: &ImageEncoding) -> Result<Vec<u8>, String> {
    let (h, w) = planes.size();
    if enc.jpeg_encoder == JpegEncoder::Mozjpeg {
        let views: Vec<_> = planes.planes.iter().map(|p| p.view()).collect();
        return jpeg::moz::encode(
            &views,
            w as u32,
            h as u32,
            planes.gray,
            enc.quality,
            enc.chroma,
        );
    }
    JPEG.with_borrow_mut(|slot| {
        let key = (enc.quality, enc.chroma);
        if slot.as_ref().is_none_or(|(k, _)| *k != key) {
            *slot = Some((key, jpeg::Encoder::new(enc.quality, enc.chroma)?));
        }
        let (_, encoder) = slot.as_mut().expect("an encoder was just made");
        let views: Vec<_> = planes.planes.iter().map(|p| p.view()).collect();
        encoder.encode(&views, w as u32, h as u32, planes.gray)
    })
}

/// The FFmpeg encoder, input pixel format and options of a storage codec.
fn ffmpeg_encoder(
    enc: &ImageEncoding,
) -> (&'static std::ffi::CStr, i32, Vec<(&'static str, String)>) {
    match enc.codec {
        ImageCodec::Av1 => {
            let format = match enc.chroma {
                Chroma::Yuv420 => ffi::AV_PIX_FMT_YUV420P,
                Chroma::Yuv444 => ffi::AV_PIX_FMT_YUV444P,
            };
            let options = vec![
                ("crf", enc.crf.to_string()),
                ("cpu-used", enc.effort().to_string()),
                ("usage", "allintra".to_string()),
                ("still-picture", "1".to_string()),
            ];
            (c"libaom-av1", format, options)
        }
        ImageCodec::Jxl => {
            let options = vec![
                ("distance", enc.distance.to_string()),
                ("effort", enc.effort().to_string()),
            ];
            (c"libjxl", ffi::AV_PIX_FMT_RGB24, options)
        }
        ImageCodec::Webp => {
            // WebP's own format: limited-range BT.601 4:2:0 (converted in `input_frame`)
            let options = vec![("quality", enc.quality.to_string())];
            (c"libwebp", ffi::AV_PIX_FMT_YUV420P, options)
        }
        ImageCodec::Jpeg | ImageCodec::Png => unreachable!("not encoded through FFmpeg"),
    }
}

fn encode_ffmpeg(planes: &Planes, enc: &ImageEncoding) -> Result<Vec<u8>, String> {
    let (name, format, options) = ffmpeg_encoder(enc);
    let codec = AVCodec::find_encoder_by_name(name)
        .ok_or_else(|| format!("this FFmpeg has no {} encoder", name.to_string_lossy()))?;
    let (h, w) = planes.size();

    let mut context = AVCodecContext::new(&codec);
    context.set_width(w as i32);
    context.set_height(h as i32);
    context.set_pix_fmt(format);
    context.set_time_base(ffi::AVRational { num: 1, den: 1 });
    // SAFETY: plain field writes on an unopened context
    unsafe {
        let raw = context.as_mut_ptr();
        (*raw).thread_count = 1;
        // one encoder per image: its info messages (versions, settings) become verbose ones
        (*raw).log_level_offset = 8;
        match enc.codec {
            ImageCodec::Av1 => {
                // stored like a JPEG: full-range BT.601
                (*raw).color_range = ffi::AVCOL_RANGE_JPEG;
                (*raw).colorspace = ffi::AVCOL_SPC_BT470BG;
            }
            ImageCodec::Jxl => {
                // sRGB
                (*raw).color_range = ffi::AVCOL_RANGE_JPEG;
                (*raw).color_primaries = ffi::AVCOL_PRI_BT709;
                (*raw).color_trc = ffi::AVCOL_TRC_IEC61966_2_1;
                (*raw).colorspace = ffi::AVCOL_SPC_RGB;
            }
            _ => {
                // WebP
                (*raw).color_range = ffi::AVCOL_RANGE_MPEG;
                (*raw).colorspace = ffi::AVCOL_SPC_SMPTE170M;
                (*raw).compression_level = enc.effort() as i32;
            }
        }
    }
    let mut dict: Option<AVDictionary> = None;
    for (key, value) in options {
        let key = CString::new(key).unwrap();
        let value = CString::new(value).unwrap();
        dict = Some(match dict {
            Some(d) => d.set(&key, &value, 0),
            None => AVDictionary::new(&key, &value, 0),
        });
    }
    open_encoder(&mut context, dict.expect("every encoder has options"))?;

    let limited = enc.codec == ImageCodec::Webp;
    let frame = input_frame(planes, format, limited)?;
    let fail = |e: rsmpeg::error::RsmpegError| format!("{}: {e}", name.to_string_lossy());
    context.send_frame(Some(&frame)).map_err(fail)?;
    context.send_frame(None).map_err(fail)?;
    let mut out = Vec::new();
    while let Ok(packet) = context.receive_packet() {
        // SAFETY: a received packet holds `size` bytes at `data`
        out.extend_from_slice(unsafe {
            std::slice::from_raw_parts(packet.data, packet.size as usize)
        });
    }
    if out.is_empty() {
        return Err(format!("{}: no output", name.to_string_lossy()));
    }
    Ok(out)
}

/// Full-range Y / chroma samples -> limited range (16-235 / 16-240), rounded.
fn to_limited(row: &mut [u8], luma: bool) {
    for v in row {
        let x = *v as u32;
        *v = if luma {
            (16 + (x * 219 + 127) / 255) as u8
        } else {
            ((128 * 31 + x * 224 + 127) / 255) as u8
        };
    }
}

/// The encoder's input frame: the planes copied in (planar YUV; `limited`: converted to limited
/// range), or interleaved (RGB24).
fn input_frame(planes: &Planes, format: i32, limited: bool) -> Result<AVFrame, String> {
    let (h, w) = planes.size();
    let mut frame = AVFrame::new();
    frame.set_width(w as i32);
    frame.set_height(h as i32);
    frame.set_format(format);
    frame.set_pts(0);
    frame
        .alloc_buffer()
        .map_err(|e| format!("frame buffer: {e}"))?;

    let plane_mut = |frame: &mut AVFrame, p: usize, rows: usize| {
        let stride = frame.linesize[p] as usize;
        // SAFETY: an allocated frame holds `rows` rows of `stride` bytes in plane p
        let data = unsafe { std::slice::from_raw_parts_mut(frame.data[p], stride * rows) };
        (data, stride)
    };

    if format == ffi::AV_PIX_FMT_YUV420P || format == ffi::AV_PIX_FMT_YUV444P {
        for p in 0..3 {
            let (rows, cols) = if planes.gray {
                let (ch, cw) = if p == 0 || format == ffi::AV_PIX_FMT_YUV444P {
                    (h, w)
                } else {
                    (h.div_ceil(2), w.div_ceil(2))
                };
                (ch, cw)
            } else {
                (planes.planes[p].h, planes.planes[p].w)
            };
            let (data, stride) = plane_mut(&mut frame, p, rows);
            for y in 0..rows {
                let row = &mut data[y * stride..y * stride + cols];
                if planes.gray && p > 0 {
                    row.fill(128);
                } else {
                    row.copy_from_slice(&planes.planes[p].data[y * cols..(y + 1) * cols]);
                    if limited {
                        to_limited(row, p == 0);
                    }
                }
            }
        }
        return Ok(frame);
    }

    let (data, stride) = plane_mut(&mut frame, 0, h);
    let [r, g, b] = [0, 1, 2].map(|c| &planes.planes[c.min(planes.planes.len() - 1)].data);
    for y in 0..h {
        let row = &mut data[y * stride..y * stride + 3 * w];
        let source = y * w..(y + 1) * w;
        let pixels = r[source.clone()]
            .iter()
            .zip(&g[source.clone()])
            .zip(&b[source]);
        for (out, ((&r, &g), &b)) in row.chunks_exact_mut(3).zip(pixels) {
            out.copy_from_slice(&[r, g, b]);
        }
    }
    Ok(frame)
}
