//! Decoding a writer's input image: JPEG through libjpeg-turbo (planes as stored, EXIF
//! orientation read), anything else FFmpeg reads (PNG, WebP, GIF, BMP, TIFF, AVIF, JPEG XL, ...)
//! through its demuxers and decoders, converted to RGB with alpha composited onto white.

use std::sync::{Arc, Mutex};

use rsmpeg::avcodec::AVCodecContext;
use rsmpeg::avformat::{AVFormatContextInput, AVIOContextContainer};
use rsmpeg::avutil::AVFrame;
use rsmpeg::ffi;
use rsmpeg::swscale::SwsContext;

use crate::codec::jpeg;
use crate::image::{Color, Picture, Plane};
use crate::write::memio::{MemFile, custom_io};

const JPEG_MAGIC: [u8; 3] = [0xFF, 0xD8, 0xFF];

/// A decoded input: what `with_picture` hands over.
pub struct Input<'a> {
    pub picture: Picture<'a>,
    /// Stored size of the input before any DCT shrinking (in the file's orientation).
    pub h: u32,
    pub w: u32,
    /// EXIF orientation (1-8; 1 for everything but JPEG).
    pub orientation: u8,
}

/// Decode `data` and run `f` on the picture. `shrink(h, w)` says how far a JPEG may be shrunk
/// by the IDCT (1 / 2^shift) given its stored size.
pub fn with_picture<T>(
    data: &[u8],
    shrink: impl Fn(u32, u32) -> u32,
    f: impl FnOnce(Input) -> Result<T, String>,
) -> Result<T, String> {
    if data.starts_with(&JPEG_MAGIC) {
        let mut decoder = jpeg::Decoder::new()?;
        // JPEGs libjpeg-turbo refuses (CMYK, 12-bit, lossless) go through FFmpeg
        if let Ok(header) = decoder.header(data) {
            let shift = shrink(header.h, header.w);
            let picture = decoder.decode(data, shift)?;
            return f(Input {
                picture,
                h: header.h,
                w: header.w,
                orientation: jpeg::orientation(data),
            });
        }
    }
    let frame = decode_file(data)?;
    let mut rgb = Vec::new();
    let picture = to_rgb_planes(&frame, &mut rgb)?;
    f(Input {
        picture,
        h: frame.height as u32,
        w: frame.width as u32,
        orientation: 1,
    })
}

/// The first picture of an image file, through FFmpeg's demuxers (probed from the bytes).
fn decode_file(data: &[u8]) -> Result<AVFrame, String> {
    let file = Arc::new(Mutex::new(MemFile {
        data: data.to_vec(),
        pos: 0,
    }));
    let io = AVIOContextContainer::Custom(custom_io(file, false));
    let mut input =
        AVFormatContextInput::from_io_context(io).map_err(|e| format!("open image: {e}"))?;
    let (stream_index, decoder) = input
        .find_best_stream(ffi::AVMEDIA_TYPE_VIDEO)
        .map_err(|e| format!("find image stream: {e}"))?
        .ok_or("no image stream")?;

    let mut context = AVCodecContext::new(&decoder);
    context
        .apply_codecpar(&input.streams()[stream_index].codecpar())
        .map_err(|e| format!("decoder parameters: {e}"))?;
    // SAFETY: plain field write on an unopened context
    unsafe { (*context.as_mut_ptr()).thread_count = 1 };
    context
        .open(None)
        .map_err(|e| format!("open image decoder: {e}"))?;

    while let Some(packet) = input
        .read_packet()
        .map_err(|e| format!("read image: {e}"))?
    {
        if packet.stream_index as usize != stream_index {
            continue;
        }
        context
            .send_packet(Some(&packet))
            .map_err(|e| format!("decode image: {e}"))?;
        if let Ok(frame) = context.receive_frame() {
            return Ok(frame);
        }
    }
    context
        .send_packet(None)
        .map_err(|e| format!("decode image: {e}"))?;
    context
        .receive_frame()
        .map_err(|_| "no picture in the image".to_string())
}

/// Any decoded frame -> R, G, B planes in `buffer` (alpha composited onto white), through
/// swscale with the frame's own colorspace and range.
fn to_rgb_planes<'a>(frame: &AVFrame, buffer: &'a mut Vec<u8>) -> Result<Picture<'a>, String> {
    let (h, w) = (frame.height as usize, frame.width as usize);
    let mut sws = SwsContext::get_context(
        frame.width,
        frame.height,
        frame.format,
        frame.width,
        frame.height,
        ffi::AV_PIX_FMT_RGBA,
        ffi::SWS_BICUBIC | ffi::SWS_ACCURATE_RND | ffi::SWS_FULL_CHR_H_INT,
        None,
        None,
        None,
    )
    .ok_or("cannot convert this pixel format")?;
    // SAFETY: the context is valid; the coefficient tables are static
    unsafe {
        let full = (frame.color_range == ffi::AVCOL_RANGE_JPEG) as i32;
        let source = ffi::sws_getCoefficients(frame.colorspace as i32);
        let target = ffi::sws_getCoefficients(ffi::SWS_CS_DEFAULT as i32);
        let (brightness, contrast, saturation) = (0, 1 << 16, 1 << 16);
        ffi::sws_setColorspaceDetails(
            sws.as_mut_ptr(),
            source,
            full,
            target,
            1,
            brightness,
            contrast,
            saturation,
        );
    }
    let mut rgba = AVFrame::new();
    rgba.set_width(frame.width);
    rgba.set_height(frame.height);
    rgba.set_format(ffi::AV_PIX_FMT_RGBA);
    rgba.alloc_buffer()
        .map_err(|e| format!("frame buffer: {e}"))?;
    sws.scale_frame(frame, 0, frame.height, &mut rgba)
        .map_err(|e| format!("convert to RGB: {e}"))?;

    buffer.resize(3 * h * w, 0);
    let (r, rest) = buffer.split_at_mut(h * w);
    let (g, b) = rest.split_at_mut(h * w);
    let stride = rgba.linesize[0] as usize;
    // SAFETY: an RGBA frame holds h rows of `stride` bytes
    let src = unsafe { std::slice::from_raw_parts(rgba.data[0], stride * (h - 1) + 4 * w) };
    for y in 0..h {
        let row = &src[y * stride..y * stride + 4 * w];
        let out = y * w..(y + 1) * w;
        let outputs = r[out.clone()]
            .iter_mut()
            .zip(g[out.clone()].iter_mut())
            .zip(b[out].iter_mut());
        for (pixel, ((r, g), b)) in row.as_chunks::<4>().0.iter().zip(outputs) {
            let alpha = pixel[3] as u32;
            // c * a + white * (1 - a), rounded
            let over_white = |c: u8| ((c as u32 * alpha + 255 * (255 - alpha) + 127) / 255) as u8;
            *r = over_white(pixel[0]);
            *g = over_white(pixel[1]);
            *b = over_white(pixel[2]);
        }
    }

    let plane = |data: &'a [u8]| Plane {
        data,
        stride: w,
        h,
        w,
    };
    let (r, rest) = buffer.split_at(h * w);
    let (g, b) = rest.split_at(h * w);
    Ok(Picture {
        planes: [plane(r), plane(g), plane(b)],
        color: Color::Rgb,
    })
}
