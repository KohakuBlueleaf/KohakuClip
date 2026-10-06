//! Writing images: decode any input, cap its short side (never upscaling; images whose short
//! side is under the floor are dropped), turn it upright, encode it in the storage codec (JPEG
//! through mozjpeg by default; see README for the measurements behind it).

mod planes;
mod source;
mod store;

use crate::codec::ImageCodec;
use crate::codec::jpeg::{self, Chroma};
use planes::{Target, convert, orient};

/// Which library encodes JPEGs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JpegEncoder {
    /// mozjpeg: trellis quantization, baseline scans (the default; 5-8 % smaller than
    /// libjpeg-turbo at the same quality, decoded as fast).
    Mozjpeg,
    /// libjpeg-turbo with optimized Huffman tables (~8x faster to encode).
    Turbo,
}

impl JpegEncoder {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "mozjpeg" => Some(JpegEncoder::Mozjpeg),
            "turbo" => Some(JpegEncoder::Turbo),
            _ => None,
        }
    }
}

/// How images are stored.
#[derive(Clone, Debug)]
pub struct ImageEncoding {
    pub codec: ImageCodec,
    /// The JPEG encoder.
    pub jpeg_encoder: JpegEncoder,
    /// JPEG / WebP quality, 1-100.
    pub quality: u32,
    /// JPEG / AV1 chroma subsampling.
    pub chroma: Chroma,
    /// AV1 (libaom) constant quality, 0-63.
    pub crf: u32,
    /// JPEG XL distance (1.0: visually lossless).
    pub distance: f32,
    /// AV1 cpu-used / JPEG XL effort / WebP method; None: the codec's default (6 / 7 / 4).
    pub effort: Option<u32>,
    /// Short side cap (never upscaled) and floor (smaller images are dropped).
    pub max_short: u32,
    pub min_short: u32,
    /// Let libjpeg-turbo shrink JPEG inputs by 1/2, 1/4 or 1/8 while decoding when the stored
    /// size allows (faster; a different low-pass than the resize, see README).
    pub dct_scale: bool,
}

impl ImageEncoding {
    fn effort(&self) -> u32 {
        self.effort.unwrap_or(match self.codec {
            ImageCodec::Av1 => 6,
            ImageCodec::Jxl => 7,
            _ => 4,
        })
    }
}

/// A stored image: its bytes and upright size, and the input's upright size.
pub struct Encoded {
    pub data: Vec<u8>,
    pub w: u32,
    pub h: u32,
    pub source_w: u32,
    pub source_h: u32,
}

pub enum Outcome {
    Stored(Encoded),
    /// The input's short side is under the floor: not stored. Its upright size.
    TooSmall {
        w: u32,
        h: u32,
    },
}

/// The stored size of an (h, w) input: the short side capped at `max_short`, the long side
/// keeping the aspect ratio.
fn stored_size(h: u32, w: u32, max_short: u32) -> (usize, usize) {
    let short = h.min(w);
    if short <= max_short {
        return (h as usize, w as usize);
    }
    let scale = max_short as f64 / short as f64;
    let side = |x: u32| ((x as f64 * scale).round() as u32).max(max_short) as usize;
    (side(h), side(w))
}

/// How far a JPEG input of (h, w) may be shrunk by the IDCT: the planes keep at least the
/// stored size (no shrinking unless `dct_scale`).
fn dct_shift(h: u32, w: u32, enc: &ImageEncoding) -> u32 {
    if !enc.dct_scale {
        return 0;
    }
    let (sh, sw) = stored_size(h, w, enc.max_short);
    jpeg::max_shift(h, w, sh as f64, sw as f64)
}

/// Encode one input image (the bytes of any image file FFmpeg or libjpeg-turbo reads).
pub fn encode_image(data: &[u8], enc: &ImageEncoding) -> Result<Outcome, String> {
    let target = match enc.codec {
        ImageCodec::Jpeg | ImageCodec::Av1 => Target::Yuv(enc.chroma),
        ImageCodec::Webp => Target::Yuv(Chroma::Yuv420),
        ImageCodec::Jxl => Target::Rgb,
        ImageCodec::Png => return Err("png is a source format, not a storage codec".into()),
    };
    let shrink = |h: u32, w: u32| dct_shift(h, w, enc);
    source::with_picture(data, shrink, |input| {
        let transposed = input.orientation >= 5;
        let upright = |h: u32, w: u32| if transposed { (w, h) } else { (h, w) };
        let (source_h, source_w) = upright(input.h, input.w);
        if input.h.min(input.w) < enc.min_short {
            return Ok(Outcome::TooSmall {
                w: source_w,
                h: source_h,
            });
        }

        let (h, w) = stored_size(input.h, input.w, enc.max_short);
        let mut planes = convert(&input.picture, h, w, target);
        orient(&mut planes, input.orientation);
        let data = store::encode(&planes, enc)?;
        let (h, w) = planes.size();
        Ok(Outcome::Stored(Encoded {
            data,
            w: w as u32,
            h: h as u32,
            source_w,
            source_h,
        }))
    })
}
