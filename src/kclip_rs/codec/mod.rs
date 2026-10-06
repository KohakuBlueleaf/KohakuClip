//! Still-image codecs, each through its own library's fastest decoding path: JPEG through
//! libjpeg-turbo, AV1 (intra, raw OBUs) through libdav1d, JPEG XL through libjxl, WebP through
//! libwebp; PNG through the linked FFmpeg.

mod dav1d;
pub mod jpeg;
mod jxl;
mod png;
mod webp;

use serde::{Deserialize, Serialize};

use crate::image::Picture;

/// How an image is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[repr(u8)]
pub enum ImageCodec {
    Jpeg = 1,
    /// One AV1 temporal unit (sequence header + intra frame), as raw OBUs.
    Av1 = 2,
    Jxl = 3,
    Webp = 4,
    Png = 5,
}

impl ImageCodec {
    pub const ALL: [ImageCodec; 5] = [
        ImageCodec::Jpeg,
        ImageCodec::Av1,
        ImageCodec::Jxl,
        ImageCodec::Webp,
        ImageCodec::Png,
    ];

    pub fn name(self) -> &'static str {
        match self {
            ImageCodec::Jpeg => "jpeg",
            ImageCodec::Av1 => "av1",
            ImageCodec::Jxl => "jxl",
            ImageCodec::Webp => "webp",
            ImageCodec::Png => "png",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.name() == name)
    }

    pub fn from_byte(byte: u8) -> Option<Self> {
        Self::ALL.into_iter().find(|&c| c as u8 == byte)
    }

    /// The codec of a file name's extension (sources without an index).
    pub fn from_extension(name: &str) -> Option<Self> {
        let ext = name.rsplit_once('.')?.1.to_ascii_lowercase();
        match ext.as_str() {
            "jpg" | "jpeg" => Some(ImageCodec::Jpeg),
            "obu" => Some(ImageCodec::Av1),
            "jxl" => Some(ImageCodec::Jxl),
            "webp" => Some(ImageCodec::Webp),
            "png" => Some(ImageCodec::Png),
            _ => None,
        }
    }
}

/// Decode one stored non-JPEG image and run `f` on its planes (JPEG goes through
/// `jpeg::Decoder`, which can also shrink in the IDCT).
pub fn decode_still<T>(
    codec: ImageCodec,
    data: &[u8],
    f: impl FnOnce(&Picture) -> Result<T, String>,
) -> Result<T, String> {
    match codec {
        ImageCodec::Av1 => dav1d::decode(data, f),
        ImageCodec::Jxl => jxl::decode(data, f),
        ImageCodec::Webp => webp::decode(data, f),
        ImageCodec::Png => png::decode(data, f),
        ImageCodec::Jpeg => Err("jpeg: decoded through jpeg::Decoder".into()),
    }
}
