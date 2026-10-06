//! Writing shards: re-encode any video (a file or bytes in memory) to the storage format with the
//! linked FFmpeg libraries, then pack mp4 files into a zip / tar shard with its index; or encode
//! images (`image`) and pack them with an image index.
//!
//! Storage format (defaults): native fps, short side capped at 512 (never upscaled), closed GOPs
//! of 16, AV1 (SVT-AV1) with the in-loop filters off for faster decoding, faststart mp4.

mod av1;
mod encode;
mod image;
mod memio;
mod pack;

use std::ffi::{CStr, CString};

use rsmpeg::avcodec::AVCodecContext;
use rsmpeg::avutil::AVDictionary;
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;

pub use encode::{Source, encode};
pub use image::{ImageEncoding, JpegEncoder, Outcome, encode_image};
pub use pack::{ImageMember, pack, pack_images};

/// How videos are stored.
#[derive(Clone, Debug)]
pub struct Encoding {
    /// "av1" (libsvtav1), "h264" (libx264) or "hevc" (libx265).
    pub codec: String,
    pub crf: u32,
    /// Frames per closed GOP.
    pub gop: u32,
    /// SVT-AV1 preset (x264 / x265 use "medium").
    pub preset: u32,
    /// Short side cap (never upscaled).
    pub max_short: u32,
    /// AV1 deblocking / CDEF / loop restoration.
    pub loop_filters: bool,
}

impl Encoding {
    fn encoder(&self) -> Result<(&'static CStr, AVDictionary), String> {
        let gop = self.gop;
        let dict = |key: &str, value: String| {
            let key = CString::new(key).unwrap();
            let value = CString::new(value).unwrap();
            (key, value)
        };
        let options: Vec<(CString, CString)>;
        let name = match self.codec.as_str() {
            "av1" => {
                let lf = self.loop_filters as u32;
                let filters = format!("enable-dlf={lf}:enable-cdef={lf}:enable-restoration={lf}");
                let params = format!("{filters}:scd=0:irefresh-type=2:lp=1");
                options = vec![
                    dict("crf", self.crf.to_string()),
                    dict("preset", self.preset.to_string()),
                    dict("svtav1-params", params),
                ];
                c"libsvtav1"
            }
            "h264" => {
                let params =
                    format!("keyint={gop}:min-keyint={gop}:scenecut=0:bframes=0:threads=1");
                options = vec![
                    dict("crf", self.crf.to_string()),
                    dict("preset", "medium".into()),
                    dict("tune", "fastdecode".into()),
                    dict("x264-params", params),
                ];
                c"libx264"
            }
            "hevc" => {
                let params = format!(
                    "keyint={gop}:min-keyint={gop}:scenecut=0:bframes=0:frame-threads=1:pools=none"
                );
                options = vec![
                    dict("crf", self.crf.to_string()),
                    dict("preset", "medium".into()),
                    dict("x265-params", params),
                ];
                c"libx265"
            }
            other => return Err(format!("unknown codec {other:?} (av1, h264, hevc)")),
        };

        let mut options = options.into_iter();
        let (key, value) = options.next().expect("at least one option");
        let mut dict = AVDictionary::new(&key, &value, 0);
        for (key, value) in options {
            dict = dict.set(&key, &value, 0);
        }
        Ok((name, dict))
    }
}

/// Open an encoder with options. `avcodec_open2` frees the options dictionary it is given and
/// hands back one of the unused entries, also when opening fails; rsmpeg's `open` frees the
/// original again on failure (a double free, e.g. when SVT-AV1 rejects a parameter), so the
/// dictionary is owned here instead.
pub(crate) fn open_encoder(
    context: &mut AVCodecContext,
    options: AVDictionary,
) -> Result<(), String> {
    let mut dict = options.into_raw().as_ptr();
    // SAFETY: the context is allocated and not opened; `dict` is a valid dictionary that
    // avcodec_open2 replaces in place
    let ret = unsafe { ffi::avcodec_open2(context.as_mut_ptr(), std::ptr::null(), &mut dict) };
    // SAFETY: whatever avcodec_open2 left in `dict` (unused entries, or null) belongs to us
    unsafe { ffi::av_dict_free(&mut dict) };
    if ret < 0 {
        return Err(format!("open encoder: {}", RsmpegError::AVError(ret)));
    }
    Ok(())
}
