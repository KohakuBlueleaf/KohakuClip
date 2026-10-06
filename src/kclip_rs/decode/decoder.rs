//! Per-thread decoder state: one single-threaded decoder per codec, reused buffers.

use std::cell::RefCell;

use rsmpeg::avcodec::{AVCodec, AVCodecContext, AVPacket};
use rsmpeg::avutil::AVFrame;
use rsmpeg::ffi;

use crate::storage::index::Codec;

pub struct Decoders {
    pub contexts: [Option<AVCodecContext>; 3],
    pub packet: AVPacket,
    pub frame: AVFrame,
    pub bytes: Vec<u8>,
    /// A packed picture split into planes, and the resized planes of the rgb mode.
    pub buffers: (Vec<u8>, Vec<u8>),
}

impl Decoders {
    fn new() -> Self {
        Self {
            contexts: [None, None, None],
            packet: AVPacket::new(),
            frame: AVFrame::new(),
            bytes: Vec::new(),
            buffers: (Vec::new(), Vec::new()),
        }
    }
}

thread_local! {
    pub static DECODERS: RefCell<Decoders> = RefCell::new(Decoders::new());
}

/// A single-threaded decoder (parallelism comes from decoding many clips at once).
pub fn open_decoder(codec: Codec) -> Result<AVCodecContext, String> {
    let decoder = match codec {
        Codec::Av1 => AVCodec::find_decoder_by_name(c"libdav1d"),
        Codec::H264 => AVCodec::find_decoder(ffi::AV_CODEC_ID_H264),
        Codec::Hevc => AVCodec::find_decoder(ffi::AV_CODEC_ID_HEVC),
    };
    let decoder = decoder.ok_or_else(|| format!("no {} decoder in this FFmpeg", codec.name()))?;
    let mut context = AVCodecContext::new(&decoder);

    // SAFETY: the context is allocated and not opened yet; priv_data belongs to the decoder
    unsafe {
        let raw = context.as_mut_ptr();
        (*raw).thread_count = 1;
        if codec == Codec::Av1 {
            ffi::av_opt_set_int((*raw).priv_data, c"max_frame_delay".as_ptr(), 1, 0);
            ffi::av_opt_set_int((*raw).priv_data, c"tilethreads".as_ptr(), 1, 0);
        }
    }
    context
        .open(None)
        .map_err(|e| format!("open decoder: {e}"))?;
    Ok(context)
}

/// mp4 H.264 / HEVC samples are length-prefixed NAL units; the decoder takes Annex B: swap each
/// 4-byte length for a start code, in place.
pub fn to_annexb(sample: &mut [u8]) -> bool {
    let mut i = 0;
    while i + 4 <= sample.len() {
        let len = u32::from_be_bytes(sample[i..i + 4].try_into().unwrap()) as usize;
        sample[i..i + 4].copy_from_slice(&[0, 0, 0, 1]);
        i += 4 + len;
        if i > sample.len() {
            return false;
        }
    }
    true
}
