//! PNG images through the linked FFmpeg: one packet in, one frame out, on a persistent
//! single-threaded decoder per thread. The packet is a copy of the image followed by FFmpeg's
//! input padding (`AV_INPUT_BUFFER_PADDING_SIZE` zero bytes).

use std::cell::RefCell;

use rsmpeg::avcodec::{AVCodec, AVCodecContext, AVPacket};
use rsmpeg::avutil::AVFrame;
use rsmpeg::ffi;

use crate::decode::picture;
use crate::image::Picture;

struct PngDecoder {
    context: Option<AVCodecContext>,
    packet: AVPacket,
    frame: AVFrame,
    /// The image's bytes and the input padding.
    input: Vec<u8>,
    /// RGB(A) frames split into planes.
    planes: Vec<u8>,
}

thread_local! {
    static DECODER: RefCell<PngDecoder> = RefCell::new(PngDecoder {
        context: None,
        packet: AVPacket::new(),
        frame: AVFrame::new(),
        input: Vec::new(),
        planes: Vec::new(),
    });
}

/// A single-threaded FFmpeg PNG decoder.
fn open() -> Result<AVCodecContext, String> {
    let decoder = AVCodec::find_decoder(ffi::AV_CODEC_ID_PNG)
        .ok_or_else(|| "no png decoder in this FFmpeg".to_string())?;
    let mut context = AVCodecContext::new(&decoder);
    // SAFETY: the context is allocated and not opened yet
    unsafe { (*context.as_mut_ptr()).thread_count = 1 };
    context
        .open(None)
        .map_err(|e| format!("open png decoder: {e}"))?;
    Ok(context)
}

/// Decode one PNG image (the whole of `data` is one packet) and run `f` on its planes.
pub fn decode<T>(data: &[u8], f: impl FnOnce(&Picture) -> Result<T, String>) -> Result<T, String> {
    DECODER.with_borrow_mut(|d| {
        let context = match &mut d.context {
            Some(context) => context,
            empty => empty.insert(open()?),
        };
        let fail = |e: rsmpeg::error::RsmpegError| format!("png: {e}");

        // SAFETY: the context is open; the previous image ended with a flush (end of stream)
        unsafe { ffi::avcodec_flush_buffers(context.as_mut_ptr()) };
        d.input.clear();
        d.input.extend_from_slice(data);
        d.input
            .resize(data.len() + ffi::AV_INPUT_BUFFER_PADDING_SIZE as usize, 0);
        // SAFETY: the packet borrows `input` (image + padding) for the send call; not
        // reference counted, so the decoder copies what it keeps
        unsafe {
            let raw = d.packet.as_mut_ptr();
            (*raw).data = d.input.as_mut_ptr();
            (*raw).size = data.len() as i32;
        }
        context.send_packet(Some(&d.packet)).map_err(fail)?;
        context.send_packet(None).map_err(fail)?;

        // SAFETY: both are valid; receive_frame unreferences `frame` before filling it
        let got = unsafe { ffi::avcodec_receive_frame(context.as_mut_ptr(), d.frame.as_mut_ptr()) };
        if got < 0 {
            return Err("png: no picture decoded".into());
        }
        let decoded = picture(&d.frame, &mut d.planes)?;
        f(&decoded)
    })
}
