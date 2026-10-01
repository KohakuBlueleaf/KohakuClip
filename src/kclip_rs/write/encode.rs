//! Re-encoding one video: demux, decode, scale, encode, mux into memory.

use std::ffi::CString;
use std::path::Path;
use std::sync::{Arc, Mutex};

use rsmpeg::avcodec::{AVCodec, AVCodecContext};
use rsmpeg::avformat::{AVFormatContextInput, AVFormatContextOutput, AVIOContextContainer};
use rsmpeg::avutil::AVFrame;
use rsmpeg::ffi;
use rsmpeg::swscale::SwsContext;

use super::Encoding;
use super::memio::{MemFile, custom_io};
use crate::mp4;

/// A video to encode: a path FFmpeg can open, or the bytes of a video file.
pub enum Source<'a> {
    Path(&'a Path),
    Bytes(Vec<u8>),
}

fn open_input(source: Source) -> Result<AVFormatContextInput, String> {
    let opened = match source {
        Source::Path(path) => {
            let url = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
            AVFormatContextInput::open(&url)
        }
        Source::Bytes(data) => {
            let file = Arc::new(Mutex::new(MemFile { data, pos: 0 }));
            AVFormatContextInput::from_io_context(AVIOContextContainer::Custom(custom_io(
                file, false,
            )))
        }
    };
    opened.map_err(|e| format!("open input: {e}"))
}

/// The color description written to the output: the source's, or for an untagged source the
/// usual guess by resolution (HD: BT.709, SD: BT.601), so readers never see "unspecified".
struct Color {
    matrix: ffi::AVColorSpace,
    primaries: ffi::AVColorPrimaries,
    transfer: ffi::AVColorTransferCharacteristic,
}

impl Color {
    fn of(dec: &AVCodecContext) -> Self {
        let hd = dec.height.min(dec.width) >= 720;
        let pick = |tagged: u32, unspecified: u32, hd_value: u32, sd_value: u32| {
            if tagged != unspecified {
                tagged
            } else if hd {
                hd_value
            } else {
                sd_value
            }
        };
        Self {
            matrix: pick(
                dec.colorspace,
                ffi::AVCOL_SPC_UNSPECIFIED,
                ffi::AVCOL_SPC_BT709,
                ffi::AVCOL_SPC_SMPTE170M,
            ),
            primaries: pick(
                dec.color_primaries,
                ffi::AVCOL_PRI_UNSPECIFIED,
                ffi::AVCOL_PRI_BT709,
                ffi::AVCOL_PRI_SMPTE170M,
            ),
            transfer: pick(
                dec.color_trc,
                ffi::AVCOL_TRC_UNSPECIFIED,
                ffi::AVCOL_TRC_BT709,
                ffi::AVCOL_TRC_SMPTE170M,
            ),
        }
    }
}

/// Output size: the short side capped at `max_short`, the long side keeping the aspect ratio,
/// rounded to an even number (as ffmpeg's `scale=...:-2`).
fn output_size(h: i32, w: i32, max_short: u32) -> (i32, i32) {
    let short = h.min(w).min(max_short as i32);
    let long_of = |long: i32, short_in: i32| {
        let exact = short as f64 * long as f64 / (short_in as f64 * 2.0);
        (exact.round() as i32) * 2
    };
    if h <= w {
        (short, long_of(w, h))
    } else {
        (long_of(h, w), short)
    }
}

/// Error mapper: "what: ffmpeg error".
fn err(what: &'static str) -> impl Fn(rsmpeg::error::RsmpegError) -> String {
    move |e| format!("{what}: {e}")
}

/// Re-encode `source` (first video stream) to a faststart mp4 in memory.
pub fn encode(source: Source, enc: &Encoding) -> Result<Vec<u8>, String> {
    let mut input = open_input(source)?;
    let (stream_index, decoder) = input
        .find_best_stream(ffi::AVMEDIA_TYPE_VIDEO)
        .map_err(err("find video stream"))?
        .ok_or("no video stream")?;

    // decoder
    let stream = &input.streams()[stream_index];
    let time_base = stream.time_base;
    let frame_rate = stream.guess_framerate().unwrap_or(stream.avg_frame_rate);
    let mut dec = AVCodecContext::new(&decoder);
    dec.apply_codecpar(&stream.codecpar())
        .map_err(err("decoder parameters"))?;
    dec.set_pkt_timebase(time_base);
    // SAFETY: plain field write on an unopened context
    unsafe { (*dec.as_mut_ptr()).thread_count = 1 };
    dec.open(None).map_err(err("open decoder"))?;
    let (h, w) = output_size(dec.height, dec.width, enc.max_short);

    // muxer into memory (not faststart: the mov muxer re-opens its output by name for that, so
    // the moov is moved to the front afterwards, in `mp4::faststart`)
    let out_file = Arc::new(Mutex::new(MemFile::default()));
    let mut muxer = AVFormatContextOutput::builder()
        .format_name(c"mp4")
        .io_context(AVIOContextContainer::Custom(custom_io(
            out_file.clone(),
            true,
        )))
        .build()
        .map_err(err("open muxer"))?;

    // encoder
    let (name, options) = enc.encoder()?;
    let encoder = AVCodec::find_encoder_by_name(name)
        .ok_or_else(|| format!("this FFmpeg has no {} encoder", name.to_string_lossy()))?;
    let mut encoder_ctx = AVCodecContext::new(&encoder);
    encoder_ctx.set_width(w);
    encoder_ctx.set_height(h);
    encoder_ctx.set_pix_fmt(ffi::AV_PIX_FMT_YUV420P);
    encoder_ctx.set_time_base(time_base);
    encoder_ctx.set_framerate(frame_rate);
    encoder_ctx.set_gop_size(enc.gop as i32);
    encoder_ctx.set_max_b_frames(0);
    if muxer.oformat().flags & ffi::AVFMT_GLOBALHEADER as i32 != 0 {
        encoder_ctx.set_flags(encoder_ctx.flags | ffi::AV_CODEC_FLAG_GLOBAL_HEADER as i32);
    }
    let color = Color::of(&dec);
    // SAFETY: plain field writes on an unopened context
    unsafe {
        let raw = encoder_ctx.as_mut_ptr();
        (*raw).thread_count = 1;
        (*raw).color_range = ffi::AVCOL_RANGE_MPEG;
        (*raw).colorspace = color.matrix;
        (*raw).color_primaries = color.primaries;
        (*raw).color_trc = color.transfer;
    }
    encoder_ctx
        .open(Some(options))
        .map_err(err("open encoder"))?;

    let out_index = {
        let mut out_stream = muxer.new_stream();
        out_stream.set_codecpar(encoder_ctx.extract_codecpar());
        out_stream.set_time_base(time_base);
        out_stream.index as usize
    };
    muxer.write_header(&mut None).map_err(err("write header"))?;
    let out_time_base = muxer.streams()[out_index].time_base;

    // decode -> scale -> encode -> mux
    let mut scaler: Option<SwsContext> = None;
    let mut scaled = AVFrame::new();
    scaled.set_width(w);
    scaled.set_height(h);
    scaled.set_format(ffi::AV_PIX_FMT_YUV420P);
    scaled.alloc_buffer().map_err(err("frame buffer"))?;
    let mut last_pts = i64::MIN;
    // one frame in the stream time base: the duration of packets the encoder leaves unset
    // (the mp4 muxer would give the last sample a zero duration, shifting the fps)
    // SAFETY: pure arithmetic
    let frame_ticks = unsafe { ffi::av_rescale_q(1, ffi::av_inv_q(frame_rate), time_base) };
    let write_packets = |encoder_ctx: &mut AVCodecContext, muxer: &mut AVFormatContextOutput| {
        while let Ok(mut packet) = encoder_ctx.receive_packet() {
            if packet.duration == 0 {
                packet.set_duration(frame_ticks);
            }
            packet.rescale_ts(time_base, out_time_base);
            packet.set_stream_index(out_index as i32);
            muxer
                .interleaved_write_frame(&mut packet)
                .map_err(err("write packet"))?;
        }
        Ok::<_, String>(())
    };

    let mut handle = |frame: Option<AVFrame>,
                      encoder_ctx: &mut AVCodecContext,
                      muxer: &mut AVFormatContextOutput|
     -> Result<(), String> {
        let Some(frame) = frame else {
            encoder_ctx.send_frame(None).map_err(err("flush encoder"))?;
            return write_packets(encoder_ctx, muxer);
        };
        // timestamps: the frame's, else one frame after the last; drop non-increasing ones
        let pts = match frame.best_effort_timestamp {
            i64::MIN => last_pts.saturating_add(1),
            pts => pts,
        };
        if pts <= last_pts {
            return Ok(());
        }
        last_pts = pts;

        let sws = match &mut scaler {
            Some(sws) => sws,
            empty => empty.insert(
                SwsContext::get_context(
                    frame.width,
                    frame.height,
                    frame.format,
                    w,
                    h,
                    ffi::AV_PIX_FMT_YUV420P,
                    ffi::SWS_AREA,
                    None,
                    None,
                    None,
                )
                .ok_or("cannot scale this pixel format")?,
            ),
        };
        scaled.make_writable().map_err(err("frame buffer"))?;
        sws.scale_frame(&frame, 0, frame.height, &mut scaled)
            .map_err(err("scale"))?;
        scaled.set_pts(pts);
        encoder_ctx
            .send_frame(Some(&scaled))
            .map_err(err("encode"))?;
        write_packets(encoder_ctx, muxer)
    };

    while let Some(packet) = input.read_packet().map_err(err("read packet"))? {
        if packet.stream_index as usize != stream_index {
            continue;
        }
        dec.send_packet(Some(&packet)).map_err(err("decode"))?;
        while let Ok(frame) = dec.receive_frame() {
            handle(Some(frame), &mut encoder_ctx, &mut muxer)?;
        }
    }
    dec.send_packet(None).map_err(err("flush decoder"))?;
    while let Ok(frame) = dec.receive_frame() {
        handle(Some(frame), &mut encoder_ctx, &mut muxer)?;
    }
    handle(None, &mut encoder_ctx, &mut muxer)?;
    muxer.write_trailer().map_err(err("write trailer"))?;
    drop(muxer);

    let data = std::mem::take(&mut out_file.lock().unwrap().data);
    mp4::faststart(data).map_err(|e| e.to_string())
}
