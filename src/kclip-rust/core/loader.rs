use std::fs;
use std::io::Write;
use std::path::Path;

use ffmpeg_next as ffmpeg;
use once_cell::sync::OnceCell;
use tempfile::NamedTempFile;

use crate::core::types::{
    FrameRange, LoadOptions, ResizeFilter, ResizeOptions, VideoBatch, VideoError, VideoMetadata,
    VideoResult,
};

static FFMPEG_INIT: OnceCell<()> = OnceCell::new();

fn ensure_ffmpeg() -> VideoResult<()> {
    FFMPEG_INIT.get_or_try_init(|| {
        ffmpeg::init()?;
        Ok::<(), ffmpeg::Error>(())
    })?;
    Ok(())
}

pub fn load_frames_from_path(path: &Path, options: LoadOptions) -> VideoResult<VideoBatch> {
    ensure_ffmpeg()?;

    let mut ictx = ffmpeg::format::input(&path)?;
    decode_frames(&mut ictx, options)
}

pub fn load_frames_from_bytes(bytes: &[u8], options: LoadOptions) -> VideoResult<VideoBatch> {
    ensure_ffmpeg()?;

    let mut tempfile = NamedTempFile::new()?;
    tempfile.write_all(bytes)?;
    tempfile.flush()?;

    let mut ictx = ffmpeg::format::input(&tempfile.path())?;
    decode_frames(&mut ictx, options)
}

fn decode_frames(
    ictx: &mut ffmpeg::format::context::Input,
    options: LoadOptions,
) -> VideoResult<VideoBatch> {
    let video_stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or(VideoError::StreamNotFound)?
        .index();

    let stream = ictx
        .stream(video_stream)
        .ok_or(VideoError::StreamNotFound)?;
    let parameters = stream.parameters();
    let context = ffmpeg::codec::context::Context::from_parameters(parameters)?;
    let mut decoder = context.decoder().video()?;

    let source_width = decoder.width();
    let source_height = decoder.height();

    let (target_width, target_height) =
        resolve_target_size(source_width, source_height, options.resize)?;

    let mut scaler = ffmpeg::software::scaling::context::Context::get(
        decoder.format(),
        source_width,
        source_height,
        ffmpeg::format::Pixel::RGB24,
        target_width,
        target_height,
        map_filter(options.resize.map(|r| r.filter)),
    )?;

    let mut decoded = ffmpeg::frame::Video::empty();
    let mut rgb_frame = ffmpeg::frame::Video::empty();
    rgb_frame.set_format(ffmpeg::format::Pixel::RGB24);
    rgb_frame.set_format(ffmpeg::format::Pixel::RGB24);
    rgb_frame.set_width(target_width);
    rgb_frame.set_height(target_height);

    let mut frames = Vec::new();
    let channels = VideoBatch::CHANNELS;
    let mut produced_frames = 0usize;
    let mut range_cursor = FrameRangeCursor::new(options.frame_range);

    for (stream, packet) in ictx.packets() {
        if stream.index() == video_stream {
            decoder.send_packet(&packet)?;
            let drain = receive_frames(
                &mut decoder,
                &mut scaler,
                &mut decoded,
                &mut rgb_frame,
                &mut frames,
                &mut range_cursor,
            )?;
            produced_frames += drain.produced;
            if drain.finished {
                break;
            }
        }
    }

    if !range_cursor.finished() {
        decoder.send_eof()?;
        let drain = receive_frames(
            &mut decoder,
            &mut scaler,
            &mut decoded,
            &mut rgb_frame,
            &mut frames,
            &mut range_cursor,
        )?;
        produced_frames += drain.produced;
    }

    Ok(VideoBatch {
        frames,
        frame_count: produced_frames,
        width: target_width as usize,
        height: target_height as usize,
        channels,
    })
}

pub fn inspect_video_from_path(path: &Path) -> VideoResult<VideoMetadata> {
    ensure_ffmpeg()?;

    let file_size = fs::metadata(path)?.len();
    let mut ictx = ffmpeg::format::input(path)?;
    extract_video_metadata(&mut ictx, file_size)
}

pub fn inspect_video_from_bytes(bytes: &[u8]) -> VideoResult<VideoMetadata> {
    ensure_ffmpeg()?;

    let mut tempfile = NamedTempFile::new()?;
    tempfile.write_all(bytes)?;
    tempfile.flush()?;

    let file_size = bytes.len() as u64;
    let mut ictx = ffmpeg::format::input(tempfile.path())?;
    extract_video_metadata(&mut ictx, file_size)
}

struct DrainResult {
    produced: usize,
    finished: bool,
}

fn receive_frames(
    decoder: &mut ffmpeg::decoder::Video,
    scaler: &mut ffmpeg::software::scaling::context::Context,
    decoded_frame: &mut ffmpeg::frame::Video,
    rgb_frame: &mut ffmpeg::frame::Video,
    frames: &mut Vec<u8>,
    range_cursor: &mut FrameRangeCursor,
) -> VideoResult<DrainResult> {
    let mut produced = 0usize;
    loop {
        match decoder.receive_frame(decoded_frame) {
            Ok(_) => match range_cursor.next_action() {
                RangeAction::Skip => continue,
                RangeAction::Take => {
                    scaler.run(decoded_frame, rgb_frame)?;
                    push_rgb(rgb_frame, frames)?;
                    produced += 1;
                }
                RangeAction::Stop => return Ok(DrainResult { produced, finished: true }),
            },
            Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::util::error::EAGAIN => break,
            Err(ffmpeg::Error::Eof) => break,
            Err(err) => return Err(VideoError::Decode(err.to_string())),
        }
    }
    Ok(DrainResult { produced, finished: range_cursor.finished() })
}

fn push_rgb(frame: &ffmpeg::frame::Video, target: &mut Vec<u8>) -> VideoResult<()> {
    let height = frame.height() as usize;
    let width = frame.width() as usize;
    let channels = VideoBatch::CHANNELS;
    let stride = frame.stride(0);
    let data_plane = frame.data(0);

    for row in 0..height {
        let start = row * stride;
        let end = start + width * channels;
        let row_slice = &data_plane[start..end];
        target.extend_from_slice(row_slice);
    }

    Ok(())
}

fn resolve_target_size(
    src_width: u32,
    src_height: u32,
    resize: Option<ResizeOptions>,
) -> VideoResult<(u32, u32)> {
    match resize {
        Some(resize) => Ok((resize.width.get(), resize.height.get())),
        None => {
            if src_width == 0 || src_height == 0 {
                return Err(VideoError::InvalidResize(
                    "source dimensions reported as zero".to_string(),
                ));
            }
            Ok((src_width, src_height))
        }
    }
}

fn map_filter(filter: Option<ResizeFilter>) -> ffmpeg::software::scaling::flag::Flags {
    match filter.unwrap_or_default() {
        ResizeFilter::Bilinear => ffmpeg::software::scaling::flag::Flags::BILINEAR,
    }
}

struct FrameRangeCursor {
    spec: FrameRange,
    current: usize,
    finished: bool,
}

impl FrameRangeCursor {
    fn new(spec: FrameRange) -> Self {
        Self { spec, current: 0, finished: false }
    }

    fn next_action(&mut self) -> RangeAction {
        if self.finished {
            return RangeAction::Stop;
        }

        if let Some(end) = self.spec.end {
            if self.current >= end {
                self.finished = true;
                return RangeAction::Stop;
            }
        }

        let idx = self.current;
        self.current += 1;

        if idx < self.spec.start {
            return RangeAction::Skip;
        }

        let offset = idx - self.spec.start;
        if offset.is_multiple_of(self.spec.step.get()) {
            RangeAction::Take
        } else {
            RangeAction::Skip
        }
    }

    fn finished(&self) -> bool {
        self.finished
    }
}

enum RangeAction {
    Skip,
    Take,
    Stop,
}

fn extract_video_metadata(
    ictx: &mut ffmpeg::format::context::Input,
    file_size: u64,
) -> VideoResult<VideoMetadata> {
    let stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or(VideoError::StreamNotFound)?;

    let parameters = stream.parameters();
    let (width, height) = read_dimensions(&parameters)?;
    let bit_rate = read_bit_rate(&parameters, ictx);
    let frame_count = estimate_frame_count(&stream, ictx.duration());
    let format_label = describe_format(ictx);

    Ok(VideoMetadata { format: format_label, file_size, frame_count, width, height, bit_rate })
}

fn read_dimensions(parameters: &ffmpeg::codec::Parameters) -> VideoResult<(u32, u32)> {
    unsafe {
        let width = (*parameters.as_ptr()).width;
        let height = (*parameters.as_ptr()).height;
        if width <= 0 || height <= 0 {
            return Err(VideoError::Metadata(
                "source reported non-positive dimensions".to_string(),
            ));
        }
        Ok((width as u32, height as u32))
    }
}

fn read_bit_rate(
    parameters: &ffmpeg::codec::Parameters,
    ictx: &ffmpeg::format::context::Input,
) -> Option<u64> {
    unsafe {
        let value = (*parameters.as_ptr()).bit_rate;
        if value > 0 {
            return Some(value as u64);
        }
    }
    let ctx_rate = ictx.bit_rate();
    if ctx_rate > 0 {
        Some(ctx_rate as u64)
    } else {
        None
    }
}

fn describe_format(ictx: &ffmpeg::format::context::Input) -> String {
    let input = ictx.format();
    let description = input.description().trim();
    if description.is_empty() {
        input.name().to_string()
    } else {
        description.to_string()
    }
}

fn estimate_frame_count(stream: &ffmpeg::Stream, container_duration: i64) -> Option<u64> {
    let nb_frames = stream.frames();
    if nb_frames > 0 {
        return Some(nb_frames as u64);
    }

    let fps = rational_to_positive_f64(stream.avg_frame_rate())
        .or_else(|| rational_to_positive_f64(stream.rate()))?;

    if let Some(seconds) = stream_duration_seconds(stream) {
        if let Some(count) = frames_from_seconds(seconds, fps) {
            return Some(count);
        }
    }

    if container_duration > 0 && container_duration != ffmpeg::ffi::AV_NOPTS_VALUE {
        let seconds = container_duration as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE);
        return frames_from_seconds(seconds, fps);
    }

    None
}

fn stream_duration_seconds(stream: &ffmpeg::Stream) -> Option<f64> {
    let duration = stream.duration();
    if duration <= 0 || duration == ffmpeg::ffi::AV_NOPTS_VALUE {
        return None;
    }
    let time_base = rational_to_positive_f64(stream.time_base())?;
    let seconds = (duration as f64) * time_base;
    if seconds > 0.0 {
        Some(seconds)
    } else {
        None
    }
}

fn rational_to_positive_f64(value: ffmpeg::Rational) -> Option<f64> {
    let num = value.numerator();
    let den = value.denominator();
    if num <= 0 || den <= 0 {
        None
    } else {
        Some(num as f64 / den as f64)
    }
}

fn frames_from_seconds(seconds: f64, fps: f64) -> Option<u64> {
    if seconds <= 0.0 || !seconds.is_finite() || fps <= 0.0 || !fps.is_finite() {
        return None;
    }
    let frames = (seconds * fps).round();
    if frames.is_finite() && frames > 0.0 {
        Some(frames as u64)
    } else {
        None
    }
}
