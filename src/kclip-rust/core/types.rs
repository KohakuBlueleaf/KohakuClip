use std::num::{NonZeroU32, NonZeroUsize};

use ffmpeg_next as ffmpeg;
use thiserror::Error;

pub type VideoResult<T> = Result<T, VideoError>;

#[derive(Debug, Error)]
pub enum VideoError {
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("no video stream found in input")]
    StreamNotFound,

    #[error("decoder error: {0}")]
    Decode(String),

    #[error("invalid range: {0}")]
    InvalidRange(String),

    #[error("invalid resize parameters: {0}")]
    InvalidResize(String),
}

#[derive(Debug, Clone)]
pub struct VideoBatch {
    pub frames: Vec<u8>,
    pub frame_count: usize,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}

impl VideoBatch {
    pub const CHANNELS: usize = 3;

    pub fn shape(&self) -> (usize, usize, usize, usize) {
        (self.frame_count, self.height, self.width, self.channels)
    }

    pub fn len_bytes(&self) -> usize {
        self.frames.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRange {
    pub start: usize,
    pub end: Option<usize>,
    pub step: NonZeroUsize,
}

impl FrameRange {
    pub fn new(start: usize, end: Option<usize>, step: NonZeroUsize) -> VideoResult<Self> {
        if let Some(end) = end {
            if end < start {
                return Err(VideoError::InvalidRange(format!(
                    "end ({end}) must be >= start ({start})"
                )));
            }
        }

        Ok(Self { start, end, step })
    }
}

impl Default for FrameRange {
    fn default() -> Self {
        Self { start: 0, end: None, step: NonZeroUsize::new(1).expect("non zero") }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResizeFilter {
    #[default]
    Bilinear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResizeOptions {
    pub width: NonZeroU32,
    pub height: NonZeroU32,
    pub filter: ResizeFilter,
}

impl ResizeOptions {
    pub fn new(width: NonZeroU32, height: NonZeroU32) -> Self {
        Self { width, height, filter: ResizeFilter::default() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LoadOptions {
    pub frame_range: FrameRange,
    pub resize: Option<ResizeOptions>,
}
