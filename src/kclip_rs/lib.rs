//! KohakuClip: fast random-access video clips and images for training.
//!
//! - `storage`: zip / tar archives and folders of media files, the shard indexes, shards.
//! - `mp4`: frame tables from `moov`, decoder configuration, faststart.
//! - `read`: planning clips and images, sampling, the decode thread pool.
//! - `codec`: still-image codecs (JPEG through libjpeg-turbo, others through FFmpeg).
//! - `decode`: decoding a planned clip or image into output pixels.
//! - `image`: decoded pictures, antialiased resize fused with the crop, YUV -> RGB.
//! - `write`: re-encoding videos and images, packing shards.
//! - `python`: the `kohakuclip._core` extension module.

mod codec;
mod decode;
mod image;
mod mp4;
mod python;
mod read;
mod storage;
mod write;
