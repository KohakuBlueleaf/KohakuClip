//! KohakuClip: fast random-access video clips for training.
//!
//! - `storage`: zip / tar archives and folders of mp4 files, the shard index, shards.
//! - `mp4`: frame tables from `moov`, decoder configuration, faststart.
//! - `read`: planning clips, sampling frames, the decode thread pool.
//! - `decode`: decoding a planned clip into output pixels.
//! - `image`: antialiased resize fused with the crop, YUV -> RGB.
//! - `write`: re-encoding videos with the linked FFmpeg, packing shards.
//! - `python`: the `kohakuclip._core` extension module.

mod decode;
mod image;
mod mp4;
mod python;
mod read;
mod storage;
mod write;
