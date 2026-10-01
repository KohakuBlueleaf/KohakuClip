//! The shard index: per video its metadata, per frame its byte range and flags.
//!
//! Stored as the archive member `__index__.bin`:
//! `b"KCIDX1\0\0"` | u64 json length | json `{"videos": [VideoMeta, ...]}` | pad to 8 |
//! `Frame` records.
//! The records are read in place from a memory map of the archive (shared by all workers through
//! the page cache), so `Frame` has alignment 1 and explicit little-endian fields.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

pub const INDEX_NAME: &str = "__index__.bin";
pub const MAGIC: &[u8; 8] = b"KCIDX1\0\0";

const KEY: u8 = 1;

/// One stored frame (an mp4 sample): 24 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Frame {
    off: [u8; 8],
    size: [u8; 4],
    keep: [u8; 4],
    flags: u8,
    pad: [u8; 7],
}

impl Frame {
    /// `off`: absolute byte offset in the file holding the video. `keep`: bytes of the sample that
    /// later frames depend on (decoded when this frame is not wanted; 0: left out).
    pub fn new(off: u64, size: u32, keep: u32, key: bool) -> Self {
        Self {
            off: off.to_le_bytes(),
            size: size.to_le_bytes(),
            keep: keep.to_le_bytes(),
            flags: if key { KEY } else { 0 },
            pad: [0; 7],
        }
    }

    pub fn off(&self) -> u64 {
        u64::from_le_bytes(self.off)
    }

    pub fn size(&self) -> u32 {
        u32::from_le_bytes(self.size)
    }

    pub fn keep(&self) -> u32 {
        u32::from_le_bytes(self.keep)
    }

    pub fn is_key(&self) -> bool {
        self.flags & KEY != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Codec {
    H264,
    Hevc,
    Av1,
}

impl Codec {
    pub fn name(self) -> &'static str {
        match self {
            Codec::H264 => "h264",
            Codec::Hevc => "hevc",
            Codec::Av1 => "av1",
        }
    }
}

/// Per-video metadata (the json part of the index).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VideoMeta {
    #[serde(default)]
    pub member: String,
    /// First record of this video in the record table.
    #[serde(default)]
    pub row: usize,
    pub n: usize,
    pub fps: f64,
    pub h: u32,
    pub w: u32,
    pub codec: Codec,
    /// Prepended to each GOP's first packet: the AV1 sequence header, or the H.264 / HEVC
    /// parameter sets in Annex B (mp4 keeps them out of band).
    #[serde(with = "hex", default)]
    pub prefix: Vec<u8>,
    #[serde(default = "bt709")]
    pub colorspace: String,
}

fn bt709() -> String {
    "bt709".to_string()
}

#[derive(Serialize, Deserialize)]
struct Header {
    videos: Vec<VideoMeta>,
}

/// Parsed index member: the videos and where their records start (relative to `data`).
pub struct Parsed {
    pub videos: Vec<VideoMeta>,
    pub records: usize,
}

/// Parse the index member whose bytes start at `data`.
pub fn parse(data: &[u8]) -> Result<Parsed, String> {
    if data.len() < 16 || &data[..8] != MAGIC {
        return Err("bad index magic".into());
    }
    let json_len = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
    let json = data.get(16..16 + json_len).ok_or("truncated index")?;
    let header: Header = serde_json::from_slice(json).map_err(|e| e.to_string())?;
    let records = 16 + json_len + pad8(16 + json_len);
    let total: usize = header.videos.iter().map(|v| v.n).sum();
    if data.len() < records + total * size_of::<Frame>() {
        return Err("truncated index records".into());
    }
    Ok(Parsed {
        videos: header.videos,
        records,
    })
}

/// The index member's bytes for `videos` (their `row`s are set here) and their frames, in order.
pub fn build(videos: &mut [VideoMeta], frames: &[Vec<Frame>]) -> Vec<u8> {
    let mut row = 0;
    for (meta, table) in videos.iter_mut().zip(frames) {
        meta.row = row;
        row += table.len();
    }
    let header = Header {
        videos: videos.to_vec(),
    };
    let json = serde_json::to_vec(&header).expect("index metadata serializes");

    let mut out = Vec::with_capacity(32 + json.len() + row * size_of::<Frame>());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(json.len() as u64).to_le_bytes());
    out.extend_from_slice(&json);
    out.resize(out.len() + pad8(out.len()), 0);
    for table in frames {
        out.extend_from_slice(bytemuck::cast_slice(table));
    }
    out
}

fn pad8(n: usize) -> usize {
    (8 - n % 8) % 8
}

mod hex {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8], s: S) -> Result<S::Ok, S::Error> {
        let text: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        s.serialize_str(&text)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let text = String::deserialize(d)?;
        (0..text.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&text[i..i + 2], 16).map_err(serde::de::Error::custom))
            .collect()
    }
}
