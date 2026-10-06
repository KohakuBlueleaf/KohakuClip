//! The image shard index: one fixed-size record per image, read in place from a memory map
//! (nothing is parsed or allocated per image, so a reader over tens of millions of images costs
//! no more memory than the page cache it touches).
//!
//! Stored as the archive member `__index__.bin`, the shard's last member:
//! `b"KCIMG1\0\0"` | u64 json length | json header | pad to 8 | one `ImageRecord` per image, in
//! member order. The json header holds shard-level facts (`{"images": n, ...}` plus whatever
//! the writer records, e.g. its settings); per-image metadata (keys, captions, source sizes)
//! lives in the writer's sidecar manifest.

use bytemuck::{Pod, Zeroable};

use crate::codec::ImageCodec;
use crate::image::Color;

pub const MAGIC: &[u8; 8] = b"KCIMG1\0\0";

const BT709: u8 = 1;
const FULL_RANGE: u8 = 2;
const RGB: u8 = 4;
const GRAY: u8 = 8;

/// One stored image: 24 bytes, alignment 1, little-endian fields.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ImageRecord {
    off: [u8; 8],
    size: [u8; 4],
    w: [u8; 4],
    h: [u8; 4],
    codec: u8,
    color: u8,
    pad: [u8; 2],
}

impl ImageRecord {
    /// `off`: absolute byte offset of the image in the archive; (w, h): its stored size.
    pub fn new(off: u64, size: u32, w: u32, h: u32, codec: ImageCodec, color: Color) -> Self {
        let flags = match color {
            Color::Yuv { bt709, full_range } => {
                let mut flags = 0;
                if bt709 {
                    flags |= BT709;
                }
                if full_range {
                    flags |= FULL_RANGE;
                }
                flags
            }
            Color::Rgb => RGB,
            Color::Gray => GRAY,
        };
        Self {
            off: off.to_le_bytes(),
            size: size.to_le_bytes(),
            w: w.to_le_bytes(),
            h: h.to_le_bytes(),
            codec: codec as u8,
            color: flags,
            pad: [0; 2],
        }
    }

    pub fn off(&self) -> u64 {
        u64::from_le_bytes(self.off)
    }

    pub fn size(&self) -> u32 {
        u32::from_le_bytes(self.size)
    }

    pub fn w(&self) -> u32 {
        u32::from_le_bytes(self.w)
    }

    pub fn h(&self) -> u32 {
        u32::from_le_bytes(self.h)
    }

    pub fn codec(&self) -> Option<ImageCodec> {
        ImageCodec::from_byte(self.codec)
    }

    pub fn color(&self) -> Color {
        if self.color & RGB != 0 {
            return Color::Rgb;
        }
        if self.color & GRAY != 0 {
            return Color::Gray;
        }
        Color::Yuv {
            bt709: self.color & BT709 != 0,
            full_range: self.color & FULL_RANGE != 0,
        }
    }
}

/// The index member's bytes: `header` (a json object) and the records.
pub fn build(header: &serde_json::Value, records: &[ImageRecord]) -> Vec<u8> {
    let json = serde_json::to_vec(header).expect("index header serializes");
    let mut out = Vec::with_capacity(24 + json.len() + std::mem::size_of_val(records));
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(json.len() as u64).to_le_bytes());
    out.extend_from_slice(&json);
    out.resize(out.len().next_multiple_of(8), 0);
    out.extend_from_slice(bytemuck::cast_slice(records));
    out
}

/// A parsed index member: where the records start (relative to `data`) and how many there are.
pub struct Parsed {
    pub records: usize,
    pub count: usize,
}

/// Parse the index member whose bytes start at `data` (and run to the end of `data` at least).
pub fn parse(data: &[u8]) -> Result<Parsed, String> {
    if data.len() < 16 || &data[..8] != MAGIC {
        return Err("bad image index magic".into());
    }
    let json_len = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
    let json = data.get(16..16 + json_len).ok_or("truncated image index")?;
    let header: serde_json::Value = serde_json::from_slice(json).map_err(|e| e.to_string())?;
    let count = header["images"]
        .as_u64()
        .ok_or("image index header without an image count")? as usize;
    let records = (16 + json_len).next_multiple_of(8);
    if data.len() < records + count * size_of::<ImageRecord>() {
        return Err("truncated image index records".into());
    }
    Ok(Parsed { records, count })
}
