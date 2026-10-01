//! Minimal mp4 parsing and rewriting: frame tables from `moov`, decoder configuration, faststart.

mod config;
mod faststart;
mod moov;

use std::io;

pub use config::codec_prefix;
pub use faststart::{faststart, find_box};
pub use moov::{parse_moov, read_video};

pub fn be16(b: &[u8], at: usize) -> u16 {
    u16::from_be_bytes([b[at], b[at + 1]])
}

pub fn be32(b: &[u8], at: usize) -> u32 {
    u32::from_be_bytes(b[at..at + 4].try_into().unwrap())
}

pub fn be64(b: &[u8], at: usize) -> u64 {
    u64::from_be_bytes(b[at..at + 8].try_into().unwrap())
}

pub fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// A box: its type and the range of its payload in the parent buffer.
pub struct Box4 {
    pub kind: [u8; 4],
    pub start: usize,
    pub end: usize,
}

/// The boxes laid out in `buf[start..end]`.
pub fn boxes(buf: &[u8], start: usize, end: usize) -> impl Iterator<Item = Box4> + '_ {
    let mut at = start;
    std::iter::from_fn(move || {
        if at + 8 > end {
            return None;
        }
        let mut size = be32(buf, at) as usize;
        let kind: [u8; 4] = buf[at + 4..at + 8].try_into().unwrap();
        let mut head = 8;
        if size == 1 {
            size = be64(buf, at + 8) as usize;
            head = 16;
        } else if size == 0 {
            size = end - at;
        }
        let b = Box4 {
            kind,
            start: at + head,
            end: (at + size).min(end),
        };
        at += size.max(head);
        Some(b)
    })
}

/// The first box of type `path[0]` in `buf[start..end]`, then `path[1]` inside it, and so on.
pub fn find(buf: &[u8], path: &[&[u8; 4]], start: usize, end: usize) -> Option<(usize, usize)> {
    let b = boxes(buf, start, end).find(|b| &b.kind == path[0])?;
    if path.len() == 1 {
        Some((b.start, b.end))
    } else {
        find(buf, &path[1..], b.start, b.end)
    }
}
