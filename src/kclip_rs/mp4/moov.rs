//! The frame table of the first video track, from the `moov` box.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;

use super::{bad, be16, be32, be64, boxes, codec_prefix, find};
use crate::storage::index::{Codec, Frame, VideoMeta};

/// Read the `moov` box of the mp4 starting at byte `base` of `file` (any box order, faststart or
/// not) and parse its first video track.
pub fn read_video(file: &File, base: u64) -> io::Result<(VideoMeta, Vec<Frame>)> {
    let mut at = base;
    let mut head = [0u8; 16];
    loop {
        file.read_exact_at(&mut head, at)?;
        let mut size = be32(&head, 0) as u64;
        if size == 1 {
            size = be64(&head, 8);
        }
        if size < 8 {
            return Err(bad("bad mp4 box"));
        }
        if &head[4..8] == b"moov" {
            let mut moov = vec![0u8; size as usize];
            file.read_exact_at(&mut moov, at)?;
            return parse_moov(&moov, base);
        }
        at += size;
    }
}

/// The frame table of the first video track in `moov` (the whole box, header included): sizes
/// (stsz), chunk offsets (stco / co64) and samples per chunk (stsc) -> absolute offsets (the mp4
/// starts at byte `base` of its file), sync samples (stss), fps (mdhd + stts), size, codec and
/// decoder prefix (stsd).
pub fn parse_moov(moov: &[u8], base: u64) -> io::Result<(VideoMeta, Vec<Frame>)> {
    let trak = boxes(moov, 8, moov.len())
        .filter(|b| &b.kind == b"trak")
        .find(|b| {
            let hdlr = find(moov, &[b"mdia", b"hdlr"], b.start, b.end);
            hdlr.is_some_and(|(s, _)| &moov[s + 8..s + 12] == b"vide")
        })
        .ok_or_else(|| bad("no video track"))?;
    let (stbl, stbl_end) = find(moov, &[b"mdia", b"minf", b"stbl"], trak.start, trak.end)
        .ok_or_else(|| bad("no stbl"))?;
    let table = |kind: &[u8; 4]| boxes(moov, stbl, stbl_end).find(|b| &b.kind == kind);
    let need = |kind: &[u8; 4]| table(kind).ok_or_else(|| bad(format!("no {}", show(kind))));

    // sample sizes
    let stsz = need(b"stsz")?.start;
    let fixed = be32(moov, stsz + 4);
    let n = be32(moov, stsz + 8) as usize;
    let size: Vec<u64> = if fixed == 0 {
        (0..n)
            .map(|i| be32(moov, stsz + 12 + 4 * i) as u64)
            .collect()
    } else {
        vec![fixed as u64; n]
    };

    // chunk offsets
    let chunks: Vec<u64> = match table(b"stco") {
        Some(b) => {
            let count = be32(moov, b.start + 4) as usize;
            (0..count)
                .map(|i| be32(moov, b.start + 8 + 4 * i) as u64)
                .collect()
        }
        None => {
            let b = need(b"co64")?;
            let count = be32(moov, b.start + 4) as usize;
            (0..count)
                .map(|i| be64(moov, b.start + 8 + 8 * i))
                .collect()
        }
    };

    // samples per chunk: runs of (first chunk, samples per chunk)
    let stsc = need(b"stsc")?.start;
    let runs = be32(moov, stsc + 4) as usize;
    let mut offsets = Vec::with_capacity(n);
    let mut sample = 0;
    for run in 0..runs {
        let at = stsc + 8 + 12 * run;
        let first = be32(moov, at) as usize - 1;
        let per_chunk = be32(moov, at + 4) as usize;
        let last = if run + 1 < runs {
            be32(moov, at + 12) as usize - 1
        } else {
            chunks.len()
        };
        for &chunk in &chunks[first..last] {
            let mut off = chunk;
            for _ in 0..per_chunk {
                if sample == n {
                    break;
                }
                offsets.push(base + off);
                off += size[sample];
                sample += 1;
            }
        }
    }
    if offsets.len() != n {
        return Err(bad("sample table does not cover every sample"));
    }

    // keyframes (no stss: every sample is a sync sample)
    let mut key = vec![table(b"stss").is_none(); n];
    if let Some(b) = table(b"stss") {
        let count = be32(moov, b.start + 4) as usize;
        for i in 0..count {
            let k = be32(moov, b.start + 8 + 4 * i) as usize - 1;
            if k < n {
                key[k] = true;
            }
        }
    }

    // fps: samples / duration
    let (mdhd, _) =
        find(moov, &[b"mdia", b"mdhd"], trak.start, trak.end).ok_or_else(|| bad("no mdhd"))?;
    let timescale = match moov[mdhd] {
        1 => be32(moov, mdhd + 20),
        _ => be32(moov, mdhd + 12),
    } as f64;
    let stts = need(b"stts")?.start;
    let (mut samples, mut ticks) = (0u64, 0u64);
    for i in 0..be32(moov, stts + 4) as usize {
        let count = be32(moov, stts + 8 + 8 * i) as u64;
        let delta = be32(moov, stts + 12 + 8 * i) as u64;
        samples += count;
        ticks += count * delta;
    }
    let fps = timescale * samples as f64 / ticks.max(1) as f64;

    // sample entry: codec, size, decoder configuration, color
    let stsd = need(b"stsd")?;
    let entry = stsd.start + 8;
    let kind: [u8; 4] = moov[entry + 4..entry + 8].try_into().unwrap();
    let codec = match &kind {
        b"av01" => Codec::Av1,
        b"avc1" | b"avc3" => Codec::H264,
        b"hvc1" | b"hev1" => Codec::Hevc,
        _ => return Err(bad(format!("unsupported codec {}", show(&kind)))),
    };
    let w = be16(moov, entry + 32) as u32;
    let h = be16(moov, entry + 34) as u32;
    let entry_end = entry + be32(moov, entry) as usize;
    let config_kind: &[u8; 4] = match codec {
        Codec::Av1 => b"av1C",
        Codec::H264 => b"avcC",
        Codec::Hevc => b"hvcC",
    };
    let mut prefix = Vec::new();
    let mut colorspace = "bt709".to_string();
    for b in boxes(moov, entry + 86, entry_end) {
        if &b.kind == config_kind {
            prefix = codec_prefix(codec, &moov[b.start..b.end])?;
        } else if &b.kind == b"colr" && &moov[b.start..b.start + 4] == b"nclx" {
            colorspace = matrix_name(be16(moov, b.start + 8)).to_string();
        }
    }

    let frames = (0..n)
        .map(|i| Frame::new(offsets[i], size[i] as u32, size[i] as u32, key[i]))
        .collect();
    let meta = VideoMeta {
        member: String::new(),
        row: 0,
        n,
        fps,
        h,
        w,
        codec,
        prefix,
        colorspace,
    };
    Ok((meta, frames))
}

fn show(kind: &[u8; 4]) -> String {
    String::from_utf8_lossy(kind).into_owned()
}

/// ISO/IEC 23091-2 matrix coefficients -> the conversion KohakuClip applies.
fn matrix_name(matrix: u16) -> &'static str {
    match matrix {
        5 | 6 => "bt601",
        _ => "bt709",
    }
}
