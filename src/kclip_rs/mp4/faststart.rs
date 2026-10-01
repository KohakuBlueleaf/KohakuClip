//! Faststart: the `moov` box in front of `mdat`.

use std::io;

use super::{bad, be32, be64, boxes, find};

/// The byte range (header included) of the first top-level box of type `kind` in an mp4 file.
pub fn find_box(data: &[u8], kind: &[u8; 4]) -> Option<std::ops::Range<usize>> {
    boxes(data, 0, data.len())
        .find(|b| &b.kind == kind)
        .map(|b| header_start(data, b.start)..b.end)
}

/// Where a box starts, given where its payload starts (8- or 16-byte header).
fn header_start(data: &[u8], payload: usize) -> usize {
    if payload >= 16 && be32(data, payload - 16) == 1 {
        payload - 16
    } else {
        payload - 8
    }
}

/// Move the `moov` box in front of `mdat` (faststart), shifting every chunk offset by its size.
pub fn faststart(data: Vec<u8>) -> io::Result<Vec<u8>> {
    let moov = find_box(&data, b"moov").ok_or_else(|| bad("no moov box"))?;
    let mdat = find_box(&data, b"mdat").ok_or_else(|| bad("no mdat box"))?;
    if moov.start < mdat.start {
        return Ok(data);
    }

    let mut moved = data[moov.clone()].to_vec();
    shift_chunk_offsets(&mut moved, moov.len() as u64)?;
    let mut out = Vec::with_capacity(data.len());
    out.extend_from_slice(&data[..mdat.start]);
    out.extend_from_slice(&moved);
    out.extend_from_slice(&data[mdat.start..moov.start]);
    out.extend_from_slice(&data[moov.end..]);
    Ok(out)
}

/// Add `delta` to every chunk offset (stco / co64) of every track in `moov` (header included).
fn shift_chunk_offsets(moov: &mut [u8], delta: u64) -> io::Result<()> {
    let mut tables = Vec::new();
    for trak in boxes(moov, 8, moov.len()).filter(|b| &b.kind == b"trak") {
        let Some((stbl, end)) = find(moov, &[b"mdia", b"minf", b"stbl"], trak.start, trak.end)
        else {
            continue;
        };
        for b in boxes(moov, stbl, end) {
            if &b.kind == b"stco" || &b.kind == b"co64" {
                tables.push((b.kind, b.start));
            }
        }
    }

    for (kind, start) in tables {
        let count = be32(moov, start + 4) as usize;
        for i in 0..count {
            if &kind == b"stco" {
                let at = start + 8 + 4 * i;
                let shifted = u32::try_from(be32(moov, at) as u64 + delta)
                    .map_err(|_| bad("chunk offset over 4 GiB after faststart"))?;
                moov[at..at + 4].copy_from_slice(&shifted.to_be_bytes());
            } else {
                let at = start + 8 + 8 * i;
                let shifted = be64(moov, at) + delta;
                moov[at..at + 8].copy_from_slice(&shifted.to_be_bytes());
            }
        }
    }
    Ok(())
}
