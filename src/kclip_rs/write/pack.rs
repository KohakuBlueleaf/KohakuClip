//! Packing mp4 files into a shard with its index.

use std::path::Path;

use super::av1;
use crate::decode;
use crate::mp4;
use crate::storage::index::{self, Codec, Frame, INDEX_NAME, VideoMeta};
use crate::storage::tar::TarWriter;
use crate::storage::zip::ZipWriter;

/// The index entry of one mp4 (its bytes start at `base` in the archive): frame table from its
/// moov; for AV1 also the bytes later frames depend on, per sample.
fn frame_table(name: &str, data: &[u8], base: u64) -> Result<(VideoMeta, Vec<Frame>), String> {
    let moov = mp4::find_box(data, b"moov").ok_or_else(|| format!("{name}: no moov box"))?;
    let (mut meta, mut frames) =
        mp4::parse_moov(&data[moov], base).map_err(|e| format!("{name}: {e}"))?;
    meta.member = name.to_string();

    let first = frames.first().ok_or_else(|| format!("{name}: no frames"))?;
    let start = (first.off() - base) as usize;
    let sample = &data[start..start + first.size() as usize];
    let colorspace = decode::probe_colorspace(meta.codec, &meta.prefix, sample);
    meta.colorspace = colorspace.map_err(|e| format!("{name}: {e}"))?.to_string();

    if meta.codec == Codec::Av1 {
        let samples = frames.iter().map(|f| {
            let start = (f.off() - base) as usize;
            &data[start..start + f.size() as usize]
        });
        let keep = av1::keep_bytes(&meta.prefix, samples)
            .ok_or_else(|| format!("{name}: no AV1 sequence header"))?;
        for (frame, keep) in frames.iter_mut().zip(keep) {
            *frame = Frame::new(frame.off(), frame.size(), keep, frame.is_key());
        }
    }
    Ok((meta, frames))
}

/// Pack mp4 files (name, bytes) into a `.tar` or (otherwise) stored `.zip` shard, then append
/// the index member.
pub fn pack(path: &Path, members: &[(String, Vec<u8>)]) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut videos = Vec::with_capacity(members.len());
    let mut tables = Vec::with_capacity(members.len());

    if path.extension().is_some_and(|ext| ext == "tar") {
        let mut tar = TarWriter::create(path).map_err(io)?;
        for (name, data) in members {
            let base = tar.add(name, data).map_err(io)?;
            let (meta, frames) = frame_table(name, data, base)?;
            videos.push(meta);
            tables.push(frames);
        }
        tar.add(INDEX_NAME, &index::build(&mut videos, &tables))
            .map_err(io)?;
        tar.finish().map_err(io)
    } else {
        let mut zip = ZipWriter::create(path).map_err(io)?;
        for (name, data) in members {
            let base = zip.add(name, data).map_err(io)?;
            let (meta, frames) = frame_table(name, data, base)?;
            videos.push(meta);
            tables.push(frames);
        }
        zip.add(INDEX_NAME, &index::build(&mut videos, &tables))
            .map_err(io)?;
        zip.finish().map_err(io)
    }
}
