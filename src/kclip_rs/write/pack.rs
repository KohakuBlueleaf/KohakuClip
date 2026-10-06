//! Packing mp4 files or images into a shard with its index.

use std::path::Path;

use super::av1;
use crate::codec::ImageCodec;
use crate::decode;
use crate::mp4;
use crate::storage::image_index::{self, ImageRecord};
use crate::storage::image_shard::default_color;
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

/// One stored image to pack: member name, bytes, stored size, codec.
pub struct ImageMember {
    pub name: String,
    pub data: Vec<u8>,
    pub w: u32,
    pub h: u32,
    pub codec: ImageCodec,
}

/// Pack images into a `.tar` or (otherwise) stored `.zip` shard, then append the image index
/// as the last member (`header`: a json object of shard-level facts; the image count is added).
pub fn pack_images(
    path: &Path,
    members: &[ImageMember],
    header: serde_json::Value,
) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut header = match header {
        serde_json::Value::Object(map) => map,
        _ => return Err("the index header must be a json object".into()),
    };
    header.insert("images".into(), members.len().into());
    let header = serde_json::Value::Object(header);

    let record = |m: &ImageMember, offset: u64| {
        let size = u32::try_from(m.data.len()).map_err(|_| format!("{}: over 4 GiB", m.name))?;
        let color = default_color(m.codec);
        Ok::<_, String>(ImageRecord::new(offset, size, m.w, m.h, m.codec, color))
    };
    let mut records = Vec::with_capacity(members.len());
    if path.extension().is_some_and(|ext| ext == "tar") {
        let mut tar = TarWriter::create(path).map_err(io)?;
        for m in members {
            let offset = tar.add(&m.name, &m.data).map_err(io)?;
            records.push(record(m, offset)?);
        }
        tar.add(INDEX_NAME, &image_index::build(&header, &records))
            .map_err(io)?;
        tar.finish().map_err(io)
    } else {
        let mut zip = ZipWriter::create(path).map_err(io)?;
        for m in members {
            let offset = zip.add(&m.name, &m.data).map_err(io)?;
            records.push(record(m, offset)?);
        }
        zip.add(INDEX_NAME, &image_index::build(&header, &records))
            .map_err(io)?;
        zip.finish().map_err(io)
    }
}
