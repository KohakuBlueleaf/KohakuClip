//! Zip archives of stored (uncompressed) members: listing (zip64 aware), writing shards.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::OnceLock;

use super::index::INDEX_NAME;
use super::{Counted, Listing, Location, Member};

const EOCD: u32 = 0x0605_4b50;
const EOCD64: u32 = 0x0606_4b50;
const EOCD64_LOCATOR: u32 = 0x0706_4b50;
const CENTRAL: u32 = 0x0201_4b50;
const LOCAL: u32 = 0x0403_4b50;
const ZIP64_EXTRA: u16 = 0x0001;

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

/// Where the central directory is: member count, size and offset (zip64 aware).
struct CentralDirectory {
    count: u64,
    size: u64,
    offset: u64,
}

fn central_directory(file: &File) -> io::Result<CentralDirectory> {
    let len = file.metadata()?.len();
    let tail_len = len.min(22 + 65535 + 20);
    let mut tail = vec![0u8; tail_len as usize];
    file.read_exact_at(&mut tail, len - tail_len)?;
    let eocd = (0..=tail.len().saturating_sub(22))
        .rev()
        .find(|&i| u32_at(&tail, i) == EOCD)
        .ok_or_else(|| bad("not a zip or tar archive"))?;

    if eocd >= 20 && u32_at(&tail, eocd - 20) == EOCD64_LOCATOR {
        let mut record = [0u8; 56];
        file.read_exact_at(&mut record, u64_at(&tail, eocd - 20 + 8))?;
        if u32_at(&record, 0) != EOCD64 {
            return Err(bad("bad zip64 end of central directory"));
        }
        return Ok(CentralDirectory {
            count: u64_at(&record, 32),
            size: u64_at(&record, 40),
            offset: u64_at(&record, 48),
        });
    }
    Ok(CentralDirectory {
        count: u16_at(&tail, eocd + 10) as u64,
        size: u32_at(&tail, eocd + 12) as u64,
        offset: u32_at(&tail, eocd + 16) as u64,
    })
}

/// The data offset of member `name` when it is the archive's last member (where KohakuClip's
/// writers put the index), found from the end of the central directory alone: opening a shard
/// does not read its whole directory. None when the last member is something else.
pub fn find_last(file: &File, name: &str) -> io::Result<Option<u64>> {
    let cd = central_directory(file)?;
    // the entry is 46 bytes + the name, + a 12-byte zip64 extra field past 4 GiB
    for extra_len in [0u64, 12] {
        let entry_len = 46 + name.len() as u64 + extra_len;
        if entry_len > cd.size {
            continue;
        }
        let mut entry = vec![0u8; entry_len as usize];
        file.read_exact_at(&mut entry, cd.offset + cd.size - entry_len)?;
        let matches = u32_at(&entry, 0) == CENTRAL
            && u16_at(&entry, 28) as usize == name.len()
            && u16_at(&entry, 30) as u64 == extra_len
            && u16_at(&entry, 32) == 0
            && &entry[46..46 + name.len()] == name.as_bytes();
        if matches {
            let header = zip64_header_offset(&entry, &entry[46 + name.len()..]);
            return Ok(Some(data_offset(file, header)?));
        }
    }
    Ok(None)
}

/// Read the central directory (zip64 aware); member data offsets are resolved lazily.
pub fn list(path: &Path, file: &File) -> io::Result<Listing> {
    let CentralDirectory {
        count,
        size: cd_size,
        offset: cd_offset,
    } = central_directory(file)?;

    let mut cd = vec![0u8; cd_size as usize];
    file.read_exact_at(&mut cd, cd_offset)?;
    let mut members = Vec::new();
    let mut index = None;
    let mut at = 0;
    for _ in 0..count {
        if u32_at(&cd, at) != CENTRAL {
            return Err(bad("bad zip central directory"));
        }
        let method = u16_at(&cd, at + 10);
        let name_len = u16_at(&cd, at + 28) as usize;
        let extra_len = u16_at(&cd, at + 30) as usize;
        let comment_len = u16_at(&cd, at + 32) as usize;
        let name = String::from_utf8_lossy(&cd[at + 46..at + 46 + name_len]).into_owned();
        let extra = &cd[at + 46 + name_len..at + 46 + name_len + extra_len];
        let header = zip64_header_offset(&cd[at..], extra);
        let entry = at;
        at += 46 + name_len + extra_len + comment_len;

        if name.ends_with('/') {
            continue;
        }
        // stored, or deflated (images: inflated after the read)
        if method != 0 && method != 8 {
            return Err(bad(&format!(
                "{name}: zip member compressed with method {method} (stored or deflated only)"
            )));
        }
        let member = Member {
            name,
            path: path.to_path_buf(),
            size: zip64_size(&cd[entry..], extra),
            deflated: method == 8,
            location: Location::ZipLocalHeader(header),
            data: OnceLock::new(),
        };
        if member.name == INDEX_NAME {
            index = Some(member.data_offset(file)?);
        } else {
            members.push(member);
        }
    }
    Ok(Listing { members, index })
}

/// The (stored) size of a central directory entry's member (from the zip64 extra field if
/// needed).
fn zip64_size(entry: &[u8], extra: &[u8]) -> u64 {
    let size = u32_at(entry, 20);
    if size != u32::MAX {
        return size as u64;
    }
    // zip64 extra: the uncompressed size first (if saturated), then the compressed size
    let mut i = 0;
    while i + 4 <= extra.len() {
        let id = u16_at(extra, i);
        let len = u16_at(extra, i + 2) as usize;
        if id == ZIP64_EXTRA {
            let mut field = i + 4;
            if u32_at(entry, 24) == u32::MAX {
                field += 8;
            }
            return u64_at(extra, field);
        }
        i += 4 + len;
    }
    size as u64
}

/// The local header offset of a central directory entry (from the zip64 extra field if needed).
fn zip64_header_offset(entry: &[u8], extra: &[u8]) -> u64 {
    let offset = u32_at(entry, 42);
    if offset != u32::MAX {
        return offset as u64;
    }
    // zip64 extra: uncompressed size, compressed size, offset; each present only if its 32-bit
    // field is saturated
    let mut i = 0;
    while i + 4 <= extra.len() {
        let id = u16_at(extra, i);
        let len = u16_at(extra, i + 2) as usize;
        if id == ZIP64_EXTRA {
            let mut field = i + 4;
            if u32_at(entry, 24) == u32::MAX {
                field += 8;
            }
            if u32_at(entry, 20) == u32::MAX {
                field += 8;
            }
            return u64_at(extra, field);
        }
        i += 4 + len;
    }
    offset as u64
}

/// Writes a zip archive of stored members (zip64 end records, so any archive size works).
pub struct ZipWriter {
    out: Counted<BufWriter<File>>,
    central: Vec<u8>,
    count: u64,
}

impl ZipWriter {
    pub fn create(path: &Path) -> io::Result<Self> {
        Ok(Self {
            out: Counted::new(BufWriter::new(File::create(path)?)),
            central: Vec::new(),
            count: 0,
        })
    }

    /// Append a member; returns the absolute offset of its data.
    pub fn add(&mut self, name: &str, data: &[u8]) -> io::Result<u64> {
        let size = u32::try_from(data.len()).map_err(|_| bad("zip member over 4 GiB"))?;
        let crc = crc32fast::hash(data);
        let header = self.out.pos;

        let mut local = Vec::with_capacity(30 + name.len());
        local.extend_from_slice(&LOCAL.to_le_bytes());
        local.extend_from_slice(&45u16.to_le_bytes()); // version needed (zip64)
        local.extend_from_slice(&[0; 6]); // flags, method (stored), time
        local.extend_from_slice(&0x21u16.to_le_bytes()); // date: 1980-01-01
        local.extend_from_slice(&crc.to_le_bytes());
        local.extend_from_slice(&size.to_le_bytes());
        local.extend_from_slice(&size.to_le_bytes());
        local.extend_from_slice(&(name.len() as u16).to_le_bytes());
        local.extend_from_slice(&0u16.to_le_bytes()); // extra length
        local.extend_from_slice(name.as_bytes());
        self.out.write_all(&local)?;
        let data_offset = self.out.pos;
        self.out.write_all(data)?;

        // central directory entry; the header offset goes to a zip64 extra field past 4 GiB
        let big = header >= u32::MAX as u64;
        let c = &mut self.central;
        c.extend_from_slice(&CENTRAL.to_le_bytes());
        c.extend_from_slice(&45u16.to_le_bytes()); // version made by
        c.extend_from_slice(&local[4..30]); // shared with the local header (from version needed)
        let extra_len: u16 = if big { 12 } else { 0 };
        let end = c.len();
        c[end - 2..].copy_from_slice(&extra_len.to_le_bytes());
        c.extend_from_slice(&[0; 10]); // comment length, disk, internal and external attributes
        let offset_field = if big { u32::MAX } else { header as u32 };
        c.extend_from_slice(&offset_field.to_le_bytes());
        c.extend_from_slice(name.as_bytes());
        if big {
            c.extend_from_slice(&ZIP64_EXTRA.to_le_bytes());
            c.extend_from_slice(&8u16.to_le_bytes());
            c.extend_from_slice(&header.to_le_bytes());
        }
        self.count += 1;
        Ok(data_offset)
    }

    /// Write the central directory and the (zip64) end records.
    pub fn finish(mut self) -> io::Result<()> {
        let cd_offset = self.out.pos;
        let cd_size = self.central.len() as u64;
        self.out.write_all(&self.central)?;

        let eocd64 = self.out.pos;
        let mut end = Vec::with_capacity(56 + 20 + 22);
        end.extend_from_slice(&EOCD64.to_le_bytes());
        end.extend_from_slice(&44u64.to_le_bytes()); // size of the rest of this record
        end.extend_from_slice(&45u16.to_le_bytes());
        end.extend_from_slice(&45u16.to_le_bytes());
        end.extend_from_slice(&[0; 8]); // this disk, central directory disk
        end.extend_from_slice(&self.count.to_le_bytes());
        end.extend_from_slice(&self.count.to_le_bytes());
        end.extend_from_slice(&cd_size.to_le_bytes());
        end.extend_from_slice(&cd_offset.to_le_bytes());

        end.extend_from_slice(&EOCD64_LOCATOR.to_le_bytes());
        end.extend_from_slice(&0u32.to_le_bytes());
        end.extend_from_slice(&eocd64.to_le_bytes());
        end.extend_from_slice(&1u32.to_le_bytes());

        end.extend_from_slice(&EOCD.to_le_bytes());
        end.extend_from_slice(&[0; 4]); // this disk, central directory disk
        let count16 = self.count.min(u16::MAX as u64) as u16;
        end.extend_from_slice(&count16.to_le_bytes());
        end.extend_from_slice(&count16.to_le_bytes());
        end.extend_from_slice(&(cd_size.min(u32::MAX as u64) as u32).to_le_bytes());
        end.extend_from_slice(&(cd_offset.min(u32::MAX as u64) as u32).to_le_bytes());
        end.extend_from_slice(&0u16.to_le_bytes()); // comment length
        self.out.write_all(&end)?;
        self.out.inner.flush()
    }
}

/// Absolute offset of a member's data: after its local header at `header`.
pub fn data_offset(file: &File, header: u64) -> io::Result<u64> {
    let mut head = [0u8; 30];
    file.read_exact_at(&mut head, header)?;
    let name_len = u16_at(&head, 26) as u64;
    let extra_len = u16_at(&head, 28) as u64;
    Ok(header + 30 + name_len + extra_len)
}
