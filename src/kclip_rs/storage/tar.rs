//! Tar archives: listing members, writing shards.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use super::index::INDEX_NAME;
use super::{Counted, Listing, Member};

pub fn list(path: &Path) -> io::Result<Listing> {
    let mut archive = tar::Archive::new(File::open(path)?);
    let mut members = Vec::new();
    let mut index = None;
    for entry in archive.entries_with_seek()? {
        let entry = entry?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let name = entry.path()?.to_string_lossy().into_owned();
        let offset = entry.raw_file_position();
        if name == INDEX_NAME {
            index = Some(offset);
        } else {
            members.push(Member::at(name, path.to_path_buf(), offset));
        }
    }
    Ok(Listing { members, index })
}

/// Writes a tar archive (ustar / GNU long names through the `tar` crate).
pub struct TarWriter {
    builder: tar::Builder<Counted<BufWriter<File>>>,
}

impl TarWriter {
    pub fn create(path: &Path) -> io::Result<Self> {
        let out = Counted::new(BufWriter::new(File::create(path)?));
        Ok(Self {
            builder: tar::Builder::new(out),
        })
    }

    /// Append a member; returns the absolute offset of its data.
    pub fn add(&mut self, name: &str, data: &[u8]) -> io::Result<u64> {
        let mut header = tar::Header::new_gnu();
        header.set_size(data.len() as u64);
        header.set_mode(0o644);
        header.set_entry_type(tar::EntryType::Regular);
        self.builder.append_data(&mut header, name, data)?;
        // the data ends the entry, padded to 512 bytes
        let padded = (data.len() as u64).div_ceil(512) * 512;
        Ok(self.builder.get_ref().pos - padded)
    }

    pub fn finish(self) -> io::Result<()> {
        let mut out = self.builder.into_inner()?;
        out.inner.flush()
    }
}
