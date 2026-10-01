//! Storage: where videos live (zip / tar archives of mp4 files, or folder trees), the shard
//! index, and shards themselves.

mod folder;
pub mod index;
pub mod shard;
pub mod tar;
pub mod zip;

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// One mp4 inside a source.
pub struct Member {
    pub name: String,
    /// The file holding the bytes: the archive, or the mp4 itself.
    pub path: PathBuf,
    location: Location,
    data: OnceLock<u64>,
}

enum Location {
    /// The mp4 starts at this byte of `path` (tar member data, or a plain file at 0).
    At(u64),
    /// A zip member: its data follows the local header at this byte.
    ZipLocalHeader(u64),
}

impl Member {
    fn at(name: String, path: PathBuf, offset: u64) -> Self {
        Self {
            name,
            path,
            location: Location::At(offset),
            data: OnceLock::new(),
        }
    }

    /// Absolute offset of the mp4's first byte in `path` (zip: read from the local header once).
    pub fn data_offset(&self, file: &File) -> io::Result<u64> {
        if let Some(&offset) = self.data.get() {
            return Ok(offset);
        }
        let offset = match self.location {
            Location::At(offset) => offset,
            Location::ZipLocalHeader(header) => zip::data_offset(file, header)?,
        };
        Ok(*self.data.get_or_init(|| offset))
    }
}

/// The members of a source and the absolute offset of its index member's data, if it has one.
pub struct Listing {
    pub members: Vec<Member>,
    pub index: Option<u64>,
}

pub fn list(path: &Path) -> io::Result<Listing> {
    if path.is_dir() {
        return Ok(folder::list(path));
    }
    let mut magic = [0u8; 262];
    let mut file = File::open(path)?;
    let n = file.read(&mut magic)?;
    if n >= 262 && &magic[257..262] == b"ustar" {
        tar::list(path)
    } else {
        zip::list(path, &file)
    }
}

/// A writer that knows how many bytes went through it.
pub(crate) struct Counted<W> {
    pub inner: W,
    pub pos: u64,
}

impl<W> Counted<W> {
    pub fn new(inner: W) -> Self {
        Self { inner, pos: 0 }
    }
}

impl<W: Write> Write for Counted<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.pos += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}
