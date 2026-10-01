//! One source of videos: a zip or tar archive of mp4 files, or a folder tree of them.
//!
//! An archive written by KohakuClip carries an index member, memory-mapped in place: every frame's
//! absolute byte range, so a clip is read with one pread per GOP and nothing is parsed at load
//! time. Without it (and always for folders) each video's frame table comes from its own `moov`
//! box on first use, then stays in a cache.

use std::fs::File;
use std::io;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use lru::LruCache;
use memmap2::Mmap;

use super::index::{self, Frame, VideoMeta};
use super::{Member, list};
use crate::mp4;

/// A video parsed from its `moov`: metadata and frame table.
pub type ParsedVideo = Arc<(VideoMeta, Vec<Frame>)>;

/// A video's metadata and frame table: borrowed from the index, or parsed from its `moov`.
pub enum Video<'a> {
    Indexed {
        meta: &'a VideoMeta,
        frames: &'a [Frame],
    },
    Parsed(ParsedVideo),
}

impl Video<'_> {
    pub fn meta(&self) -> &VideoMeta {
        match self {
            Video::Indexed { meta, .. } => meta,
            Video::Parsed(parsed) => &parsed.0,
        }
    }

    pub fn frames(&self) -> &[Frame] {
        match self {
            Video::Indexed { frames, .. } => frames,
            Video::Parsed(parsed) => &parsed.1,
        }
    }
}

struct Index {
    map: Mmap,
    videos: Vec<VideoMeta>,
    /// Byte offset of the first record in `map`.
    records: usize,
}

pub struct Shard {
    pub path: PathBuf,
    pub members: Vec<Member>,
    index: Option<Index>,
    files: Mutex<LruCache<PathBuf, Arc<File>>>,
    parsed: Mutex<LruCache<usize, ParsedVideo>>,
}

impl Shard {
    /// `use_index: false` ignores an index member (the `moov` fallback, e.g. for testing);
    /// `open_files` caps the open mp4 files of a folder, `moov_cache` the cached frame tables.
    pub fn open(
        path: &Path,
        use_index: bool,
        open_files: usize,
        moov_cache: usize,
    ) -> io::Result<Self> {
        let listing = list(path)?;
        let index = match listing.index {
            Some(offset) if use_index => Some(open_index(path, offset)?),
            _ => None,
        };
        let capacity = |n: usize| NonZeroUsize::new(n.max(1)).unwrap();
        Ok(Self {
            path: path.to_path_buf(),
            members: listing.members,
            index,
            files: Mutex::new(LruCache::new(capacity(open_files))),
            parsed: Mutex::new(LruCache::new(capacity(moov_cache))),
        })
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// The open file holding video `i` (folders: least recently used files get closed; a file
    /// stays open while a queued read holds it).
    pub fn file(&self, i: usize) -> io::Result<Arc<File>> {
        let path = &self.members[i].path;
        let mut files = self.files.lock().unwrap();
        if let Some(file) = files.get(path) {
            return Ok(file.clone());
        }
        let file = Arc::new(File::open(path)?);
        files.put(path.clone(), file.clone());
        Ok(file)
    }

    pub fn video(&self, i: usize) -> io::Result<Video<'_>> {
        if let Some(index) = &self.index {
            let meta = &index.videos[i];
            let start = index.records + meta.row * size_of::<Frame>();
            let bytes = &index.map[start..start + meta.n * size_of::<Frame>()];
            return Ok(Video::Indexed {
                meta,
                frames: bytemuck::cast_slice(bytes),
            });
        }

        if let Some(parsed) = self.parsed.lock().unwrap().get(&i) {
            return Ok(Video::Parsed(parsed.clone()));
        }
        let file = self.file(i)?;
        let base = self.members[i].data_offset(&file)?;
        let parsed = mp4::read_video(&file, base).map_err(|e| {
            let name = &self.members[i].name;
            io::Error::new(e.kind(), format!("{}:{name}: {e}", self.path.display()))
        })?;
        let parsed = Arc::new(parsed);
        self.parsed.lock().unwrap().put(i, parsed.clone());
        Ok(Video::Parsed(parsed))
    }
}

fn open_index(path: &Path, offset: u64) -> io::Result<Index> {
    let file = File::open(path)?;
    // SAFETY: the archive is opened read-only and shards are not modified while being read
    let map = unsafe { Mmap::map(&file)? };
    let start = offset as usize;
    let parsed = index::parse(&map[start..]).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {e}", path.display()),
        )
    })?;
    Ok(Index {
        map,
        videos: parsed.videos,
        records: start + parsed.records,
    })
}
