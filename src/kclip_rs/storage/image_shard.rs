//! One source of images: a zip shard written by KohakuClip (its index read in place), or any
//! zip / tar archive or folder tree of image files (listed; each image's size comes from its
//! own header when it is decoded).

use std::fs::File;
use std::io;
use std::num::NonZeroUsize;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use lru::LruCache;
use memmap2::Mmap;

use super::image_index::{self, ImageRecord};
use super::index::INDEX_NAME;
use super::{Member, is_tar, list_files, zip};
use crate::codec::ImageCodec;
use crate::image::Color;

/// Largest inflated size of a deflated image member.
const MAX_INFLATED: usize = 1 << 28;

/// Where one image's bytes are and what is known about it before decoding.
#[derive(Clone)]
pub struct ImageEntry {
    pub file: Arc<File>,
    /// Absolute offset and length of the image's bytes in `file`.
    pub offset: u64,
    pub size: u32,
    pub codec: ImageCodec,
    /// Stored size; 0 x 0 when the source has no index.
    pub w: u32,
    pub h: u32,
    pub color: Color,
    /// The bytes are a deflated zip member (a plain zip of images).
    pub deflated: bool,
}

impl ImageEntry {
    /// The image's bytes into `bytes` (one pread; inflated, up to `MAX_INFLATED` bytes, when
    /// the member is deflated).
    pub fn read_into(&self, bytes: &mut Vec<u8>) -> io::Result<()> {
        bytes.resize(self.size as usize, 0);
        self.file.read_exact_at(bytes, self.offset)?;
        if self.deflated {
            let inflated = miniz_oxide::inflate::decompress_to_vec_with_limit(bytes, MAX_INFLATED)
                .map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("inflate: {e:?}"))
                })?;
            *bytes = inflated;
        }
        Ok(())
    }

    /// Ask the kernel to start reading the image's bytes now (POSIX_FADV_WILLNEED), so the
    /// reads of a planned batch overlap with its decoding.
    pub fn readahead(&self) {
        use std::os::fd::AsRawFd;

        // SAFETY: a plain advisory call on an open descriptor; failures only lose the hint
        unsafe {
            libc::posix_fadvise(
                self.file.as_raw_fd(),
                self.offset as libc::off_t,
                self.size as libc::off_t,
                libc::POSIX_FADV_WILLNEED,
            );
        }
    }
}

/// The color a codec's images have when no index says otherwise: JPEG and WebP fix theirs,
/// KohakuClip's AV1 images are written like its JPEGs.
pub fn default_color(codec: ImageCodec) -> Color {
    match codec {
        ImageCodec::Jpeg | ImageCodec::Av1 => Color::Yuv {
            bt709: false,
            full_range: true,
        },
        ImageCodec::Webp => Color::Yuv {
            bt709: false,
            full_range: false,
        },
        ImageCodec::Jxl | ImageCodec::Png => Color::Rgb,
    }
}

fn is_image(name: &str) -> bool {
    ImageCodec::from_extension(name).is_some()
}

enum Images {
    /// A shard with an index: records read in place from the memory-mapped archive.
    Indexed {
        file: Arc<File>,
        map: Mmap,
        /// Byte offset of the first record in `map`, and the number of records.
        records: usize,
        count: usize,
        /// Member names, listed from the archive on first request.
        names: OnceLock<Vec<String>>,
    },
    /// An archive or folder without an index: its image members.
    Listed {
        members: Vec<(Member, ImageCodec)>,
        files: Mutex<LruCache<PathBuf, Arc<File>>>,
    },
}

pub struct ImageShard {
    pub path: PathBuf,
    images: Images,
}

impl ImageShard {
    /// `use_index: false` ignores an index (lists the members instead, e.g. for testing);
    /// `open_files` caps the open files of a folder.
    pub fn open(path: &Path, use_index: bool, open_files: usize) -> io::Result<Self> {
        let in_archive = path.is_file();
        let tar = in_archive && is_tar(path)?;

        // KohakuClip's zip shards end with the index: found without listing the archive
        let mut index = None;
        if use_index && in_archive && !tar {
            index = zip::find_last(&File::open(path)?, INDEX_NAME)?;
        }
        let mut listing = None;
        if index.is_none() {
            let listed = list_files(path, is_image)?;
            index = listed.index.filter(|_| use_index);
            listing = Some(listed);
        }

        let images = match (index, listing) {
            (Some(offset), _) => open_index(path, offset)?,
            (None, Some(listed)) => {
                let members = listed
                    .members
                    .into_iter()
                    .filter_map(|m| ImageCodec::from_extension(&m.name).map(|c| (m, c)))
                    .collect();
                let capacity = NonZeroUsize::new(open_files.max(1)).unwrap();
                Images::Listed {
                    members,
                    files: Mutex::new(LruCache::new(capacity)),
                }
            }
            (None, None) => unreachable!("listed when no index was found"),
        };
        Ok(Self {
            path: path.to_path_buf(),
            images,
        })
    }

    pub fn len(&self) -> usize {
        match &self.images {
            Images::Indexed { count, .. } => *count,
            Images::Listed { members, .. } => members.len(),
        }
    }

    /// Image `i`: its bytes and what the index says about it.
    pub fn entry(&self, i: usize) -> io::Result<ImageEntry> {
        match &self.images {
            Images::Indexed {
                file, map, records, ..
            } => {
                let start = records + i * size_of::<ImageRecord>();
                let bytes = &map[start..start + size_of::<ImageRecord>()];
                let record: &ImageRecord = bytemuck::from_bytes(bytes);
                let codec = record.codec().ok_or_else(|| {
                    let msg = format!("{}: image {i} has an unknown codec", self.path.display());
                    io::Error::new(io::ErrorKind::InvalidData, msg)
                })?;
                Ok(ImageEntry {
                    file: file.clone(),
                    offset: record.off(),
                    size: record.size(),
                    codec,
                    w: record.w(),
                    h: record.h(),
                    color: record.color(),
                    deflated: false,
                })
            }
            Images::Listed { members, files } => {
                let (member, codec) = &members[i];
                let file = {
                    let mut files = files.lock().unwrap();
                    match files.get(&member.path) {
                        Some(file) => file.clone(),
                        None => {
                            let file = Arc::new(File::open(&member.path)?);
                            files.put(member.path.clone(), file.clone());
                            file
                        }
                    }
                };
                let size = u32::try_from(member.size)
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "image over 4 GiB"))?;
                Ok(ImageEntry {
                    offset: member.data_offset(&file)?,
                    file,
                    size,
                    codec: *codec,
                    w: 0,
                    h: 0,
                    color: default_color(*codec),
                    deflated: member.deflated,
                })
            }
        }
    }

    /// The member name of image `i` (an indexed shard lists its archive once for this).
    pub fn name(&self, i: usize) -> io::Result<String> {
        match &self.images {
            Images::Indexed { names, count, .. } => {
                if names.get().is_none() {
                    let listed = list_files(&self.path, is_image)?;
                    let listed: Vec<String> = listed.members.into_iter().map(|m| m.name).collect();
                    if listed.len() != *count {
                        let msg = format!(
                            "{}: {} members for {count} index records",
                            self.path.display(),
                            listed.len()
                        );
                        return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
                    }
                    let _ = names.set(listed);
                }
                Ok(names.get().expect("names were listed")[i].clone())
            }
            Images::Listed { members, .. } => Ok(members[i].0.name.clone()),
        }
    }
}

fn open_index(path: &Path, offset: u64) -> io::Result<Images> {
    let file = File::open(path)?;
    // SAFETY: the archive is opened read-only and shards are not modified while being read
    let map = unsafe { Mmap::map(&file)? };
    let start = offset as usize;
    let invalid = |e: String| io::Error::new(io::ErrorKind::InvalidData, e);
    if map[start..].starts_with(super::index::MAGIC) {
        let msg = format!("{}: a video shard (read it with Reader)", path.display());
        return Err(invalid(msg));
    }
    let parsed = image_index::parse(&map[start..])
        .map_err(|e| invalid(format!("{}: {e}", path.display())))?;
    Ok(Images::Indexed {
        file: Arc::new(file),
        map,
        records: start + parsed.records,
        count: parsed.count,
        names: OnceLock::new(),
    })
}
