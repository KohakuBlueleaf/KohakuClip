//! Folder trees of mp4 files.

use std::path::{Path, PathBuf};

use super::{Listing, Member};

/// Every `*.mp4` below `root`, sorted by path.
pub fn list(root: &Path) -> Listing {
    let mut files: Vec<PathBuf> = walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(walkdir::DirEntry::into_path)
        .filter(|p| p.extension().is_some_and(|ext| ext == "mp4"))
        .collect();
    files.sort();

    let members = files
        .into_iter()
        .map(|p| {
            let name = p
                .strip_prefix(root)
                .unwrap_or(&p)
                .to_string_lossy()
                .into_owned();
            Member::at(name, p, 0)
        })
        .collect();
    Listing {
        members,
        index: None,
    }
}
