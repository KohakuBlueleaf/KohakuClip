//! Folder trees of media files.

use std::path::Path;

use super::{Listing, Member};

/// Every file below `root` whose name `keep` accepts, sorted by path.
pub fn list(root: &Path, keep: fn(&str) -> bool) -> Listing {
    let mut files: Vec<(std::path::PathBuf, u64)> = walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter(|e| keep(&e.file_name().to_string_lossy()))
        .map(|e| {
            let size = e.metadata().map_or(0, |m| m.len());
            (e.into_path(), size)
        })
        .collect();
    files.sort();

    let members = files
        .into_iter()
        .map(|(p, size)| {
            let name = p
                .strip_prefix(root)
                .unwrap_or(&p)
                .to_string_lossy()
                .into_owned();
            Member::at(name, p, 0, size)
        })
        .collect();
    Listing {
        members,
        index: None,
    }
}
