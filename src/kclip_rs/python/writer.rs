//! Python functions of the writer.

use std::path::PathBuf;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::write::{Encoding, Source, encode, pack};

/// Re-encode one video to the storage format; returns the mp4 bytes. ``source``: a path (or URL)
/// FFmpeg can open, or the bytes of a video file. Runs without the GIL.
#[pyfunction(name = "encode")]
#[pyo3(signature = (
    source,
    codec = "av1".to_string(),
    crf = 36,
    gop = 16,
    preset = 6,
    max_short = 512,
    loop_filters = false,
))]
#[allow(clippy::too_many_arguments)]
pub fn py_encode<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    codec: String,
    crf: u32,
    gop: u32,
    preset: u32,
    max_short: u32,
    loop_filters: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let enc = Encoding {
        codec,
        crf,
        gop,
        preset,
        max_short,
        loop_filters,
    };
    let encoded = if let Ok(bytes) = source.cast::<PyBytes>() {
        let data = bytes.as_bytes().to_vec();
        py.detach(|| encode(Source::Bytes(data), &enc))
    } else {
        let path: PathBuf = source.extract()?;
        py.detach(|| encode(Source::Path(&path), &enc))
    };
    let data = encoded.map_err(PyRuntimeError::new_err)?;
    Ok(PyBytes::new(py, &data))
}

/// Write the shard ``path`` (``.tar``, otherwise a stored zip) from (name, mp4 bytes) members,
/// with its index. Runs without the GIL.
#[pyfunction(name = "pack")]
pub fn py_pack(py: Python<'_>, path: PathBuf, members: Vec<(String, Vec<u8>)>) -> PyResult<()> {
    py.detach(|| pack(&path, &members))
        .map_err(PyRuntimeError::new_err)
}
