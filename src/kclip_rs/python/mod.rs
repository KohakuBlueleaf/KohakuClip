//! The Python module `kohakuclip._core`.

mod reader;
mod writer;

use pyo3::prelude::*;

#[pymodule]
mod _core {
    #[pymodule_export]
    use super::reader::{Batch, Pending, Reader, VideoInfo, profile};
    #[pymodule_export]
    use super::writer::{py_encode, py_pack};
}
