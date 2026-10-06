//! The Python module `kohakuclip._core`.

mod pending;
mod reader;
mod writer;

use pyo3::prelude::*;

#[pymodule]
mod _core {
    #[pymodule_export]
    use super::pending::Pending;
    #[pymodule_export]
    use super::reader::{Batch, Reader, VideoInfo, profile};
    #[pymodule_export]
    use super::writer::{py_encode, py_pack};
}
