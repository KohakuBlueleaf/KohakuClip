//! A submitted batch (of clips or images): decode threads write into its output array, and
//! `result()` waits for them without the GIL.

use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::read::pool;

/// A submitted batch; ``result()`` waits for it.
#[pyclass(module = "kohakuclip._core")]
pub struct Pending {
    done: Arc<pool::Batch<Result<(), String>>>,
    /// What ``result()`` hands out: a ``Batch`` or an ``ImageBatch``.
    batch: Option<Py<PyAny>>,
    /// One task's name in error messages ("clip", "image").
    item: &'static str,
}

impl Pending {
    pub fn new(
        done: Arc<pool::Batch<Result<(), String>>>,
        batch: Py<PyAny>,
        item: &'static str,
    ) -> Self {
        Self {
            done,
            batch: Some(batch),
            item,
        }
    }
}

#[pymethods]
impl Pending {
    /// Wait (without the GIL) until the batch is decoded and return it.
    pub fn result(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // taken first: after the wait no thread writes into the batch any more, so a failed
        // batch is freed here and the drop below has nothing left to wait for
        let batch = self
            .batch
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("result() was already taken"))?;
        let done = self.done.clone();
        let results = py.detach(move || done.wait());
        let item = self.item;
        let failed: Vec<String> = results
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.as_ref().err().map(|e| format!("{item} {i}: {e}")))
            .collect();
        if !failed.is_empty() {
            return Err(PyRuntimeError::new_err(failed.join("; ")));
        }
        Ok(batch)
    }
}

impl Drop for Pending {
    /// A batch dropped before `result()` still has decode threads writing into its output
    /// array: wait for them before the array can be freed.
    fn drop(&mut self) {
        if self.batch.is_some() {
            self.done.join();
        }
    }
}

/// One task's region of a batch's output array, handed to a decode thread.
pub struct OutRegion(*mut u8, usize);

// SAFETY: every task writes only its own disjoint region of the output array, and the array is
// kept alive (owned by the pending batch) until all tasks are waited for
unsafe impl Send for OutRegion {}

impl OutRegion {
    /// Region `i` of `len` bytes of the array at `base`.
    ///
    /// # Safety
    /// The array must hold at least `(i + 1) * len` bytes and outlive the region's use.
    pub unsafe fn new(base: *mut u8, i: usize, len: usize) -> Self {
        // SAFETY: inside the array (the caller's contract)
        Self(unsafe { base.add(i * len) }, len)
    }

    /// The region as a slice, for the one task it was made for.
    ///
    /// # Safety
    /// The array must stay alive while the slice is used.
    pub unsafe fn into_slice<'a>(self) -> &'a mut [u8] {
        // SAFETY: the caller's contract; regions are disjoint and each is used once
        unsafe { std::slice::from_raw_parts_mut(self.0, self.1) }
    }
}
