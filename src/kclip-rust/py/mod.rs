use std::num::{NonZeroU32, NonZeroUsize};
use std::path::Path;

use numpy::{PyArray1, PyArray4, PyArrayMethods};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PySlice;
use pyo3::wrap_pyfunction;

use crate::core::{
    load_frames_from_bytes, load_frames_from_path, FrameRange, LoadOptions, ResizeOptions,
    VideoBatch, VideoError,
};

#[pyfunction]
#[pyo3(signature = (path, frame_range=None, resize=None))]
fn load_numpy_frames_from_path(
    py: Python<'_>,
    path: &str,
    frame_range: Option<&Bound<'_, PySlice>>,
    resize: Option<(usize, usize)>,
) -> PyResult<Py<PyArray4<u8>>> {
    let options = build_load_options(frame_range, resize)?;
    let batch = load_frames_from_path(Path::new(path), options)?;
    batch_to_numpy(py, batch)
}

#[pyfunction]
#[pyo3(signature = (data, frame_range=None, resize=None))]
fn load_numpy_frames_from_bytes(
    py: Python<'_>,
    data: &[u8],
    frame_range: Option<&Bound<'_, PySlice>>,
    resize: Option<(usize, usize)>,
) -> PyResult<Py<PyArray4<u8>>> {
    let options = build_load_options(frame_range, resize)?;
    let batch = load_frames_from_bytes(data, options)?;
    batch_to_numpy(py, batch)
}

fn batch_to_numpy(py: Python<'_>, batch: VideoBatch) -> PyResult<Py<PyArray4<u8>>> {
    let dims = [batch.frame_count, batch.height, batch.width, batch.channels];
    let array = PyArray1::<u8>::from_vec(py, batch.frames);
    let array = array
        .reshape(dims)
        .map_err(|err| PyRuntimeError::new_err(format!("numpy reshape failed: {err}")))?;
    Ok(array.into())
}

fn build_load_options(
    frame_range: Option<&Bound<'_, PySlice>>,
    resize: Option<(usize, usize)>,
) -> PyResult<LoadOptions> {
    let mut options = LoadOptions::default();

    if let Some(slice) = frame_range {
        options.frame_range = parse_frame_range(slice)?;
    }

    if let Some((width, height)) = resize {
        options.resize = Some(parse_resize(width, height)?);
    }

    Ok(options)
}

fn parse_frame_range(slice: &Bound<'_, PySlice>) -> PyResult<FrameRange> {
    let start_raw: Option<isize> = slice.getattr("start")?.extract()?;
    let stop_raw: Option<isize> = slice.getattr("stop")?.extract()?;
    let step_raw: Option<isize> = slice.getattr("step")?.extract()?;

    let start = start_raw
        .map(|value| ensure_non_negative(value, "frame_range.start"))
        .transpose()?
        .unwrap_or(0);
    let end = stop_raw
        .map(|value| ensure_non_negative(value, "frame_range.stop"))
        .transpose()?;

    let step_value = step_raw.unwrap_or(1);
    if step_value <= 0 {
        return Err(PyValueError::new_err("frame_range.step must be positive"));
    }
    let step = NonZeroUsize::new(step_value as usize)
        .ok_or_else(|| PyValueError::new_err("frame_range.step cannot be zero"))?;

    FrameRange::new(start, end, step).map_err(PyErr::from)
}

fn parse_resize(width: usize, height: usize) -> PyResult<ResizeOptions> {
    if width == 0 || height == 0 {
        return Err(PyValueError::new_err("resize dimensions must be positive"));
    }
    let width = NonZeroU32::new(
        u32::try_from(width).map_err(|_| PyValueError::new_err("resize width too large"))?,
    )
    .ok_or_else(|| PyValueError::new_err("resize width must be > 0"))?;
    let height = NonZeroU32::new(
        u32::try_from(height).map_err(|_| PyValueError::new_err("resize height too large"))?,
    )
    .ok_or_else(|| PyValueError::new_err("resize height must be > 0"))?;
    Ok(ResizeOptions::new(width, height))
}

fn ensure_non_negative(value: isize, label: &str) -> PyResult<usize> {
    if value < 0 {
        Err(PyValueError::new_err(format!("{label} must be >= 0")))
    } else {
        Ok(value as usize)
    }
}

impl From<VideoError> for PyErr {
    fn from(value: VideoError) -> Self {
        match value {
            VideoError::Ffmpeg(_) | VideoError::Decode(_) => {
                PyRuntimeError::new_err(value.to_string())
            }
            VideoError::Io(_) => PyIOError::new_err(value.to_string()),
            VideoError::StreamNotFound => PyValueError::new_err(value.to_string()),
            VideoError::InvalidRange(_) | VideoError::InvalidResize(_) => {
                PyValueError::new_err(value.to_string())
            }
        }
    }
}

#[pymodule]
fn _kclip(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_numpy_frames_from_path, m)?)?;
    m.add_function(wrap_pyfunction!(load_numpy_frames_from_bytes, m)?)?;
    Ok(())
}
