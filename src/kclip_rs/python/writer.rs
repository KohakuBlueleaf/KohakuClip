//! Python functions of the writer.

use std::path::PathBuf;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::codec::ImageCodec;
use crate::codec::jpeg::Chroma;
use crate::write::{
    Encoding, ImageEncoding, ImageMember, JpegEncoder, Outcome, Source, encode, encode_image, pack,
    pack_images,
};

/// The default JPEG quality: mozjpeg 4:4:4 at this quality lands at 41-42 dB PSNR on random
/// resized crops (see README).
const QUALITY: u32 = 70;

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

/// (the stored bytes or None, stored w, h, source w, h)
type EncodedImage<'py> = (Option<Bound<'py, PyBytes>>, u32, u32, u32, u32);

/// Encode one image for storage; returns (bytes or None, w, h, source w, source h): None when
/// the input's short side is under ``min_short`` (not stored; w, h = 0). ``source``: a path or
/// the bytes of any image file (JPEG through libjpeg-turbo, others through FFmpeg). JPEGs are
/// encoded by ``jpeg_encoder``: "mozjpeg" (default) or "turbo" (libjpeg-turbo). The stored
/// image is upright (EXIF orientation applied), its short side capped at ``max_short``. Runs
/// without the GIL.
#[pyfunction(name = "encode_image")]
#[pyo3(signature = (
    source,
    codec = "jpeg",
    quality = QUALITY,
    chroma = "444",
    crf = 30,
    distance = 1.0,
    effort = None,
    max_short = 512,
    min_short = 384,
    dct_scale = false,
    jpeg_encoder = "mozjpeg",
))]
#[allow(clippy::too_many_arguments)]
pub fn py_encode_image<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    codec: &str,
    quality: u32,
    chroma: &str,
    crf: u32,
    distance: f32,
    effort: Option<u32>,
    max_short: u32,
    min_short: u32,
    dct_scale: bool,
    jpeg_encoder: &str,
) -> PyResult<EncodedImage<'py>> {
    let jpeg_encoder = JpegEncoder::from_name(jpeg_encoder).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown jpeg_encoder {jpeg_encoder:?} (mozjpeg, turbo)"
        ))
    })?;
    let codec = ImageCodec::from_name(codec)
        .filter(|&c| c != ImageCodec::Png)
        .ok_or_else(|| {
            PyValueError::new_err(format!("unknown codec {codec:?} (jpeg, av1, jxl, webp)"))
        })?;
    let chroma = match chroma {
        "420" => Chroma::Yuv420,
        "444" => Chroma::Yuv444,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown chroma {other:?} (420, 444)"
            )));
        }
    };
    let enc = ImageEncoding {
        codec,
        jpeg_encoder,
        quality,
        chroma,
        crf,
        distance,
        effort,
        max_short,
        min_short,
        dct_scale,
    };

    let outcome = if let Ok(bytes) = source.cast::<PyBytes>() {
        let data = bytes.as_bytes().to_vec();
        py.detach(|| encode_image(&data, &enc))
    } else {
        let path: PathBuf = source.extract()?;
        py.detach(|| {
            let data = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            encode_image(&data, &enc)
        })
    };
    match outcome.map_err(PyRuntimeError::new_err)? {
        Outcome::Stored(e) => Ok((
            Some(PyBytes::new(py, &e.data)),
            e.w,
            e.h,
            e.source_w,
            e.source_h,
        )),
        Outcome::TooSmall { w, h } => Ok((None, 0, 0, w, h)),
    }
}

/// Write the image shard ``path`` (``.tar``, otherwise a stored zip) from (name, bytes, w, h,
/// codec) members, with its image index as the last member. ``header``: shard-level facts (a
/// json object, e.g. the encoding) stored in the index. Runs without the GIL.
#[pyfunction(name = "pack_images")]
#[pyo3(signature = (path, members, header = "{}"))]
pub fn py_pack_images(
    py: Python<'_>,
    path: PathBuf,
    members: Vec<(String, Vec<u8>, u32, u32, String)>,
    header: &str,
) -> PyResult<()> {
    let header: serde_json::Value =
        serde_json::from_str(header).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let members = members
        .into_iter()
        .map(|(name, data, w, h, codec)| {
            let codec = ImageCodec::from_name(&codec)
                .ok_or_else(|| PyValueError::new_err(format!("unknown codec {codec:?}")))?;
            Ok(ImageMember {
                name,
                data,
                w,
                h,
                codec,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    py.detach(|| pack_images(&path, &members, header))
        .map_err(PyRuntimeError::new_err)
}
