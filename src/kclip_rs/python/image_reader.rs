//! The Python-facing image reader: plans a batch of images natively and decodes it on the
//! pool, all without the GIL.

use std::path::PathBuf;
use std::sync::Mutex;

use numpy::{PyArray2, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::pending::{OutRegion, Pending};
use crate::codec::{ImageCodec, decode_still, jpeg};
use crate::decode::decode_image;
use crate::image::Color;
use crate::read::image_plan::{self, Crop, ImagePlan, ImageSettings};
use crate::read::order;
use crate::read::plan::Mode;
use crate::read::pool::{self, Pool};
use crate::storage::image_shard::{ImageEntry, ImageShard};

/// Metadata of one image.
#[pyclass(module = "kohakuclip._core", frozen, get_all)]
pub struct ImageInfo {
    /// Stored height and width.
    pub h: u32,
    pub w: u32,
    pub codec: &'static str,
    /// The source (archive or folder), the image's position in it and its member name.
    pub source: PathBuf,
    pub index: usize,
    pub name: String,
}

/// A decoded batch of images. ``image``: rgb uint8 [B, 3, S, S]; yuv modes uint8
/// [B, H * W * 3 / 2] (Y, U, V planes). For yuv modes also the plane size per image ([B, 2]),
/// the flips the GPU applies ([B, 2]: horizontal, vertical) and the colorspace per image.
#[pyclass(module = "kohakuclip._core", frozen, get_all)]
pub struct ImageBatch {
    pub image: Py<PyAny>,
    pub window: Option<Py<PyAny>>,
    pub flips: Option<Py<PyAny>>,
    pub colorspace: Option<Vec<String>>,
}

/// Reads batches of images from KohakuClip image shards, and from zip / tar archives and
/// folders of image files.
#[pyclass(module = "kohakuclip._core", frozen)]
pub struct ImageReader {
    shards: Vec<ImageShard>,
    /// First image id of each shard (prefix sums of their sizes), and the total.
    starts: Vec<usize>,
    total: usize,
    settings: ImageSettings,
    /// Hint the kernel to read each planned image ahead (see `ImageEntry::readahead`).
    readahead: bool,
    rng: Mutex<StdRng>,
    pool: Pool,
}

impl ImageReader {
    /// The shard of image `id` and the image's index there.
    fn find(&self, id: usize) -> Result<(&ImageShard, usize), String> {
        if id >= self.total {
            return Err(format!("image {id} of {}", self.total));
        }
        let shard = self.starts.partition_point(|&start| start <= id) - 1;
        Ok((&self.shards[shard], id - self.starts[shard]))
    }

    fn entry(&self, id: usize) -> Result<ImageEntry, String> {
        let (shard, i) = self.find(id)?;
        shard.entry(i).map_err(|e| e.to_string())
    }

    /// Run `f` with the generator for one call: seeded from `seed`, or the reader's own stream.
    fn with_rng<T>(&self, seed: Option<u64>, f: impl FnOnce(&mut StdRng) -> T) -> T {
        match seed {
            Some(seed) => f(&mut StdRng::seed_from_u64(seed)),
            None => f(&mut self.rng.lock().unwrap()),
        }
    }

    /// Plan every image (all native; the GIL is not needed). Returns the plans and the yuv
    /// window side.
    fn plan_batch(
        &self,
        ids: &[usize],
        seed: Option<u64>,
    ) -> Result<(Vec<ImagePlan>, u32), String> {
        let entries = ids
            .iter()
            .map(|&id| self.entry(id))
            .collect::<Result<Vec<_>, _>>()?;
        if self.readahead {
            for entry in &entries {
                entry.readahead();
            }
        }

        let mut side = self.settings.size;
        let rgb_coded = entries.iter().find(|e| e.color == Color::Rgb);
        if let Some(entry) = rgb_coded.filter(|_| self.settings.mode != Mode::Rgb) {
            let codec = entry.codec.name();
            return Err(format!("yuv modes need YUV-coded images, not {codec}"));
        }
        if self.settings.mode == Mode::Yuv {
            if entries.iter().any(|e| e.w == 0) {
                return Err("the yuv mode needs indexed shards (stored sizes)".into());
            }
            // one window side for the batch: the smallest stored short side (even)
            side = entries.iter().map(|e| e.h.min(e.w)).min().unwrap_or(0) & !1;
        }

        let plans = self.with_rng(seed, |rng| {
            entries
                .into_iter()
                .map(|entry| image_plan::plan(entry, side, rng))
                .collect()
        });
        Ok((plans, side))
    }
}

/// The stored (h, w) of an image whose source has no index: from its header (JPEG) or by
/// decoding it.
fn probe_size(entry: &ImageEntry) -> Result<(u32, u32), String> {
    let mut bytes = Vec::new();
    entry
        .read_into(&mut bytes)
        .map_err(|e| format!("read: {e}"))?;
    if entry.codec == ImageCodec::Jpeg {
        let header = jpeg::Decoder::new()?.header(&bytes)?;
        return Ok((header.h, header.w));
    }
    decode_still(entry.codec, &bytes, |picture| {
        let (h, w) = picture.size();
        Ok((h as u32, w as u32))
    })
}

/// Open every source on up to `threads` threads (thousands of shards open in parallel).
fn open_shards(
    sources: &[PathBuf],
    use_index: bool,
    open_files: usize,
    threads: usize,
) -> std::io::Result<Vec<ImageShard>> {
    let workers = threads.clamp(1, 32).min(sources.len().max(1));
    let chunk = sources.len().div_ceil(workers).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = sources
            .chunks(chunk)
            .map(|paths| {
                scope.spawn(move || {
                    paths
                        .iter()
                        .map(|p| ImageShard::open(p, use_index, open_files))
                        .collect::<std::io::Result<Vec<_>>>()
                })
            })
            .collect();
        let mut shards = Vec::with_capacity(sources.len());
        for handle in handles {
            shards.extend(handle.join().expect("a shard-opening thread panicked")?);
        }
        Ok(shards)
    })
}

#[pymethods]
impl ImageReader {
    /// ``sources``: KohakuClip image shards, zip / tar archives and folders of image files
    /// (jpg, obu, jxl, webp, png). ``size``: output side. ``mode``: "rgb" | "yuv" |
    /// "yuv_resized". ``threads``: decode threads. ``crop``: "random" | "center" (short side
    /// resized to ``size``, then a square crop) | "resized" (RandomResizedCrop with ``scale``
    /// and ``ratio``). ``hflip`` / ``vflip``: flip probabilities. ``dct_scale``: let the JPEG
    /// decoder shrink by 1/2, 1/4 or 1/8 when the output needs no more pixels. ``readahead``:
    /// hint the kernel to read a submitted batch's images ahead of their decode threads.
    /// ``use_index``: False ignores index members (lists the archive instead).
    #[new]
    #[pyo3(signature = (
        sources,
        size = 256,
        mode = "rgb",
        threads = 1,
        crop = "random",
        scale = (0.08, 1.0),
        ratio = (0.75, 4.0 / 3.0),
        hflip = 0.0,
        vflip = 0.0,
        seed = None,
        dct_scale = true,
        readahead = true,
        use_index = true,
        open_files = 4096,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        sources: Vec<PathBuf>,
        size: u32,
        mode: &str,
        threads: usize,
        crop: &str,
        scale: (f64, f64),
        ratio: (f64, f64),
        hflip: f64,
        vflip: f64,
        seed: Option<u64>,
        dct_scale: bool,
        readahead: bool,
        use_index: bool,
        open_files: usize,
    ) -> PyResult<Self> {
        let mode = match mode {
            "rgb" => Mode::Rgb,
            "yuv" => Mode::Yuv,
            "yuv_resized" => Mode::YuvResized,
            other => return Err(PyValueError::new_err(format!("unknown mode {other:?}"))),
        };
        let crop = match crop {
            "random" => Crop::Short { random: true },
            "center" => Crop::Short { random: false },
            "resized" => Crop::Resized { scale, ratio },
            other => return Err(PyValueError::new_err(format!("unknown crop {other:?}"))),
        };
        if mode != Mode::Rgb && size % 2 == 1 {
            return Err(PyValueError::new_err("yuv modes need an even size"));
        }
        let valid_range = |(lo, hi): (f64, f64)| lo > 0.0 && lo <= hi;
        if !valid_range(scale) || scale.1 > 1.0 || !valid_range(ratio) {
            return Err(PyValueError::new_err(
                "scale needs 0 < lo <= hi <= 1, ratio 0 < lo <= hi",
            ));
        }

        let shards = py
            .detach(|| open_shards(&sources, use_index, open_files, threads))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut starts = Vec::with_capacity(shards.len());
        let mut total = 0;
        for shard in &shards {
            starts.push(total);
            total += shard.len();
        }
        let rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => rand::make_rng(),
        };

        Ok(Self {
            shards,
            starts,
            total,
            settings: ImageSettings {
                size,
                mode,
                crop,
                hflip,
                vflip,
                dct_scale,
            },
            readahead,
            rng: Mutex::new(rng),
            pool: Pool::new(threads),
        })
    }

    /// Output side.
    #[getter]
    fn size(&self) -> u32 {
        self.settings.size
    }

    /// "rgb", "yuv" or "yuv_resized".
    #[getter]
    fn mode(&self) -> &'static str {
        match self.settings.mode {
            Mode::Rgb => "rgb",
            Mode::Yuv => "yuv",
            Mode::YuvResized => "yuv_resized",
        }
    }

    fn __len__(&self) -> usize {
        self.total
    }

    fn info(&self, py: Python<'_>, image: usize) -> PyResult<ImageInfo> {
        let (shard, i) = self.find(image).map_err(PyIndexError::new_err)?;
        let entry = shard
            .entry(i)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (h, w) = if entry.w == 0 {
            py.detach(|| probe_size(&entry))
                .map_err(PyValueError::new_err)?
        } else {
            (entry.h, entry.w)
        };
        let name = shard
            .name(i)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(ImageInfo {
            h,
            w,
            codec: entry.codec.name(),
            source: shard.path.clone(),
            index: i,
            name,
        })
    }

    /// ``n`` image ids drawn uniformly (with replacement). ``seed``: draw from a generator
    /// seeded with it instead of the reader's stream (reproducible batches).
    #[pyo3(signature = (n, seed = None))]
    fn sample(&self, n: usize, seed: Option<u64>) -> PyResult<Vec<usize>> {
        if self.total == 0 {
            return Err(PyValueError::new_err("no images"));
        }
        Ok(self.with_rng(seed, |rng| {
            (0..n).map(|_| rng.random_range(0..self.total)).collect()
        }))
    }

    /// Queue a batch of images for decoding and return at once. ``images``: image ids.
    /// ``out``: optional uint8 array to decode into (e.g. a pinned tensor's ``.numpy()``), of
    /// the batch's output shape. ``seed``: draw crops and flips from a generator seeded with it
    /// (reproducible batches).
    #[pyo3(signature = (images, out = None, seed = None))]
    fn submit<'py>(
        &self,
        py: Python<'py>,
        images: Vec<usize>,
        out: Option<Bound<'py, PyArrayDyn<u8>>>,
        seed: Option<u64>,
    ) -> PyResult<Pending> {
        if images.is_empty() {
            return Err(PyValueError::new_err("an empty batch"));
        }
        let (plans, side) = py
            .detach(|| self.plan_batch(&images, seed))
            .map_err(PyValueError::new_err)?;

        let size = self.settings.size as usize;
        let mut shape = vec![images.len()];
        match self.settings.mode {
            Mode::Rgb => shape.extend([3, size, size]),
            Mode::Yuv => shape.push(side as usize * side as usize * 3 / 2),
            Mode::YuvResized => shape.push(size * size * 3 / 2),
        }
        let out = match out {
            Some(out) => {
                if out.shape() != shape.as_slice() || !out.is_c_contiguous() {
                    return Err(PyValueError::new_err(format!(
                        "out must be C-contiguous {shape:?}"
                    )));
                }
                out
            }
            None => PyArrayDyn::<u8>::zeros(py, shape.as_slice(), false),
        };
        let image_len = out.len() / images.len();
        // SAFETY: the array is C-contiguous with `images.len() * image_len` bytes (checked)
        let base = unsafe { out.as_array_mut().as_mut_ptr() };

        // yuv modes flip on the GPU: hand the planned flips over with the batch
        let yuv = self.settings.mode != Mode::Rgb;
        let mut flips = Vec::with_capacity(plans.len());
        let mut colorspace = Vec::with_capacity(plans.len());
        if yuv {
            for plan in &plans {
                // the flips come from the image's own seed: the same draw its decode makes
                let (hflip, vflip) = image_plan::flips(&self.settings, plan);
                flips.push(vec![hflip as i8, vflip as i8]);
                colorspace.push(plan.entry.color.name().to_string());
            }
        }

        let done = pool::Batch::new(plans.len());
        for (i, plan) in plans.into_iter().enumerate() {
            // SAFETY: image i owns bytes [i * image_len, (i + 1) * image_len) of the array
            let region = unsafe { OutRegion::new(base, i, image_len) };
            let done = done.clone();
            let settings = self.settings;
            self.pool.spawn(move || {
                // SAFETY: the pending batch keeps the array alive until every task finished
                let out = unsafe { region.into_slice() };
                done.finish(i, decode_image(&plan, &settings, out));
            });
        }

        let image = out.into_any().unbind();
        let batch = if yuv {
            let window_side = if self.settings.mode == Mode::Yuv {
                side
            } else {
                self.settings.size
            };
            let window = vec![vec![window_side, window_side]; images.len()];
            ImageBatch {
                image,
                window: Some(PyArray2::from_vec2(py, &window)?.into_any().unbind()),
                flips: Some(PyArray2::from_vec2(py, &flips)?.into_any().unbind()),
                colorspace: Some(colorspace),
            }
        } else {
            ImageBatch {
                image,
                window: None,
                flips: None,
                colorspace: None,
            }
        };
        Ok(Pending::new(done, Py::new(py, batch)?.into_any(), "image"))
    }

    /// ``submit(images, out, seed).result()``.
    #[pyo3(signature = (images, out = None, seed = None))]
    fn read<'py>(
        &self,
        py: Python<'py>,
        images: Vec<usize>,
        out: Option<Bound<'py, PyArrayDyn<u8>>>,
        seed: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        self.submit(py, images, out, seed)?.result(py)
    }
}

/// The images at ``positions`` of a seeded permutation of [0, n): an epoch order evaluated per
/// batch (no O(n) table).
#[pyfunction]
pub fn permute(n: u64, seed: u64, positions: Vec<u64>) -> PyResult<Vec<u64>> {
    if let Some(&bad) = positions.iter().find(|&&p| p >= n) {
        return Err(PyIndexError::new_err(format!("position {bad} of {n}")));
    }
    Ok(positions
        .into_iter()
        .map(|p| order::permute(n, seed, p))
        .collect())
}
