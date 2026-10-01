//! The Python-facing reader: plans a batch of clips natively and decodes it on the pool, all
//! without the GIL.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use numpy::{PyArray2, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::decode::decode_clip;
use crate::read::plan::{self, Augment, ClipPlan, Mode};
use crate::read::pool::{self, Pool};
use crate::read::sample;
use crate::storage::shard::Shard;

/// Metadata of one video.
#[pyclass(module = "kohakuclip._core", frozen, get_all)]
pub struct VideoInfo {
    /// Number of frames.
    pub n: usize,
    pub fps: f64,
    /// Stored height and width.
    pub h: u32,
    pub w: u32,
    pub codec: &'static str,
    pub colorspace: String,
    /// The source (archive or folder) and the mp4's name in it.
    pub source: PathBuf,
    pub name: String,
}

/// A decoded batch. ``video``: rgb uint8 [B, T, 3, S, S]; yuv modes uint8 [B, T, H * W * 3 / 2]
/// (Y, U, V planes). For yuv modes also the plane size per clip ([B, 2]), the flips the GPU
/// applies ([B, 2]: horizontal, vertical) and the colorspace per clip.
#[pyclass(module = "kohakuclip._core", frozen, get_all)]
pub struct Batch {
    pub video: Py<PyAny>,
    pub window: Option<Py<PyAny>>,
    pub flips: Option<Py<PyAny>>,
    pub colorspace: Option<Vec<String>>,
}

/// A submitted batch; ``result()`` waits for it.
#[pyclass(module = "kohakuclip._core")]
pub struct Pending {
    done: Arc<pool::Batch<Result<(), String>>>,
    batch: Option<Batch>,
}

#[pymethods]
impl Pending {
    /// Wait (without the GIL) until the batch is decoded and return it.
    fn result(&mut self, py: Python<'_>) -> PyResult<Batch> {
        let done = self.done.clone();
        let results = py.detach(move || done.wait());
        let failed: Vec<String> = results
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.as_ref().err().map(|e| format!("clip {i}: {e}")))
            .collect();
        if !failed.is_empty() {
            return Err(PyRuntimeError::new_err(failed.join("; ")));
        }
        self.batch
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("result() was already taken"))
    }
}

impl Drop for Pending {
    /// A batch dropped before `result()` still has decode threads writing into its output
    /// array: wait for them before the array can be freed.
    fn drop(&mut self) {
        if self.batch.is_some() {
            self.done.wait();
        }
    }
}

/// Raw output pointer handed to the decode threads.
struct Out(*mut u8, usize);

// SAFETY: every clip writes only its own disjoint region of the output array, and the array is
// kept alive (owned by the pending `Batch`) until all tasks are waited for
unsafe impl Send for Out {}

/// Reads batches of clips from zip / tar archives and folders of mp4 files.
#[pyclass(module = "kohakuclip._core", frozen)]
pub struct Reader {
    shards: Vec<Shard>,
    /// (shard, video in shard) per video id.
    videos: Vec<(usize, usize)>,
    size: u32,
    mode: Mode,
    augment: Augment,
    skip: bool,
    rng: Mutex<StdRng>,
    pool: Pool,
}

impl Reader {
    /// The shard of video `vid` and its index there (plain Rust: usable without the GIL).
    fn find(&self, vid: usize) -> Result<(&Shard, usize), String> {
        let &(shard, i) = self
            .videos
            .get(vid)
            .ok_or_else(|| format!("video {vid} of {}", self.videos.len()))?;
        Ok((&self.shards[shard], i))
    }

    fn locate(&self, vid: usize) -> PyResult<(&Shard, usize)> {
        self.find(vid).map_err(PyIndexError::new_err)
    }

    /// Run `f` with the generator for one call: seeded from `seed` (reproducible, e.g. a function
    /// of the step for exact resumption), or the reader's own stream.
    fn with_rng<T>(&self, seed: Option<u64>, f: impl FnOnce(&mut StdRng) -> T) -> T {
        match seed {
            Some(seed) => f(&mut StdRng::seed_from_u64(seed)),
            None => f(&mut self.rng.lock().unwrap()),
        }
    }

    /// Plan every clip (all native; the GIL is not needed).
    fn plan_batch(
        &self,
        items: &[(usize, Vec<u32>)],
        seed: Option<u64>,
    ) -> Result<(Vec<ClipPlan>, u32), String> {
        let mut metas = Vec::with_capacity(items.len());
        for (vid, frames) in items {
            let (shard, i) = self.find(*vid)?;
            let video = shard.video(i).map_err(|e| e.to_string())?;
            let meta = video.meta();
            if let Some(&bad) = frames.iter().find(|&&f| f as usize >= meta.n) {
                return Err(format!("video {vid}: frame {bad} of {}", meta.n));
            }
            metas.push((shard, i, video));
        }

        // yuv: one window side for the batch, the smallest stored short side (even)
        let side = if self.mode == Mode::Yuv {
            let short_sides = metas.iter().map(|(_, _, v)| v.meta().h.min(v.meta().w));
            short_sides.min().unwrap_or(0) & !1
        } else {
            self.size
        };

        let plans = self.with_rng(seed, |rng| {
            let mut plans = Vec::with_capacity(items.len());
            for ((_, frames), (shard, i, video)) in items.iter().zip(&metas) {
                let file = shard.file(*i).map_err(|e| e.to_string())?;
                plans.push(plan::plan(
                    video.meta(),
                    video.frames(),
                    file,
                    frames,
                    self.size,
                    self.mode,
                    side,
                    &self.augment,
                    self.skip,
                    rng,
                ));
            }
            Ok::<_, String>(plans)
        })?;
        Ok((plans, side))
    }
}

#[pymethods]
impl Reader {
    /// ``sources``: zip / tar archives and folders (every ``*.mp4`` below them).
    /// ``size``: output side (short side resized to it, then a size x size crop).
    /// ``mode``: "rgb" | "yuv" | "yuv_resized". ``threads``: decode threads.
    /// ``crop``: "random" | "center"; ``hflip`` / ``vflip``: flip probabilities.
    /// ``skip``: decode only what later frames need of unwanted samples (needs an index).
    /// ``use_index``: False ignores index members (parse each mp4's moov instead).
    #[new]
    #[pyo3(signature = (
        sources,
        size = 256,
        mode = "rgb",
        threads = 1,
        crop = "random",
        hflip = 0.0,
        vflip = 0.0,
        seed = None,
        skip = true,
        use_index = true,
        open_files = 4096,
        moov_cache = 4096,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        sources: Vec<PathBuf>,
        size: u32,
        mode: &str,
        threads: usize,
        crop: &str,
        hflip: f64,
        vflip: f64,
        seed: Option<u64>,
        skip: bool,
        use_index: bool,
        open_files: usize,
        moov_cache: usize,
    ) -> PyResult<Self> {
        let mode = match mode {
            "rgb" => Mode::Rgb,
            "yuv" => Mode::Yuv,
            "yuv_resized" => Mode::YuvResized,
            other => return Err(PyValueError::new_err(format!("unknown mode {other:?}"))),
        };
        let random_crop = match crop {
            "random" => true,
            "center" => false,
            other => return Err(PyValueError::new_err(format!("unknown crop {other:?}"))),
        };
        if mode != Mode::Rgb && size % 2 == 1 {
            return Err(PyValueError::new_err("yuv modes need an even size"));
        }

        let shards = py
            .detach(|| {
                sources
                    .iter()
                    .map(|p| Shard::open(p, use_index, open_files, moov_cache))
                    .collect::<std::io::Result<Vec<_>>>()
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let videos = shards
            .iter()
            .enumerate()
            .flat_map(|(s, shard)| (0..shard.len()).map(move |i| (s, i)))
            .collect();
        let rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => rand::make_rng(),
        };

        Ok(Self {
            shards,
            videos,
            size,
            mode,
            augment: Augment {
                random_crop,
                hflip,
                vflip,
            },
            skip,
            rng: Mutex::new(rng),
            pool: Pool::new(threads),
        })
    }

    /// Output side.
    #[getter]
    fn size(&self) -> u32 {
        self.size
    }

    /// "rgb", "yuv" or "yuv_resized".
    #[getter]
    fn mode(&self) -> &'static str {
        match self.mode {
            Mode::Rgb => "rgb",
            Mode::Yuv => "yuv",
            Mode::YuvResized => "yuv_resized",
        }
    }

    fn __len__(&self) -> usize {
        self.videos.len()
    }

    fn info(&self, vid: usize) -> PyResult<VideoInfo> {
        let (shard, i) = self.locate(vid)?;
        let video = shard
            .video(i)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let meta = video.meta();
        Ok(VideoInfo {
            n: meta.n,
            fps: meta.fps,
            h: meta.h,
            w: meta.w,
            codec: meta.codec.name(),
            colorspace: meta.colorspace.clone(),
            source: shard.path.clone(),
            name: shard.members[i].name.clone(),
        })
    }

    /// ``n`` clips (video id, frames) of random videos, or of the given ``videos``: each
    /// ``frames`` frames resampled to ``fps`` (None: native rate) from a random start, or
    /// ``frames`` random frames of the whole video if ``spread``. ``seed``: draw from a generator
    /// seeded with it instead of the reader's stream (reproducible batches).
    #[pyo3(signature = (n, frames, fps = None, spread = false, seed = None, videos = None))]
    #[allow(clippy::too_many_arguments)]
    fn sample(
        &self,
        n: usize,
        frames: usize,
        fps: Option<f64>,
        spread: bool,
        seed: Option<u64>,
        videos: Option<Vec<usize>>,
    ) -> PyResult<Vec<(usize, Vec<u32>)>> {
        self.with_rng(seed, |rng| {
            let count = videos.as_ref().map_or(n, Vec::len);
            let mut items = Vec::with_capacity(count);
            for k in 0..count {
                let vid = match &videos {
                    Some(videos) => videos[k],
                    None => rand::RngExt::random_range(rng, 0..self.videos.len()),
                };
                let (shard, i) = self.locate(vid)?;
                let video = shard
                    .video(i)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let meta = video.meta();
                let picked = if spread {
                    sample::random_frames(meta.n, frames, rng)
                } else {
                    sample::clip(meta.n, meta.fps, frames, fps, rng)
                };
                items.push((vid, picked));
            }
            Ok(items)
        })
    }

    /// Queue a batch for decoding and return at once. ``items``: (video id, frame indices) per
    /// clip, all with the same number of frames. ``out``: optional uint8 array to decode into
    /// (e.g. a pinned tensor's ``.numpy()``), of the batch's output shape. ``seed``: draw crops
    /// and flips from a generator seeded with it (reproducible batches).
    #[pyo3(signature = (items, out = None, seed = None))]
    fn submit<'py>(
        &self,
        py: Python<'py>,
        items: Vec<(usize, Vec<u32>)>,
        out: Option<Bound<'py, PyArrayDyn<u8>>>,
        seed: Option<u64>,
    ) -> PyResult<Pending> {
        let frames = items.first().map_or(0, |(_, f)| f.len());
        if items.iter().any(|(_, f)| f.len() != frames || f.is_empty()) {
            return Err(PyValueError::new_err(
                "every clip needs the same, nonzero number of frames",
            ));
        }
        let (plans, side) = py
            .detach(|| self.plan_batch(&items, seed))
            .map_err(PyValueError::new_err)?;

        let per_frame = match self.mode {
            Mode::Rgb => vec![3, self.size as usize, self.size as usize],
            Mode::Yuv => vec![(side * side * 3 / 2) as usize],
            Mode::YuvResized => vec![(self.size * self.size * 3 / 2) as usize],
        };
        let mut shape = vec![items.len(), frames];
        shape.extend(&per_frame);
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
        let clip_len = out.len() / items.len().max(1);
        // SAFETY: the array is C-contiguous with `items.len() * clip_len` bytes (checked above)
        let base = unsafe { out.as_array_mut().as_mut_ptr() };

        // yuv modes flip on the GPU: hand the planned flips over with the batch
        let flips: Vec<Vec<i8>> = plans
            .iter()
            .map(|p| vec![p.geometry.hflip as i8, p.geometry.vflip as i8])
            .collect();
        let colorspace: Vec<String> = plans.iter().map(|p| p.colorspace.clone()).collect();

        let done = pool::Batch::new(plans.len());
        for (i, plan) in plans.into_iter().enumerate() {
            // SAFETY: clip i owns bytes [i * clip_len, (i + 1) * clip_len) of the array
            let region = Out(unsafe { base.add(i * clip_len) }, clip_len);
            let done = done.clone();
            self.pool.spawn(move || {
                let region = region;
                // SAFETY: see `Out`: a disjoint region of a live array
                let out = unsafe { std::slice::from_raw_parts_mut(region.0, region.1) };
                done.finish(i, decode_clip(&plan, out));
            });
        }

        let video = out.into_any().unbind();
        let batch = if self.mode == Mode::Rgb {
            Batch {
                video,
                window: None,
                flips: None,
                colorspace: None,
            }
        } else {
            let window_side = if self.mode == Mode::Yuv {
                side
            } else {
                self.size
            };
            let window = vec![vec![window_side, window_side]; items.len()];
            Batch {
                video,
                window: Some(PyArray2::from_vec2(py, &window)?.into_any().unbind()),
                flips: Some(PyArray2::from_vec2(py, &flips)?.into_any().unbind()),
                colorspace: Some(colorspace),
            }
        };
        Ok(Pending {
            done,
            batch: Some(batch),
        })
    }

    /// ``submit(items, out, seed).result()``.
    #[pyo3(signature = (items, out = None, seed = None))]
    fn read<'py>(
        &self,
        py: Python<'py>,
        items: Vec<(usize, Vec<u32>)>,
        out: Option<Bound<'py, PyArrayDyn<u8>>>,
        seed: Option<u64>,
    ) -> PyResult<Batch> {
        self.submit(py, items, out, seed)?.result(py)
    }
}

/// Cumulative seconds per native stage (read, decode, convert, resize; summed over threads) and
/// frames emitted / decoded since the last reset.
#[pyfunction]
#[pyo3(signature = (reset = false))]
pub fn profile(py: Python<'_>, reset: bool) -> PyResult<Py<PyAny>> {
    let (seconds, emitted, decoded) = crate::decode::profile::profile(reset);
    let stats = pyo3::types::PyDict::new(py);
    for (name, s) in ["read", "decode", "convert", "resize"].iter().zip(seconds) {
        stats.set_item(name, s)?;
    }
    stats.set_item("frames", emitted)?;
    stats.set_item("decoded", decoded)?;
    Ok(stats.into_any().unbind())
}
