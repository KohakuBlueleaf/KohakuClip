# KohakuClip

Fast random-access video clips and images for training. Videos are stored as mp4 files (AV1,
H.264 or HEVC) in zip or tar shards with an in-archive frame index, or read straight from folders
and archives of mp4 files. Each batch of clips is planned and decoded natively (Rust, on the
system's FFmpeg libraries) on a persistent thread pool, straight into a (pinned) uint8 buffer,
without the GIL. Images work the same way ([Images](#images)): JPEG shards with a fixed-size
index, written with mozjpeg and decoded by libjpeg-turbo.

```python
from kohakuclip import Reader

reader = Reader(
    ["data/shard_00000.zip", "data/more.tar", "data/mp4_folder"],
    size=256,
    threads=16,
    hflip=0.5,
)
# 16 clips of 8 frames at 6 fps from random videos and random starts
items = reader.sample(16, frames=8, fps=6.0)
# uint8 [16, 8, 3, 256, 256]
video = reader.read(items).video
```

`items` is a list of (video id, frame indices); any sampler works (`reader.info(vid)` gives the
frame count, fps, size and codec, `kohakuclip.clip` / `random_frames` are the stock samplers).

## In a PyTorch training loop

`kohakuclip.torch.ClipDataset` is an `IterableDataset` whose items are whole batches; iterate it
directly (decoding in this process, into pinned memory) or through a `DataLoader`.
`ClipLoader` runs either way and keeps the resume state:

```python
from kohakuclip.torch import ClipDataset, ClipLoader

dataset = ClipDataset(
    shards, batch_size=16, frames=8, fps=6.0, threads=8, hflip=0.5, seed=0
)
loader = ClipLoader(dataset, workers=0)  # or workers=2: DataLoader worker processes
for video in loader:  # uint8 [16, 8, 3, 256, 256]
    video = video.cuda(non_blocking=True).float() / 127.5 - 1
    ...
checkpoint["loader"] = loader.state_dict()  # resume: loader.load_state_dict(...)
```

- Batch `i` of a rank is a pure function of (`seed`, rank, `i`): the same batches whatever the
  number of DataLoader workers, and a resumed run continues with exactly the batches the
  original would have seen next (state: the seed and a batch count).
- Ranks come from `torch.distributed` (or `RANK` / `WORLD_SIZE`); each rank draws its own
  batches, no sampler needed.
- `epochs=True` visits every video once per epoch in a seeded order, split over ranks;
  the default draws videos at random.
- Lightning: pass `ClipLoader(dataset, lookahead=1)` as the train loader. Lightning stores its
  state in every checkpoint and restores it on `ckpt_path=...`; `lookahead=1` because its
  training loop fetches one batch ahead of the one it trains on.

Examples: `examples/torch_loop.py` (a plain loop with checkpoint and resume) and
`examples/lightning_module.py` (Lightning, DDP, resume).

Loader alone (`benchmarks/loader_bench.py`: no model, 16 clips of 8 frames at 6 fps per batch,
copied to a B300; Xeon 6747P):

| setup | clips/s | training thread CPU per batch |
|---|---|---|
| in process, 8 threads | 320 | 0.11 ms |
| in process, 16 threads | 650 | 0.13 ms |
| in process, 24 threads | 886 | 0.32 ms |
| 1 DataLoader worker x 16 threads | 633 | 0.12 ms |
| 2 workers x 8 threads | 642 | 0.11 ms |
| 2 workers x 12 threads | 932 | 0.26 ms |
| 4 workers x 4 threads | 644 | 0.14 ms |

Throughput follows the total number of decode threads, however they are split; the training
thread spends a fraction of a millisecond per batch either way (with workers, the DataLoader's
pin-memory thread also copies each batch once more in the main process). Pinned copies run at
~52 GB/s against ~13 GB/s from pageable memory, and torch's caching host allocator hands out a
pinned buffer in ~3 us, so decoding each batch into a fresh pinned tensor is the cheap path.

In a video pretraining run (a B/16 video encoder + a DiT-S decoder, 2 x B300, 16 clips per GPU)
every setup trains at the same speed within the run-to-run spread (about +-0.1 it/s over 3
interleaved rounds of 500 steps): tar of JPEG with 16 DataLoader workers 7.10 it/s, KohakuClip in
process (8 threads) 7.13, 2 workers x 8 threads 6.99, 4 workers x 4 threads 7.07. A profile of that
loop shows the GPU busy for the whole step, so the loader is not on the critical path at 115 clips/s
per GPU.

Memory (`benchmarks/leak_check.py`, anonymous resident memory): flat over 1500 yuv batches and
100 encode + pack rounds; rgb reading grows during warm-up (per-thread buffers and decoders)
and by under 1 KB per batch afterwards; creating and dropping 200 readers leaves ~17 KB each.

## Writing shards

```bash
kohakuclip-write out_dir videos/ more_videos.tar clip.mp4 --codec av1 --crf 36 --per-shard 1000
```

Sources are video files, folders (every video file below them) and tar archives of videos (read
into memory, nothing is extracted). Each video is re-encoded by the FFmpeg libraries KohakuClip is
linked against: demux and decode anything FFmpeg reads, scale, encode, mux into memory, all
without the GIL, one video per thread (`--workers`). `--backend ffmpeg` runs the same encode on
the `ffmpeg` command line instead (for encoders the linked FFmpeg lacks). Packing and the index
are native in both cases. From Python: `kohakuclip.writer.write(...)`, or `kohakuclip._core.encode`
(a path or the bytes of a video in, mp4 bytes out) and `kohakuclip._core.pack`.

Defaults (measured below): native fps, short side capped at 512 (never upscaled), closed GOP of
16, AV1 via SVT-AV1 preset 6 with the in-loop filters (deblocking, CDEF, loop restoration) off,
faststart mp4, zip (stored) or tar. `--codec h264` / `hevc` use libx264 / libx265 if the FFmpeg
build has them. Untagged sources are tagged as BT.601 (SD) or BT.709 (HD), the usual guess.

The writer also parses the AV1 frame headers and records, per sample, how many bytes later frames
depend on: SVT-AV1 puts half of all frames in a top layer nothing references, so unwanted samples
are decoded only up to their last referenced frame (25 % of samples are dropped entirely, 25 %
truncated), bit-exactly.

## Sources

A `Reader` takes any mix of zip archives (stored members), tar archives and folder trees (every
`*.mp4` below them). Archives written by `kohakuclip-write` carry an index member
(`__index__.bin`): per frame the absolute byte offset, size, the bytes later frames depend on and
a keyframe flag; per video fps, size, codec, colorspace and the decoder's codec configuration. It
is memory-mapped from the archive (shared by all workers through the page cache). Archives without
one, and folders, fall back to parsing each mp4's `moov` box on first use (any box order) and
caching the result; they return the same clips (see `tests`).

## How a clip is read

1. Planning (native, no GIL): per GOP holding wanted frames, one byte range from its keyframe to
   its last wanted frame; unwanted samples cut to the bytes later frames need; the crop and flips.
2. Decoding, per clip on the pool: one `pread` per GOP, a persistent single-threaded decoder per
   thread (libdav1d for AV1, libavcodec for H.264 / HEVC) and one reused frame, then per wanted
   frame an antialiased bilinear resize of each YUV plane fused with the crop (only the source
   window the crop needs is read), conversion of the output pixels to RGB, flips.
3. Batches queued back to back share one FIFO on the pool: no per-batch barrier.

Output modes (storage and disk reads are the same in all three; `kohakuclip.torch.yuv_to_rgb`
finishes the yuv modes on the GPU with one Triton kernel):

| `mode` | CPU, ms per frame | host-to-GPU bytes per frame | GPU per frame (copy + kernel) | vs `rgb` |
|---|---|---|---|---|
| `"rgb"` (default): resize + convert on the CPU | 3.23 | 196,608 (256 x 256 x 3) | 3.8 us + none | - |
| `"yuv_resized"`: planes resized to the crop, 4:2:0 | 3.09 | 98,304 (0.5x) | 1.9 us + 0.8 us | 43.5 dB (chroma at half the output resolution) |
| `"yuv"`: stored-resolution window, 4:2:0 | 3.12 | 393,216 at 512p storage (2x) | 7.3 us + 7.0 us | 49.2 dB |

CPU: 8 frames at 6 fps, one core. GPU: B300, pinned copies, 16 x 8 frames per batch. The Triton
kernel (chroma upsampling, conversion, antialiased resize, flips, scaling in one pass) agrees
with the plain torch version to within float rounding and is 3-9x faster (torch 7.7 / 22.2 us per
frame; `torch.compile` does not help).

## Measurements

VidGen-1M sample (981 videos, 720p H.264 sources, mean 10.5 s, ~28 fps) stored at 910x512,
native fps, AV1 CRF 36, GOP 16, filters off: 4.3 TB per 50M seconds at 768x512 (38.5 dB stored,
39.9 dB on the 256 crop). Cold reads on network storage; Intel Xeon 6747P (2 x 48 cores, 2
threads per core, max 2.7 GHz), 48 hardware threads per job. ms per output frame per core
(256 x 256 rgb crop):

| clip | 1 core | 24 cores |
|---|---|---|
| 8 frames @ 6 fps | 3.23 | 3.50 (857 clips/s) |
| 8 frames @ 24 fps | 1.84 | 1.99 (1506 clips/s) |
| 16 frames @ 6 fps | 2.83 | 3.04 (494 clips/s) |
| 8 random frames (whole video) | 5.21 | 5.73 (524 clips/s) |

Per frame at 8 @ 6 fps, one core: decode 2.66 ms (2.9 frames decoded per output frame), resize
0.19, read 0.31, convert 0.06. Clips from 60-300 s videos cost the same as from 3-30 s ones (one
read per GOP): 3.24 ms at 8 @ 6 fps, 6.56 for 8 random frames (more GOPs per clip).

| setting | effect (ms per output frame, one core) |
|---|---|
| AV1 in-loop filters off (writer) | -20 to -27 % decode, same size, same crop quality |
| decode only what later frames need of unwanted samples (`skip=True`) | 8 @ 6 fps 3.96 -> 3.23, 8 random 6.73 -> 5.21; bit-exact |
| resize YUV planes, then convert (vs convert, then resize) | 4.97 -> 4.22; the two agree at 48.4 dB |
| SIMD resize (AVX-512 VNNI vs the auto-vectorized code) | resize 0.29 -> 0.20 |
| conversion written to auto-vectorize (no per-pixel branches) | convert 0.20 -> 0.06 |
| 2 batches in flight vs synchronous (threads) | 24 cores 5.99 -> 4.53 (= separate processes) |
| index vs moov fallback | within ~3 % |
| this core vs the previous C++ one (same plan, both `-march=native`) | 8 @ 6 fps 3.85 -> 3.23, 8 @ 24 fps 2.25 -> 1.84 |

## Images

```python
from kohakuclip import ImageReader

reader = ImageReader(["data/images/shard_00000.zip"], size=256, threads=16, hflip=0.5)
images = reader.read(reader.sample(256)).image  # uint8 [256, 3, 256, 256]
```

```bash
# mozjpeg 4:4:4 q70, short side <= 512
kohakuclip-write-images out_dir images/ webdataset.tar photo.png
```

`ImageReader` mirrors `Reader`: `sample(n, seed)` draws image ids, `submit(ids, out, seed)` /
`read(...)` decode a batch on the native pool (no GIL), `info(id)` gives the stored size, codec
and member name. Crops: `"random"` / `"center"` (short side resized to `size`, then a square, as
for clips) or `"resized"` (RandomResizedCrop with `scale` and `ratio`); flips; the `rgb`, `yuv`
and `yuv_resized` modes (`yuv_to_rgb` takes image batches too; JPEG is full-range BT.601,
reported as `"bt601-full"`). `kohakuclip.torch.ImageDataset` is the `ClipDataset` of images
(same batches for any worker count, exact resumption through `ClipLoader`, rank-aware, epoch
mode over a seeded permutation evaluated per batch, `kohakuclip.permute`, so epochs over tens of
millions of images need no O(n) table).

Storage: images in stored zip shards, `<key>.jpg` members and an index member (`__index__.bin`,
the last member, found from the end of the central directory without listing the archive): per
image 24 bytes (absolute offset, size, stored width and height, codec, color), memory-mapped, no
per-image allocation (a reader over 55M images costs no more memory than the pages it touches).
Captions and other metadata go to the writer's `manifest.parquet` (shard, index in the shard,
key, stored and source sizes, `native_res`, caption, the json sidecar). Plain sources work too,
through the same decode / resize / crop path: zip archives of images (stored or deflated
members), tar archives and folders (listed; sizes from each image's header).

Writer: any input FFmpeg or libjpeg-turbo reads (JPEG, PNG, WebP, GIF, BMP, TIFF, AVIF, JPEG XL;
files, folders with `<stem>.txt` / `<stem>.json` sidecars, WebDataset tars), decoded natively
(JPEG through libjpeg-turbo, others through FFmpeg, alpha composited onto white), EXIF
orientation applied, short side capped at 512 with Lanczos-3 (never upscaled), images under 384
dropped (`skipped.jsonl`; `min_short=0` keeps every image), encoded with mozjpeg (trellis
quantization, baseline scans, built in) one image per thread without the GIL. Inputs the native
path cannot read (e.g. CMYK JPEGs) are decoded by Pillow and encoded natively; `--backend
pillow` uses Pillow for all. `--jpeg-encoder turbo` encodes with libjpeg-turbo instead (~8x
faster, 5-8 % larger). Other storage codecs: `--codec av1` (libaom, intra, raw OBUs), `jxl`,
`webp` (through FFmpeg; read back through libdav1d, libjxl and libwebp directly). From Python:
`kohakuclip.image_writer.write_images`, or `kohakuclip._core.encode_image` / `pack_images`.

### Storage format (measured)

3000 images (1500 cc12m, 1500 LAION-COCO originals) stored at short side 512 (native when
smaller); quality = PSNR of 256 x 256 random resized crops (area 0.2-1) against the same crop of
the original; every codec's own library called directly at its best still-image settings (a
grid of 705 settings: libavif 1.4.2 with libaom 3.15.1 and SVT-AV1 4.2.0, mozjpeg 4.1.1,
libjpeg-turbo 3.2, libwebp 1.6, libjxl 0.12), each family interpolated to the same quality.
Decoding: CPU only, every format through this reader's path (its own library to native planes,
the resize fused with the crop to 256, RGB), single-threaded decoders checked against the
libraries' own tools (dav1d, djxl, tjbench, `WebPDecodeYUVInto`); one core, and 64 processes on
64 CPUs (32 cores x 2 hyperthreads) of a Xeon 6747P:

| storage, at 43 dB | KB / image | images/s, 1 core | images/s, 64 CPUs | encode ms / image |
|---|---|---|---|---|
| AVIF 4:4:4 (aom speed 6, tune ssimulacra2) | 35.3 | 220 | 10,700 | 158 |
| AVIF 4:2:0 (SVT-AV1 preset 8, tune iq) | 48.3 | 221 | 10,700 | 20 |
| **JPEG 4:4:4, mozjpeg baseline** (DCT shrink) | **50.3** | **1,088** | **45,700** | **21** |
| JPEG 4:4:4, libjpeg-turbo baseline (DCT shrink) | 53.1 | 1,071 | 45,200 | 2.7 |
| JPEG 4:4:4, mozjpeg progressive (DCT shrink) | 48.8 | 557 | 24,900 | 33 |
| WebP (method 6, sharp YUV) | 50.2 | 298 | 14,800 | 77 |
| JPEG XL (effort 7, decoding speed 4) | 54.3 | 194 | 8,800 | 70 |
| JPEG 4:2:0, libjpeg-turbo baseline (DCT shrink) | 66.3 | 1,116 | 49,900 | 2.2 |

AVIF 4:4:4 is 27-35 % smaller than JPEG at the same quality and decodes 4x slower; WebP and JPEG
XL are larger than AVIF and slower than JPEG; progressive JPEG decodes at half the speed of
baseline. The writer's default, mozjpeg 4:4:4 baseline quality 70, lands at 41.3 dB on these
crops at 43 KB per image, through this writer from the originals (588 of 600 images: the
transparent PNGs, composited onto white here, have no comparable reference).

### Reading speed

mozjpeg 4:4:4 q70 shard (3000 images), 256 x 256 rgb output, batches of 256, 64 CPUs (32 cores
x 2 threads) of a Xeon 6747P, the shard in the page cache:

| crop | ms / image, 1 thread | 16 threads | 32 threads | 64 threads |
|---|---|---|---|---|
| random (short side to 256): DCT shrink 1/2 | 0.89 | 12,800 / s | 26,000 / s | 42,700 / s |
| same, no DCT shrinking (`dct_scale=False`) | 1.38 | 8,500 / s | 18,700 / s | 29,600 / s |
| resized (area 0.08-1) | 1.22 | 8,800 / s | 19,000 / s | 30,200 / s |

Per image on one thread (random crop): decode 0.74 ms, resize 0.08, convert 0.06, read 0.01.
Baseline on the same bytes (from memory, no file reads): Pillow (`draft`, resize, crop) 1.22 ms
/ image / core, 32,600 / s on 64 processes. As loaders (decode threads vs DataLoader workers
over a zip + Pillow dataset): `ImageDataset` 26,700 / s with 32 threads, the DataLoader 14,800 /
s with 32 workers.

DCT shrinking reads the JPEG at half resolution when the output allows it: the outputs differ
from full decoding + the resize at ~40 dB PSNR (a sharper low-pass); `dct_scale=False` gives the
resize's filter alone. 4:4:4 JPEGs read at their stored size are libjpeg's decode + the
conversion, bit for bit (tests).

## Install and build

```bash
pip install kohakuclip
```

The wheels (manylinux_2_35: glibc 2.35+, e.g. Ubuntu 22.04 or newer; x86-64-v3; Python 3.13
/ 3.14) bundle an LGPL FFmpeg 8.1 from conda-forge with libdav1d (AV1 decoding), SVT-AV1 and
aom (AV1 encoding) and openh264, libjpeg-turbo (images) and mozjpeg (built in); they need
nothing else installed. That build has no x264 / x265, so `--codec h264` / `hevc` needs a source
build against an FFmpeg that has them, or `--backend ffmpeg`.

To build against your own FFmpeg (libavcodec, libavformat, libswscale, libavutil; with libdav1d for
AV1 decoding and libsvtav1 for AV1 encoding), libjpeg-turbo >= 3.0 (TurboJPEG API), libdav1d, libjxl
and libwebp (in the same library directory), e.g. conda-forge / micromamba (`micromamba install
ffmpeg "libjpeg-turbo=3" dav1d-devel "libclang=18" "clang=18"`) or the distro's, and `nasm` on PATH
(mozjpeg is compiled in by `mozjpeg-sys`, without its SIMD code when nasm is missing). The Rust
bindings are generated at build time by bindgen, with libclang 18 (newer versions generate broken
FFmpeg bindings with this bindgen) and clang's own headers:

```bash
export FFMPEG_INCLUDE_DIR=$CONDA_PREFIX/include FFMPEG_LIBS_DIR=$CONDA_PREFIX/lib
export FFMPEG_LINK_MODE=dynamic LIBCLANG_PATH=$CONDA_PREFIX/lib
export BINDGEN_EXTRA_CLANG_ARGS="-I$CONDA_PREFIX/lib/clang/18/include"
export RUSTFLAGS="-C target-cpu=native -C link-arg=-Wl,-rpath,$CONDA_PREFIX/lib"
pip install .  # or: maturin develop --release
```

Linux, x86-64 (other targets build without the SIMD kernels). The resize picks SSE2 / AVX2 /
AVX-512 VNNI kernels at runtime (`KOHAKUCLIP_SIMD=plain|avx2` caps it).

Releases: `.github/workflows/release.yml` is the only workflow that builds or publishes
(wheels, sdist, PyPI through trusted publishing, GitHub release with notes from `Change.md`);
`nightly.yml` dispatches it daily when `main` moved, `auto-patch-release.yml` cuts a patch
weekly. The version lives in `pyproject.toml` (`scripts/ci/version.py`).

## Code

- `src/kclip_rs/` (Rust): `storage/` (zip / tar / folder listing and writing, the video and image
  indexes, shards), `mp4/` (moov parsing, decoder configuration, faststart), `codec/` (JPEG
  through libjpeg-turbo and mozjpeg, AV1 through libdav1d, JPEG XL through libjxl, WebP through
  libwebp, PNG through FFmpeg), `read/` (planning clips and images, sampling, epoch
  permutations, the thread pool), `decode/` (decoding a planned clip or image, output pixels per
  mode), `image/` (pictures, resize, conversion, SIMD kernels), `write/` (encoding videos with
  FFmpeg, AV1 dependency parsing, encoding images, packing), `python/` (the `kohakuclip._core`
  module).
- `src/kohakuclip/` (Python): the package, `torch.py` (`ClipDataset`, `ImageDataset`,
  `ClipLoader`, `yuv_to_rgb`), `gpu.py` (the Triton kernel), `writer.py` (video sources, the
  ffmpeg command-line backend, the CLI), `image_writer.py` (image sources, the Pillow backend,
  manifests, the CLI).
- `examples/`: a plain PyTorch loop and a Lightning module. `benchmarks/`: reader speed
  (`bench.py`, images: `image_bench.py` with Pillow / torchvision / DataLoader baselines), the
  PyTorch side (`loader_bench.py`), the GPU side (`gpu_check.py`), memory (`leak_check.py`).

Tests (`pytest tests`) write AV1 and H.264 shards from synthetic videos (native and command-line
encoders) and compare the reader with a PyAV + torch reference; zip / tar / folder sources with
the index and the moov fallback with each other; skipping with full decoding; tar and in-memory
inputs with files; the yuv modes with rgb; `ClipDataset` batches across worker counts and after
resuming; encoder errors and failed batches. Images: shards written from synthetic images
(folders with captions, WebDataset tars, EXIF-rotated, transparent, gray and CMYK inputs) against
a Pillow + torch reference; mozjpeg output (baseline, 4:4:4, smaller than libjpeg-turbo's)
decoded bit for bit against libjpeg's own decode; indexed zip / listed zip / tar / folder sources
with each other, and plain tars, zips (stored and deflated) and folders of JPEGs; odd-sized 4:2:0
chroma alignment (reader and writer); DCT shrinking, augmentations, the yuv modes, AV1 / JPEG XL /
WebP storage, the Pillow backend, `ImageDataset` across worker counts, resumption and epochs.
`cargo test` checks the SIMD kernels against the plain code, the Lanczos filter, the EXIF and
orientation handling, the permutation and the crop geometry.
Rust is formatted with rustfmt and checked with clippy; Python is formatted with black and
checked with ruff and mypy.
