# Change Log

## (unreleased) update to 1.1.0

### Highlights

Images: JPEG shards written with mozjpeg (4:4:4, baseline, quality 70 by default: 41-42 dB PSNR
on 256 x 256 random resized crops at ~43 KB per image, chosen from a measured study of
optimized JPEG / AVIF / WebP / JPEG XL encoders) with a fixed-size memory-mapped index, read
natively without the GIL at ~0.9 ms per 256 x 256 image per core (42.7k images/s on 64 CPUs of
a Xeon 6747P), plus plain tars, zips and folders of images, and an image writer for any input
format.

### Full change log

#### New Features

* **Images**: `ImageReader` (rgb / yuv / yuv_resized, short-side or random resized crops, flips,
  DCT shrinking, kernel readahead; indexed shards, or plain zips (stored or deflated members),
  tars and folders), `ImageDataset` (the `ClipDataset` of images; epochs over a seeded
  permutation, `permute`), `encode_image` / `pack_images` and `kohakuclip-write-images` (files,
  folders with caption sidecars, WebDataset tars; EXIF orientation, alpha onto white, Lanczos-3
  downscaling, Pillow fallback; manifest.parquet with `native_res`). JPEGs are encoded by
  mozjpeg (built in; `jpeg_encoder="turbo"` for libjpeg-turbo). AV1 (intra), JPEG XL and WebP
  storage selectable, each read through its own library directly (libdav1d, libjxl, libwebp;
  single-threaded, native planes into the same resize and conversion as JPEG); PNG through
  FFmpeg.
* `yuv_to_rgb` (Triton and torch) handles full-range colorspaces ("bt601-full") and image
  batches.

#### Fixes

* Encoding a source tagged with the RGB (identity) matrix on YUV content crashed (rsmpeg frees
  the encoder options twice when an encoder fails to open): encoder open errors now raise, and
  such sources are encoded as untagged.
* Dropping a `Pending` whose `result()` raised no longer panics.

## 2026/10/01 update to 1.0.0

### Highlights

A full redo of KohakuClip: mp4 files in zip / tar shards (or plain folders and archives of
mp4), read by a Rust core on the system's FFmpeg libraries. Per batch, planning and decoding
run natively without the GIL, one pread per GOP, an antialiased resize fused with the crop
and SIMD kernels; 1.8-3.2 ms per output frame per core on a Xeon 6747P, 1500 clips/s on 24
cores (8 frames at 24 fps, 256 x 256).

### Full change log

#### New Features

* **Storage**: faststart mp4 (AV1 by default, H.264 / HEVC) in stored zip or tar shards with
  an in-archive, memory-mapped frame index; folders and archives without an index are read
  through each mp4's moov box, with the same results.
* **Reader**: rgb, yuv and yuv_resized output modes; random or center crops and flips;
  per-call seeds; AV1 samples nothing depends on are skipped or truncated, bit-exactly.
* **Writer**: any input FFmpeg reads (files, folders, tar archives of videos, bytes in
  memory) re-encoded by the linked FFmpeg into memory, faststart and indexed natively;
  `--backend ffmpeg` uses the command line instead.
* **PyTorch**: `ClipDataset` (an `IterableDataset`; the same batches for any number of
  DataLoader workers, rank-aware, epoch mode) and `ClipLoader` (exact resumption through
  `state_dict()`, Lightning checkpoints included); `yuv_to_rgb` as one Triton kernel.

#### Improvements

* Python 3.13+, type stubs for the native module; Rust checked with clippy, Python with ruff
  and mypy, both formatted (rustfmt, black).
* CI, nightly builds and release automation under `.github/workflows/`; manylinux_2_35
  wheels bundle an LGPL FFmpeg (libdav1d, SVT-AV1, aom, openh264).
