# KohakuClip

Fast random-access video clips for training. Videos are stored as faststart mp4 (AV1 or H.264)
in zip shards with an in-zip frame index; a batch of clips is read, decoded, resized and cropped
natively on a persistent thread pool, straight into a (pinned) uint8 buffer.

```python
from kohakuclip import Reader, Augment, clip, random_frames

reader = Reader(["data/shard_00000.zip", ...], size=256, threads=16,
                augment=Augment(crop="random", hflip=0.5))
v = reader.info(vid)                                   # n frames, fps, stored h/w, codec
frames = clip(v.n, v.fps, frames=8, target_fps=6, rng=rng)   # or random_frames(v.n, 8, rng)
video = reader.read([(vid, frames), ...]).video       # uint8 [B, T, 3, 256, 256]
```

With PyTorch, `kohakuclip.torch.Loader` keeps batches queued on the native pool from one
background thread (the native calls release the GIL) and returns pinned tensors:

```python
from kohakuclip.torch import Loader

def sample(rng):
    vid = rng.randrange(len(reader)); v = reader.info(vid)
    return vid, clip(v.n, v.fps, 8, 6.0, rng)

for video in Loader(reader, sample, batch_size=16):   # uint8 [16, 8, 3, 256, 256], pinned
    video = video.cuda(non_blocking=True).float() / 127.5 - 1
```

## Writing shards

```bash
kohakuclip-write out_dir videos/*.mp4 --codec av1 --crf 36 --per-shard 1000
```

Defaults (measured below): native fps, short side capped at 512 (never upscaled), closed GOP of
16, AV1 via SVT-AV1 preset 6 with the in-loop filters (deblocking, CDEF, loop restoration) off,
faststart mp4, zip in stored mode. Needs an `ffmpeg` with libsvtav1 (or libx264 for
`--codec h264`) and PyAV. The writer also parses the AV1 frame headers and records, per sample,
how many bytes later frames depend on: SVT-AV1 puts half of all frames in a top layer nothing
references, so unwanted samples are decoded only up to their last referenced frame (25 % of
samples are dropped entirely, 25 % truncated), bit-exactly.

Sources: a `Reader` takes any mix of zip archives (stored), uncompressed tar archives and folder
trees (every `*.mp4` below them, recursively). Archives written by `kohakuclip-write`
(`--container zip|tar`) carry an index member; archives without one, and folders, fall back to
parsing each mp4's `moov` box on first use (all return the same clips; see `tests`).

Shard layout: `shard_XXXXX.zip` (or `.tar`) = the mp4 files + `__index__.bin` (per frame: absolute byte
offset, size, bytes later frames depend on, keyframe flag; per video: fps, size, codec, decoder
prefix). The index
is memory-mapped from the zip (shared by all workers through the page cache). Without it, each
video's frame table is parsed from its own `moov` box on first use (faststart puts it first).

## How a clip is read

1. Python plans each clip from the index: wanted frames -> one byte range per GOP (keyframe to
   last wanted frame), unwanted samples cut to the bytes later frames need, crop and flips.
2. One native call per batch (`Reader.submit` / `Pending.result`): per GOP one `pread`, decode
   with a persistent per-thread decoder (libdav1d for AV1, libavcodec for H.264 / HEVC), then per
   wanted frame: antialiased bilinear resize of each YUV plane fused with the crop (only the source
   window the crop needs is read), conversion of the output pixels to RGB, flips.
3. Batches queued back to back share one FIFO on the pool: no per-batch barrier, and planning
   overlaps decoding.

`Reader(mode="yuv")` returns the stored-resolution crop window as YUV 4:2:0 instead, and
`kohakuclip.torch.yuv_to_rgb` converts and resizes it on the GPU.

## Measurements

VidGen-1M sample (981 videos, 720p H.264 sources, mean 10.5 s, ~28 fps) stored at 910x512,
native fps, AV1 CRF 36, GOP 16, filters off: 4.3 TB per 50M seconds at 768x512 (38.5 dB stored,
39.9 dB on the 256 crop). Cold reads on network storage, B300 host CPUs. ms per output frame per
core (256 x 256 crop):

| clip | rgb, 1 core | rgb, 24 cores | yuv, 1 core | yuv, 24 cores |
|---|---|---|---|---|
| 8 frames @ 6 fps | 3.7 | 4.2 (712 clips/s) | 3.2 | 3.5 (865 clips/s) |
| 8 frames @ 24 fps | 2.2 | 2.5 (1197 clips/s) | 1.7 | 1.9 (1550 clips/s) |
| 16 frames @ 6 fps | 3.6 | 3.9 (385 clips/s) | 2.8 | 3.2 (466 clips/s) |
| 8 random frames (whole video) | 6.1 | 6.6 (455 clips/s) | 5.2 | 5.8 (519 clips/s) |

Per frame at 8 @ 6 fps, one core (rgb): decode 3.0 ms (2.9 frames decoded per output frame),
resize 0.35, convert 0.2, read 0.2. Clips from 60-300 s videos cost the same as from 3-30 s ones
(one read per GOP). 24 threads in one process scale like 24 processes.

| setting | effect (ms per output frame, one core) |
|---|---|
| AV1 in-loop filters off (writer) | -20 to -27 % decode, same size, same crop quality |
| decode only what later frames need of unwanted samples (`skip=True`) | 8f@6 4.29 -> 3.27, 8 random 6.62 -> 5.12 (whole-unit skipping alone: 3.65 / 5.72); bit-exact |
| resize YUV planes, then convert (vs convert, then resize) | 4.97 -> 4.22; the two agree at 48.4 dB (differ only at saturated edges) |
| `mode="yuv"` (GPU converts + resizes) | CPU 3.94 -> 3.24 (8f@6), 2.27 -> 1.59 (8f@24); GPU +21 us/frame; 2x the bytes at 512p storage; 48.7 dB vs rgb |
| threads, 2 batches in flight vs synchronous | 16 cores 5.04 -> 4.72, 24 cores 5.99 -> 4.53 (= separate processes) |
| index vs moov fallback | within ~3 % |

## Build

```bash
pip install -e .                          # FFmpeg found with pkg-config
KOHAKUCLIP_FFMPEG=/opt/ffmpeg pip install -e .   # or a prefix with include/ and lib/
```

Python >= 3.13, Linux. The core (`src/kohakuclip/native/kc.cpp`, C++17, FFmpeg's libavcodec
with libdav1d) is loaded with ctypes; `KOHAKUCLIP_MARCH` sets `-march` (default `native`).

Tests (`pytest tests`) write AV1 and H.264 shards from synthetic videos and compare the reader
with a PyAV + torch reference, the index with the moov fallback, and skipping with full decoding.
