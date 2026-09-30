"""End-to-end: write shards from synthetic videos, read clips back, compare with a PyAV + torch reference.

Needs an ffmpeg with libsvtav1 and libx264 ($KOHAKUCLIP_TEST_FFMPEG, default "ffmpeg") and PyAV.
"""

import os
import random
import subprocess

import numpy as np
import pytest

av = pytest.importorskip("av")
torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from kohakuclip import Augment, Reader, Shard, clip, random_frames  # noqa: E402
from kohakuclip.writer import Encoding, write  # noqa: E402

FFMPEG = os.environ.get("KOHAKUCLIP_TEST_FFMPEG", "ffmpeg")
SIZE = 128


@pytest.fixture(scope="module", params=["av1", "h264"])
def shards(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp(request.param)
    sources = []
    for i, (w, h, sec) in enumerate([(640, 360, 3), (360, 640, 2), (1280, 720, 4)]):
        path = str(tmp / f"src{i}.mp4")
        subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=30:duration={sec}",
                        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "12", path], check=True)
        sources.append(path)
    enc = Encoding(codec=request.param, crf=30 if request.param == "av1" else 20, ffmpeg=FFMPEG)
    return write(sources, str(tmp / "out"), enc, per_shard=2, workers=3)


def reference(shard: Shard, i: int, frames: list[int], top: int, left: int) -> np.ndarray:
    """PyAV decode (YUV planes) -> RGB with the core's conventions (centered bilinear chroma, BT.709
    limited range) -> torch antialiased resize of the short side -> crop."""
    import zipfile

    name = shard.members[i][0]
    with zipfile.ZipFile(shard.path) as z, z.open(name) as f, av.open(f) as c:
        yuv = [fr.to_ndarray(format="yuv420p") for fr in c.decode(video=0)]
    x = torch.from_numpy(np.stack([yuv[k] for k in frames])).float()
    h, w = x.shape[1] * 2 // 3, x.shape[2]
    y = x[:, :h]
    u = x[:, h:h + h // 4].reshape(-1, 1, h // 2, w // 2)
    v = x[:, h + h // 4:].reshape(-1, 1, h // 2, w // 2)
    u, v = (F.interpolate(c, size=(h, w), mode="bilinear", align_corners=False)[:, 0] - 128 for c in (u, v))
    yy, s = (y - 16) * 255 / 219, 255 / 224
    rgb = torch.stack([yy + 1.5748 * s * v, yy - 0.187324 * s * u - 0.468124 * s * v, yy + 1.8556 * s * u], 1)
    rgb = rgb.round().clamp(0, 255)
    sc = SIZE / min(h, w)
    rgb = F.interpolate(rgb, size=(max(SIZE, round(h * sc)), max(SIZE, round(w * sc))), mode="bilinear", antialias=True)
    return rgb[..., top:top + SIZE, left:left + SIZE].round().clamp(0, 255).byte().numpy()


def test_index_matches_moov(shards):
    for path in shards:
        indexed, parsed = Shard(path), Shard(path)
        parsed.meta = None  # force the moov fallback
        for i in range(len(indexed)):
            a, b = indexed.video(i), parsed.video(i)
            assert (a.n, a.h, a.w, a.codec) == (b.n, b.h, b.w, b.codec)
            assert np.array_equal(a.off, b.off) and np.array_equal(a.size, b.size) and np.array_equal(a.keys, b.keys)
            assert abs(a.fps - b.fps) < 1e-3 and a.prefix == b.prefix


@pytest.mark.parametrize("pick", ["clip6", "clip24", "random", "repeat"])
def test_frames_match_reference(shards, pick):
    reader = Reader(shards, size=SIZE, threads=4, augment=Augment(crop="center"))
    rng = random.Random(0)
    items = []
    for vid in range(len(reader)):
        v = reader.info(vid)
        frames = {"clip6": lambda: clip(v.n, v.fps, 8, 6.0, rng), "clip24": lambda: clip(v.n, v.fps, 8, 24.0, rng),
                  "random": lambda: random_frames(v.n, 8, rng), "repeat": lambda: [3, 3, 40, 40, 5, 5, 0, 0]}[pick]()
        items.append((vid, frames))
    out = reader.read(items).video
    for (vid, frames), got in zip(items, out):
        shard, i = reader.videos[vid]
        v = reader.info(vid)
        s = SIZE / min(v.h, v.w)
        nh, nw = max(SIZE, round(v.h * s)), max(SIZE, round(v.w * s))
        ref = reference(shard, i, frames, (nh - SIZE) // 2, (nw - SIZE) // 2)
        diff = np.abs(got.astype(int) - ref.astype(int))
        assert diff.mean() < 0.6 and np.percentile(diff, 99.9) <= 3, (pick, vid, diff.mean(), diff.max())


def test_moov_fallback_is_bit_identical(shards):
    items = [(0, [1, 5, 9, 30]), (1, [2, 3, 4, 50])]
    a = Reader(shards, size=SIZE, augment=Augment(crop="center")).read(items).video
    fallback = [Shard(p) for p in shards]
    for s in fallback:
        s.meta = None
    b = Reader(fallback, size=SIZE, augment=Augment(crop="center")).read(items).video
    assert np.array_equal(a, b)
