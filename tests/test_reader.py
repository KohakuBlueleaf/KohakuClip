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
        subprocess.run(
            [
                FFMPEG,
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=size={w}x{h}:rate=30:duration={sec}",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-crf",
                "12",
                path,
            ],
            check=True,
        )
        sources.append(path)
    enc = Encoding(
        codec=request.param, crf=30 if request.param == "av1" else 20, ffmpeg=FFMPEG
    )
    return write(sources, str(tmp / "out"), enc, per_shard=2, workers=3)


def reference(
    shard: Shard, i: int, frames: list[int], top: int, left: int
) -> np.ndarray:
    """PyAV decode (YUV planes) -> torch antialiased resize of each plane onto the output grid (short
    side -> SIZE) -> crop -> BT.709 limited-range conversion: the core's order, in float.
    """
    import zipfile

    name = shard.members[i].name
    with zipfile.ZipFile(shard.path) as z, z.open(name) as f, av.open(f) as c:
        yuv = [fr.to_ndarray(format="yuv420p") for fr in c.decode(video=0)]
    x = torch.from_numpy(np.stack([yuv[k] for k in frames])).float()
    h, w = x.shape[1] * 2 // 3, x.shape[2]
    sc = SIZE / min(h, w)
    size = (max(SIZE, round(h * sc)), max(SIZE, round(w * sc)))
    planes = [
        x[:, None, :h],
        x[:, h : h + h // 4].reshape(-1, 1, h // 2, w // 2),
        x[:, h + h // 4 :].reshape(-1, 1, h // 2, w // 2),
    ]
    y, u, v = (
        F.interpolate(p, size=size, mode="bilinear", antialias=True)[
            :, 0, top : top + SIZE, left : left + SIZE
        ].round()
        for p in planes
    )
    yy, s, u, v = (y - 16) * 255 / 219, 255 / 224, u - 128, v - 128
    rgb = torch.stack(
        [
            yy + 1.5748 * s * v,
            yy - 0.187324 * s * u - 0.468124 * s * v,
            yy + 1.8556 * s * u,
        ],
        1,
    )
    return rgb.round().clamp(0, 255).byte().numpy()


def test_index_matches_moov(shards):
    for path in shards:
        indexed, parsed = Shard(path), Shard(path)
        parsed.meta = None  # force the moov fallback
        for i in range(len(indexed)):
            a, b = indexed.video(i), parsed.video(i)
            assert (a.n, a.h, a.w, a.codec) == (b.n, b.h, b.w, b.codec)
            assert (
                np.array_equal(a.off, b.off)
                and np.array_equal(a.size, b.size)
                and np.array_equal(a.keys, b.keys)
            )
            assert abs(a.fps - b.fps) < 1e-3 and a.prefix == b.prefix


@pytest.mark.parametrize("pick", ["clip6", "clip24", "random", "repeat"])
def test_frames_match_reference(shards, pick):
    reader = Reader(shards, size=SIZE, threads=4, augment=Augment(crop="center"))
    rng = random.Random(0)
    items = []
    for vid in range(len(reader)):
        v = reader.info(vid)
        frames = {
            "clip6": lambda: clip(v.n, v.fps, 8, 6.0, rng),
            "clip24": lambda: clip(v.n, v.fps, 8, 24.0, rng),
            "random": lambda: random_frames(v.n, 8, rng),
            "repeat": lambda: [3, 3, 40, 40, 5, 5, 0, 0],
        }[pick]()
        items.append((vid, frames))
    out = reader.read(items).video
    for (vid, frames), got in zip(items, out):
        shard, i = reader.videos[vid]
        v = reader.info(vid)
        s = SIZE / min(v.h, v.w)
        nh, nw = max(SIZE, round(v.h * s)), max(SIZE, round(v.w * s))
        ref = reference(shard, i, frames, (nh - SIZE) // 2, (nw - SIZE) // 2)
        diff = np.abs(got.astype(int) - ref.astype(int))
        assert diff.mean() < 0.6 and np.percentile(diff, 99.9) <= 3, (
            pick,
            vid,
            diff.mean(),
            diff.max(),
        )


def test_moov_fallback_is_bit_identical(shards):
    items = [(0, [1, 5, 9, 30]), (1, [2, 3, 4, 50])]
    a = Reader(shards, size=SIZE, augment=Augment(crop="center")).read(items).video
    fallback = [Shard(p) for p in shards]
    for s in fallback:
        s.meta = None
    b = Reader(fallback, size=SIZE, augment=Augment(crop="center")).read(items).video
    assert np.array_equal(a, b)


def test_skipping_unreferenced_samples_is_exact(shards):
    reader = Reader(shards, size=SIZE, threads=4, augment=Augment(crop="center"))
    rng = random.Random(1)
    items = [
        (vid, clip(reader.info(vid).n, reader.info(vid).fps, 8, 6.0, rng))
        for vid in range(len(reader))
    ]
    items += [
        (vid, random_frames(reader.info(vid).n, 8, rng)) for vid in range(len(reader))
    ]
    fast = reader.read(items).video
    reader.skip = False
    assert np.array_equal(fast, reader.read(items).video)


def test_sources_agree(shards, tmp_path):
    """zip / tar archives (with the index and with the moov fallback) and a nested folder tree of
    the same mp4 files return the same clips."""
    import tarfile
    import zipfile

    from kohakuclip.writer import pack

    mp4s = []
    for k, path in enumerate(shards):
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.endswith(".mp4"):
                    dst = tmp_path / "tree" / f"part{k}" / "nested" / name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(z.read(name))
                    mp4s.append(str(dst))
    tar = str(tmp_path / "all.tar")
    pack(mp4s, tar)
    assert tarfile.is_tarfile(tar)

    def fallback(path):
        s = Shard(path)
        s.meta = None
        return s

    sources = {
        "zip": list(shards),
        "zip-moov": [fallback(p) for p in shards],
        "tar": [tar],
        "tar-moov": [fallback(tar)],
        "folder": [str(tmp_path / "tree")],
    }
    outs = {}
    for label, src in sources.items():
        reader = Reader(src, size=SIZE, augment=Augment(crop="center"))
        names = [os.path.basename(s.members[i].name) for s, i in reader.videos]
        by_name = {n: vid for vid, n in enumerate(names)}
        items = [(by_name[n], [0, 7, 21, 40]) for n in sorted(by_name)]
        outs[label] = reader.read(items).video
    for label, out in outs.items():
        assert np.array_equal(out, outs["zip"]), label
