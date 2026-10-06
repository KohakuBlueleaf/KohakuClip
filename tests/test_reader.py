"""End-to-end: write shards from synthetic videos, read clips back, compare with a PyAV + torch
reference.

Needs an FFmpeg with libsvtav1 and libx264 (the linked libraries, and the ``ffmpeg`` command for
the command-line backend: $KOHAKUCLIP_TEST_FFMPEG, default "ffmpeg") and PyAV.
"""

import io
import os
import random
import subprocess
import tarfile
import zipfile

import numpy as np
import pytest

av = pytest.importorskip("av")
torch = pytest.importorskip("torch")
import torch.nn.functional as F

from kohakuclip import Reader, _core, clip, random_frames
from kohakuclip.writer import Encoding, write

FFMPEG = os.environ.get("KOHAKUCLIP_TEST_FFMPEG", "ffmpeg")
SIZE = 128
SOURCES = [(640, 360, 3), (360, 640, 2), (1280, 720, 4)]  # width, height, seconds


def synthetic(path: str, w: int, h: int, seconds: int) -> None:
    cmd = [
        FFMPEG,
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={w}x{h}:rate=30:duration={seconds}",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "12",
        path,
    ]
    subprocess.run(cmd, check=True)


@pytest.fixture(scope="module")
def sources(tmp_path_factory) -> list[str]:
    tmp = tmp_path_factory.mktemp("sources")
    paths = []
    for i, (w, h, seconds) in enumerate(SOURCES):
        path = str(tmp / f"src{i}.mp4")
        synthetic(path, w, h, seconds)
        paths.append(path)
    return paths


@pytest.fixture(scope="module", params=["av1", "h264", "av1-ffmpeg"])
def shards(request, sources, tmp_path_factory) -> list[str]:
    codec, _, backend = request.param.partition("-")
    tmp = tmp_path_factory.mktemp(request.param)
    enc = Encoding(codec=codec, crf=30 if codec == "av1" else 20)
    return write(
        sources,
        str(tmp / "out"),
        enc,
        per_shard=2,
        workers=3,
        backend=backend or "native",
        ffmpeg=FFMPEG,
    )


def member_bytes(source: str, name: str) -> bytes:
    if tarfile.is_tarfile(source):
        with tarfile.open(source) as t:
            return t.extractfile(name).read()
    with zipfile.ZipFile(source) as z:
        return z.read(name)


# (R from V, G from U, G from V, B from U), limited range
MATRIX = {
    "bt709": (1.5748, -0.187324, -0.468124, 1.8556),
    "bt601": (1.402, -0.344136, -0.714136, 1.772),
}


def reference(reader: Reader, vid: int, frames: list[int]) -> np.ndarray:
    """PyAV decode (YUV planes) -> torch antialiased resize of each plane onto the output grid
    (short side -> SIZE) -> center crop -> limited-range conversion with the stored colorspace:
    the core's order, in float."""
    v = reader.info(vid)
    data = member_bytes(str(v.source), v.name)
    with av.open(io.BytesIO(data)) as c:
        yuv = [f.to_ndarray(format="yuv420p") for f in c.decode(video=0)]
    x = torch.from_numpy(np.stack([yuv[k] for k in frames])).float()

    h, w = x.shape[1] * 2 // 3, x.shape[2]
    scale = SIZE / min(h, w)
    size = (max(SIZE, round(h * scale)), max(SIZE, round(w * scale)))
    top, left = (size[0] - SIZE) // 2, (size[1] - SIZE) // 2
    planes = [
        x[:, None, :h],
        x[:, h : h + h // 4].reshape(-1, 1, h // 2, w // 2),
        x[:, h + h // 4 :].reshape(-1, 1, h // 2, w // 2),
    ]

    def resize(p):
        out = F.interpolate(p, size=size, mode="bilinear", antialias=True)
        return out[:, 0, top : top + SIZE, left : left + SIZE].round()

    y, u, v = (resize(p) for p in planes)
    r_v, g_u, g_v, b_u = MATRIX[reader.info(vid).colorspace]
    yy, s, u, v = (y - 16) * 255 / 219, 255 / 224, u - 128, v - 128
    rgb = torch.stack(
        [
            yy + r_v * s * v,
            yy + g_u * s * u + g_v * s * v,
            yy + b_u * s * u,
        ],
        1,
    )
    return rgb.round().clamp(0, 255).byte().numpy()


def picks(reader: Reader, pick: str, rng: random.Random) -> list[tuple[int, list[int]]]:
    items = []
    for vid in range(len(reader)):
        v = reader.info(vid)
        if pick == "clip6":
            frames = clip(v.n, v.fps, 8, 6.0, rng)
        elif pick == "clip24":
            frames = clip(v.n, v.fps, 8, 24.0, rng)
        elif pick == "random":
            frames = random_frames(v.n, 8, rng)
        else:
            frames = [3, 3, 40, 40, 5, 5, 0, 0]
        items.append((vid, frames))
    return items


def test_storage_format(shards):
    reader = Reader(shards)
    for vid in range(len(reader)):
        v = reader.info(vid)
        assert min(v.h, v.w) in (360, 512) and abs(v.fps - 30) < 1e-3
        data = member_bytes(str(v.source), v.name)
        assert data.index(b"moov") < data.index(b"mdat")  # faststart


@pytest.mark.parametrize("pick", ["clip6", "clip24", "random", "repeat"])
def test_frames_match_reference(shards, pick):
    reader = Reader(shards, size=SIZE, threads=4, crop="center")
    items = picks(reader, pick, random.Random(0))
    out = reader.read(items).video
    for (vid, frames), got in zip(items, out):
        diff = np.abs(got.astype(int) - reference(reader, vid, frames).astype(int))
        assert diff.mean() < 0.6 and np.percentile(diff, 99.9) <= 3, (
            pick,
            vid,
            diff.mean(),
            diff.max(),
        )


def test_moov_fallback_is_bit_identical(shards):
    indexed = Reader(shards, size=SIZE, crop="center")
    parsed = Reader(shards, size=SIZE, crop="center", use_index=False)
    for vid in range(len(indexed)):
        a, b = indexed.info(vid), parsed.info(vid)
        assert (a.n, a.h, a.w, a.codec, a.name) == (b.n, b.h, b.w, b.codec, b.name)
        assert abs(a.fps - b.fps) < 1e-6

    items = [(0, [1, 5, 9, 30]), (1, [2, 3, 4, 50])]
    assert np.array_equal(indexed.read(items).video, parsed.read(items).video)


def test_skipping_unreferenced_samples_is_exact(shards):
    fast = Reader(shards, size=SIZE, threads=4, crop="center")
    full = Reader(shards, size=SIZE, threads=4, crop="center", skip=False)
    rng = random.Random(1)
    items = picks(fast, "clip6", rng) + picks(fast, "random", rng)
    assert np.array_equal(fast.read(items).video, full.read(items).video)


def test_sources_agree(shards, tmp_path):
    """zip / tar archives (with the index and with the moov fallback) and a nested folder tree of
    the same mp4 files return the same clips."""
    members = []
    for k, path in enumerate(shards):
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.endswith(".mp4"):
                    members.append((name, z.read(name)))
                    dst = tmp_path / "tree" / f"part{k}" / "nested" / name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(z.read(name))
    tar = str(tmp_path / "all.tar")
    _core.pack(tar, members)
    assert tarfile.is_tarfile(tar)

    sources = {
        "zip": {"sources": list(shards)},
        "zip-moov": {"sources": list(shards), "use_index": False},
        "tar": {"sources": [tar]},
        "tar-moov": {"sources": [tar], "use_index": False},
        "folder": {"sources": [str(tmp_path / "tree")]},
    }
    outs = {}
    for label, kwargs in sources.items():
        reader = Reader(size=SIZE, crop="center", **kwargs)
        by_name = {os.path.basename(reader.info(v).name): v for v in range(len(reader))}
        items = [(by_name[n], [0, 7, 21, 40]) for n in sorted(by_name)]
        outs[label] = reader.read(items).video
    for label, out in outs.items():
        assert np.array_equal(out, outs["zip"]), label


def test_writer_inputs(sources, tmp_path):
    """A tar of videos and in-memory bytes encode like the files themselves."""
    tar = tmp_path / "videos.tar"
    with tarfile.open(tar, "w") as t:
        for path in sources:
            t.add(path, os.path.basename(path))

    enc = Encoding(crf=30)
    from_files = Reader(
        write(sources, str(tmp_path / "a"), enc, workers=3), crop="center"
    )
    from_tar = Reader(
        write([str(tar)], str(tmp_path / "b"), enc, workers=3), crop="center"
    )
    assert len(from_files) == len(from_tar) == len(SOURCES)

    items = [(v, [0, 10, 20]) for v in range(len(from_files))]
    a = from_files.read(items).video
    b = from_tar.read(items).video
    assert np.array_equal(a, b)

    with open(sources[0], "rb") as f:
        data = f.read()
    assert _core.encode(data, crf=30) == _core.encode(sources[0], crf=30)


def test_yuv_matches_rgb(shards):
    """GPU-side conversion (run on the CPU here) of the stored-resolution yuv window vs rgb."""
    from kohakuclip.torch import yuv_to_rgb

    rgb = Reader(shards, size=SIZE, crop="center")
    yuv = Reader(shards, size=SIZE, mode="yuv", crop="center")
    same = [v for v in range(len(rgb)) if min(rgb.info(v).h, rgb.info(v).w) == 360]
    # "yuv" needs a common stored short side per batch
    items = [(v, [0, 9, 30]) for v in same]
    ref = torch.from_numpy(rgb.read(items).video).float()
    got = (yuv_to_rgb(yuv.read(items), SIZE, device="cpu") + 1) * 127.5
    psnr = 10 * np.log10(255**2 / ((got - ref) ** 2).mean().item())
    assert psnr > 38, psnr


def test_yuv_resized_planes(shards):
    """yuv_resized planes == the yuv window's planes resized in torch (Y to the crop, U / V to
    half)."""
    yuv = Reader(shards, size=SIZE, mode="yuv", crop="center")
    small = Reader(shards, size=SIZE, mode="yuv_resized", crop="center")
    same = [v for v in range(len(yuv)) if min(yuv.info(v).h, yuv.info(v).w) == 360]
    items = [(v, [0, 9, 30]) for v in same]
    a, b = yuv.read(items), small.read(items)

    side, n = int(a.window[0][0]), len(items) * 3
    x = torch.from_numpy(a.video).reshape(n, -1).float()
    y = torch.from_numpy(b.video).reshape(n, -1).float()
    for lo, hi, full, out in (
        (0, side**2, side, SIZE),
        (side**2, side**2 * 5 // 4, side // 2, SIZE // 2),
    ):
        plane = x[:, lo:hi].reshape(n, 1, full, full)
        ref = F.interpolate(plane, size=(out, out), mode="bilinear", antialias=True)
        o = SIZE**2 if lo else 0
        got = y[:, o : o + out * out].reshape(n, out, out)
        assert (got - ref[:, 0]).abs().mean() < 1.0


def test_dropping_a_pending_batch_is_safe(shards):
    """A batch dropped before ``result()`` waits for its decode threads before its output array
    is freed (they would otherwise write into freed memory)."""
    import gc

    reader = Reader(shards, size=SIZE, threads=4)
    for _ in range(20):
        pending = [reader.submit(reader.sample(4, 8, 6.0)) for _ in range(4)]
        del pending
        gc.collect()
    assert reader.read(reader.sample(4, 8, 6.0)).video.shape == (4, 8, 3, SIZE, SIZE)


def isolated(code: str) -> str:
    """Run ``code`` in a fresh interpreter (a crash fails the test instead of killing pytest);
    returns its stdout. A Rust panic (reported by pyo3, the process survives) fails too.
    """
    import sys

    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, f"exit {done.returncode}: {done.stderr[-2000:]}"
    assert "panicked" not in done.stderr, done.stderr[-2000:]
    return done.stdout


def test_mistagged_matrix_and_encoder_errors(tmp_path):
    """A YUV source tagged with the RGB (identity) matrix is encoded as untagged (SVT-AV1
    rejects that matrix for 4:2:0), and an encoder that fails to open raises instead of
    crashing."""
    src = str(tmp_path / "rgb_tagged.mp4")
    cmd = [
        FFMPEG,
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=640x360:rate=30:duration=1",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-bsf:v",
        "h264_metadata=matrix_coefficients=0",
        src,
    ]
    subprocess.run(cmd, check=True)
    shard = str(tmp_path / "shard.zip")
    out = isolated(f"""
from kohakuclip import Reader, _core
_core.pack({shard!r}, [("a.mp4", _core.encode({src!r}, crf=40, preset=12))])
print(Reader([{shard!r}]).info(0).colorspace)
try:
    _core.encode({src!r}, crf=99)
except RuntimeError as e:
    print("raised", e)
""")
    lines = out.splitlines()
    assert lines[0] == "bt601"  # the SD guess for an untagged source
    assert lines[1].startswith("raised open encoder")


def test_dropping_a_failed_batch_is_safe(shards, tmp_path):
    """A batch whose ``result()`` raised can be dropped (its threads are done; no second wait)."""
    v = Reader(shards).info(0)
    folder = tmp_path / "videos"
    folder.mkdir()
    path = folder / "a.mp4"
    path.write_bytes(member_bytes(str(v.source), v.name))
    out = isolated(f"""
import gc, os
from kohakuclip import Reader
reader = Reader([{str(folder)!r}], size=64)
items = [(0, [0, 1, 2, 40])]  # parses the moov now
os.truncate({str(path)!r}, os.path.getsize({str(path)!r}) // 3)  # the samples are gone
pending = reader.submit(items)
try:
    pending.result()
except RuntimeError as e:
    print("raised", e)
del pending
gc.collect()
print("alive")
""")
    assert "raised" in out and out.strip().endswith("alive")
