"""Memory leak check: anonymous resident memory (RssAnon: excludes the memory-mapped shards'
page cache) over long runs of each native path. A leak shows as a steady slope after warm-up.

    python benchmarks/leak_check.py SHARD_DIR VIDEO.mp4 [--batches 2000]

1. Steady reading: one Reader, many batches (rgb and yuv).
2. Reader churn: create, read one batch, drop (thread pool and per-thread decoder teardown).
3. Encode churn: ``_core.encode`` from a path and from bytes, and ``_core.pack``.
"""

import argparse
import gc
import glob
import os
import tempfile

import numpy as np

from kohakuclip import Reader, _core


def rss_anon_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("no RssAnon")


def report(name: str, samples: list[float], unit: str) -> None:
    """Slope over the second half (after warm-up) and the overall change."""
    half = samples[len(samples) // 2 :]
    x = np.arange(len(half))
    slope = np.polyfit(x, half, 1)[0] if len(half) > 1 else 0.0
    print(
        f"{name}: {samples[0]:.1f} -> {samples[-1]:.1f} MB "
        f"(max {max(samples):.1f}); second-half slope {slope * 1000:+.2f} KB per {unit}",
        flush=True,
    )


def steady(shards, batches, mode):
    reader = Reader(shards, size=256, mode=mode, threads=8, hflip=0.5, seed=0)
    samples = []
    for i in range(batches):
        reader.read(reader.sample(16, 8, 6.0))
        if i % 25 == 0:
            samples.append(rss_anon_mb())
    report(
        f"steady reading ({mode}, {batches} batches x 16 clips)", samples, "25 batches"
    )


def churn(shards, rounds):
    samples = []
    for i in range(rounds):
        reader = Reader(shards, size=256, threads=8, seed=i)
        reader.read(reader.sample(16, 8, 6.0))
        del reader
        gc.collect()
        if i % 5 == 0:
            samples.append(rss_anon_mb())
    report(f"reader churn ({rounds} readers)", samples, "5 readers")


def encode_churn(video, rounds):
    with open(video, "rb") as f:
        data = f.read()
    samples = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(rounds):
            a = _core.encode(video, max_short=256)
            b = _core.encode(data, max_short=256)
            _core.pack(os.path.join(tmp, "s.zip"), [("a.mp4", a), ("b.mp4", b)])
            if i % 5 == 0:
                samples.append(rss_anon_mb())
    report(f"encode + pack churn ({rounds} rounds)", samples, "5 rounds")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shards")
    ap.add_argument("video")
    ap.add_argument("--batches", type=int, default=2000)
    ap.add_argument("--readers", type=int, default=200)
    ap.add_argument("--encodes", type=int, default=100)
    a = ap.parse_args()
    shards = sorted(glob.glob(os.path.join(a.shards, "*.zip")))

    os.environ.setdefault("SVT_LOG", "1")
    print(f"start: {rss_anon_mb():.1f} MB", flush=True)
    steady(shards, a.batches, "rgb")
    steady(shards, a.batches // 2, "yuv")
    churn(shards, a.readers)
    encode_churn(a.video, a.encodes)


if __name__ == "__main__":
    main()
