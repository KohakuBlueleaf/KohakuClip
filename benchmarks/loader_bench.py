"""The PyTorch side of loading, without a model: batches per second into the GPU and the CPU
time the main (training) thread spends per batch, per way of running ``ClipDataset``.

    python benchmarks/loader_bench.py SHARD_DIR [--batches 300] [--batch 16]

The main thread's CPU time is what competes with a host-bound training step; the decode
threads and worker processes run elsewhere.
"""

import argparse
import glob
import os
import resource
import time

import torch

from kohakuclip.torch import ClipDataset, ClipLoader


def thread_cpu() -> float:
    usage = resource.getrusage(resource.RUSAGE_THREAD)
    return usage.ru_utime + usage.ru_stime


def run(shards, batches, batch, workers, threads, frames, fps):
    dataset = ClipDataset(
        shards,
        batch,
        frames=frames,
        fps=fps,
        size=256,
        threads=threads,
        hflip=0.5,
        seed=0,
    )
    loader = iter(ClipLoader(dataset, workers=workers, prefetch=4))
    stream = torch.cuda.Stream()

    for _ in range(20):  # warm-up: workers, decoders, first opens
        next(loader)
    torch.cuda.synchronize()

    wall, cpu = time.perf_counter(), thread_cpu()
    for _ in range(batches):
        video = next(loader)
        with torch.cuda.stream(stream):
            video.cuda(non_blocking=True)
    stream.synchronize()
    wall, cpu = time.perf_counter() - wall, thread_cpu() - cpu
    return batches / wall, 1000 * cpu / batches


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shards")
    ap.add_argument("--batches", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--fps", type=float, default=6.0)
    a = ap.parse_args()
    shards = sorted(glob.glob(os.path.join(a.shards, "*.zip")))

    configs = [(0, 8), (0, 16), (0, 24), (1, 16), (2, 8), (2, 12), (4, 4), (4, 6)]
    for workers, threads in configs:
        rate, cpu = run(shards, a.batches, a.batch, workers, threads, a.frames, a.fps)
        clips = rate * a.batch
        print(
            f"workers {workers} x threads {threads:2d}: {rate:6.1f} batches/s "
            f"({clips:6.0f} clips/s), main thread {cpu:6.2f} ms CPU per batch",
            flush=True,
        )


if __name__ == "__main__":
    main()
