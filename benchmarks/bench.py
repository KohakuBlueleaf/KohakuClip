"""Throughput of the reader: ms per output frame per core, per sampling mode, per thread count.

    python benchmarks/bench.py SHARD_DIR [--threads 1 4 16] [--modes 8f@6 8f@24 16f@6 8rand]
                               [--mode rgb|yuv] [--procs] [--cold] [--batches 20] [--batch 16]

--procs also runs N single-threaded processes (DataLoader-worker style) for each thread count N.
--inflight K keeps K batches queued on the native pool (planning overlaps decoding; 0: synchronous).
--cold drops the shards from the page cache (posix_fadvise) before each run.
Prints one JSON line per run, with the per-stage native time per output frame.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import random
import time

import numpy as np

from kohakuclip import Augment, Reader, clip, profile, random_frames

MODES = {"8f@6": (8, 6.0), "8f@24": (8, 24.0), "16f@6": (16, 6.0), "8rand": (8, None)}


def sampler(reader: Reader, mode: str):
    frames, fps = MODES[mode]

    def sample(rng):
        vid = rng.randrange(len(reader))
        v = reader.info(vid)
        return vid, (
            random_frames(v.n, frames, rng)
            if fps is None
            else clip(v.n, v.fps, frames, fps, rng)
        )

    return sample


def run(shards, mode, threads, batches, batch, fmt, inflight, skip=True, seed=0):
    """Decode ``batches`` batches with up to ``inflight`` extra batches queued (0: one at a time)."""
    reader = Reader(
        shards,
        size=256,
        mode=fmt,
        threads=threads,
        augment=Augment(hflip=0.5),
        seed=seed,
        skip=skip,
    )
    sample, rng = sampler(reader, mode), random.Random(seed)
    reader.read([sample(rng) for _ in range(batch)])  # warm-up: decoders, first opens
    profile(reset=True)
    t = time.perf_counter()
    pending = []
    for _ in range(batches):
        pending.append(reader.submit([sample(rng) for _ in range(batch)]))
        if len(pending) > inflight:
            pending.pop(0).result()
    for p in pending:
        p.result()
    wall = time.perf_counter() - t
    return wall, profile(reset=True)


def _proc(args):
    shards, mode, batches, batch, fmt, skip, seed = args
    return run(shards, mode, 1, batches, batch, fmt, 0, skip, seed)


def drop_cache(shards):
    for s in shards:
        fd = os.open(s, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("shards")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--mode", default="rgb")
    ap.add_argument("--procs", action="store_true")
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument(
        "--inflight", type=int, default=2, help="batches queued ahead (threads)"
    )
    ap.add_argument(
        "--no-skip", action="store_true", help="decode samples nothing depends on too"
    )
    a = ap.parse_args()
    shards = sorted(glob.glob(os.path.join(a.shards, "*.zip")))
    for mode in a.modes:
        t = MODES[mode][0]
        for n in a.threads:
            kinds = [("threads", n)] + ([("procs", n)] if a.procs and n > 1 else [])
            for kind, k in kinds:
                if a.cold:
                    drop_cache(shards)
                if kind == "threads":
                    wall, prof = run(
                        shards,
                        mode,
                        k,
                        a.batches,
                        a.batch,
                        a.mode,
                        a.inflight,
                        not a.no_skip,
                    )
                    clips = a.batches * a.batch
                else:
                    with mp.get_context("spawn").Pool(k) as pool:
                        t0 = time.perf_counter()
                        rs = pool.map(
                            _proc,
                            [
                                (
                                    shards,
                                    mode,
                                    a.batches,
                                    a.batch,
                                    a.mode,
                                    not a.no_skip,
                                    i,
                                )
                                for i in range(k)
                            ],
                        )
                    wall = max(r[0] for r in rs)
                    prof = {s: sum(r[1][s] for r in rs) for s in rs[0][1]}
                    clips = k * a.batches * a.batch
                frames = clips * t
                row = dict(
                    mode=mode,
                    fmt=a.mode,
                    kind=kind,
                    cores=k,
                    skip=not a.no_skip,
                    clips_per_s=clips / wall,
                    frames_per_s=frames / wall,
                    ms_per_frame_per_core=1000 * wall * k / frames,
                    **{
                        f"{s}_ms_per_frame": 1000 * prof[s] / frames
                        for s in ("read", "decode", "convert", "resize")
                    },
                    decoded_per_out=prof["decoded"] / frames,
                )
                print(
                    json.dumps(
                        {
                            k2: (round(v, 3) if isinstance(v, float) else v)
                            for k2, v in row.items()
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
