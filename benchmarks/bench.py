"""Throughput of the reader: ms per output frame per core, per clip kind, per thread count.

    python benchmarks/bench.py SHARD_DIR [--threads 1 4 16] [--clips 8f@6 8f@24 16f@6 8rand]
                               [--mode rgb|yuv|yuv_resized] [--procs] [--cold]
                               [--batches 20] [--batch 16] [--inflight 2] [--no-skip]

--procs also runs N single-threaded processes (DataLoader-worker style) for each thread count N.
--inflight K keeps K batches queued on the native pool (0: one batch at a time).
--cold drops the shards from the page cache (posix_fadvise) before each run.
Prints one JSON line per run, with the native time per stage per output frame.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import time
from collections import deque

from kohakuclip import Reader, profile

# frames, fps (None: random frames of the whole video)
CLIPS = {"8f@6": (8, 6.0), "8f@24": (8, 24.0), "16f@6": (16, 6.0), "8rand": (8, None)}
STAGES = ("read", "decode", "convert", "resize")


def run(shards, clip, threads, batches, batch, mode, inflight, skip=True, seed=0):
    """Decode ``batches`` batches with up to ``inflight`` more queued; (wall seconds, profile)."""
    reader = Reader(
        shards, size=256, mode=mode, threads=threads, hflip=0.5, seed=seed, skip=skip
    )
    frames, fps = CLIPS[clip]

    def items():
        return reader.sample(batch, frames, fps, spread=fps is None)

    reader.read(items())  # warm-up: threads, decoders, first opens
    profile(reset=True)

    start = time.perf_counter()
    pending = deque()
    for _ in range(batches):
        pending.append(reader.submit(items()))
        if len(pending) > inflight:
            pending.popleft().result()
    for p in pending:
        p.result()
    return time.perf_counter() - start, profile(reset=True)


def run_process(args):
    shards, clip, batches, batch, mode, skip, seed = args
    return run(shards, clip, 1, batches, batch, mode, 0, skip, seed)


def drop_cache(shards):
    for path in shards:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def report(row: dict) -> None:
    rounded = {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}
    print(json.dumps(rounded), flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("shards")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--clips", nargs="+", default=list(CLIPS))
    ap.add_argument("--mode", default="rgb")
    ap.add_argument("--procs", action="store_true")
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--inflight", type=int, default=2)
    ap.add_argument("--no-skip", action="store_true")
    a = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(a.shards, "*.zip")))
    shards += sorted(glob.glob(os.path.join(a.shards, "*.tar")))
    skip = not a.no_skip

    for clip in a.clips:
        frames_per_clip = CLIPS[clip][0]
        for n in a.threads:
            kinds = [("threads", n)]
            if a.procs and n > 1:
                kinds.append(("procs", n))

            for kind, k in kinds:
                if a.cold:
                    drop_cache(shards)

                if kind == "threads":
                    args = (
                        shards,
                        clip,
                        k,
                        a.batches,
                        a.batch,
                        a.mode,
                        a.inflight,
                        skip,
                    )
                    wall, prof = run(*args)
                    clips = a.batches * a.batch
                else:
                    jobs = [
                        (shards, clip, a.batches, a.batch, a.mode, skip, i)
                        for i in range(k)
                    ]
                    with mp.get_context("spawn").Pool(k) as pool:
                        results = pool.map(run_process, jobs)
                    wall = max(r[0] for r in results)
                    prof = {s: sum(r[1][s] for r in results) for s in results[0][1]}
                    clips = k * a.batches * a.batch

                frames = clips * frames_per_clip
                report(
                    dict(
                        clip=clip,
                        mode=a.mode,
                        kind=kind,
                        cores=k,
                        skip=skip,
                        clips_per_s=clips / wall,
                        ms_per_frame_per_core=1000 * wall * k / frames,
                        **{f"{s}_ms": 1000 * prof[s] / frames for s in STAGES},
                        decoded_per_frame=prof["decoded"] / frames,
                    )
                )


if __name__ == "__main__":
    main()
