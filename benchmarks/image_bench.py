"""Throughput of the image reader: images/s and ms per image per core, per thread count, crop
and DCT shrinking; Pillow and torchvision baselines on the same JPEG bytes.

    python benchmarks/image_bench.py SHARD_DIR [--threads 1 16 32 64] [--crop random resized]
                                     [--batch 256] [--batches 40] [--no-dct] [--baselines]
                                     [--loaders] [--skip-reader] [--cold]
                                     [--no-readahead]

Reader runs: ``inflight`` batches queued on the native pool, uint8 rgb output at 256, reads from
the shards (page cache warm after the first pass; ``--cold`` drops the shards from the page
cache before each run, with posix_fadvise). Baselines (``--baselines``): N processes, each
decoding its share of the same images from bytes already in memory (no file reads): Pillow
(``draft`` = libjpeg-turbo DCT shrinking, antialiased bilinear resize, crop) and
``torchvision.io.decode_jpeg`` + ``F.interpolate(antialias=True)`` + crop. Loaders
(``--loaders``): batches into pinned memory of this process, ``ImageDataset`` with N decode
threads vs a DataLoader with N worker processes over a zip + Pillow dataset (the usual way).
Prints one JSON line per run.
"""

import argparse
import glob
import io
import json
import multiprocessing as mp
import os
import time
import zipfile
from collections import deque

import numpy as np

from kohakuclip import ImageReader, profile

STAGES = ("read", "decode", "convert", "resize")


def run_reader(shards, threads, crop, dct, batch, batches, inflight=2, readahead=True):
    reader = ImageReader(
        shards,
        size=256,
        threads=threads,
        crop=crop,
        hflip=0.5,
        dct_scale=dct,
        readahead=readahead,
        seed=0,
    )
    reader.read(reader.sample(batch))  # warm-up: threads, decoders
    profile(reset=True)
    start = time.perf_counter()
    pending = deque()
    for _ in range(batches):
        pending.append(reader.submit(reader.sample(batch)))
        if len(pending) > inflight:
            pending.popleft().result()
    for p in pending:
        p.result()
    wall = time.perf_counter() - start
    return wall, profile(reset=True)


def drop_cache(shards):
    for path in shards:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def load_bytes(shards, count, seed):
    """``count`` random JPEGs of the shards, as bytes."""
    reader = ImageReader(shards)
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(reader), count)
    out = []
    opened = {}
    for i in ids:
        info = reader.info(int(i))
        z = opened.setdefault(info.source, zipfile.ZipFile(info.source))
        out.append(z.read(info.name))
    return out


def pillow_one(data, crop):
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    if crop == "random":
        img.draft("RGB", (256, 256))
    img = img.convert("RGB")
    w, h = img.size
    scale = 256 / min(w, h)
    size = (max(256, round(w * scale)), max(256, round(h * scale)))
    img = img.resize(size, Image.Resampling.BILINEAR)
    left = (size[0] - 256) // 2
    top = (size[1] - 256) // 2
    return np.asarray(img.crop((left, top, left + 256, top + 256)))


def torchvision_one(data):
    import torch
    import torch.nn.functional as F
    from torchvision.io import decode_jpeg

    x = decode_jpeg(torch.frombuffer(bytearray(data), dtype=torch.uint8))
    h, w = x.shape[1:]
    scale = 256 / min(h, w)
    size = (max(256, round(h * scale)), max(256, round(w * scale)))
    x = F.interpolate(x[None].float(), size=size, mode="bilinear", antialias=True)
    top = (size[0] - 256) // 2
    left = (size[1] - 256) // 2
    return x[0, :, top : top + 256, left : left + 256].to(torch.uint8)


def baseline_worker(args):
    kind, shards, count, seed = args
    os.environ["OMP_NUM_THREADS"] = "1"
    if kind == "torchvision":
        import torch

        torch.set_num_threads(1)
    blobs = load_bytes(shards, count, seed)
    one = (
        torchvision_one
        if kind == "torchvision"
        else (lambda b: pillow_one(b, "random"))
    )
    for b in blobs[:16]:
        one(b)
    start = time.perf_counter()
    for b in blobs:
        one(b)
    return time.perf_counter() - start, len(blobs)


class ZipJpegDataset:
    """The usual PyTorch way: one image per item, read from the zip shards and decoded with
    Pillow (``draft``, antialiased bilinear resize, random crop) in DataLoader workers.
    """

    def __init__(self, shards):
        self.items = []
        for path in shards:
            with zipfile.ZipFile(path) as z:
                names = [n for n in z.namelist() if n != "__index__.bin"]
            self.items += [(path, n) for n in names]
        self.opened = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        from PIL import Image

        path, name = self.items[i]
        if path not in self.opened:
            self.opened[path] = zipfile.ZipFile(path)
        img = Image.open(io.BytesIO(self.opened[path].read(name)))
        img.draft("RGB", (256, 256))
        img = img.convert("RGB")
        w, h = img.size
        scale = 256 / min(w, h)
        size = (max(256, round(w * scale)), max(256, round(h * scale)))
        img = img.resize(size, Image.Resampling.BILINEAR)
        rng = np.random.default_rng(i)
        left = int(rng.integers(0, size[0] - 255))
        top = int(rng.integers(0, size[1] - 255))
        crop = np.asarray(img.crop((left, top, left + 256, top + 256)))
        return torch.from_numpy(crop.copy()).permute(2, 0, 1)


def run_loader(kind, shards, workers, batch, batches):
    """Batches per second into (pinned, with CUDA) memory of the training process: ``ImageDataset``
    (decode threads in this process) or a DataLoader of ``ZipJpegDataset`` (worker processes).
    Returns (images/s, CPU ms of the main thread per batch)."""
    import resource

    import torch
    from torch.utils.data import DataLoader, RandomSampler

    from kohakuclip.torch import ClipLoader, ImageDataset

    pin = torch.cuda.is_available()  # pinned memory needs a CUDA device
    if kind == "kohakuclip":
        dataset = ImageDataset(
            shards, batch, threads=workers, hflip=0.5, inflight=3, pin=pin
        )
        loader = iter(ClipLoader(dataset))
    else:
        data = ZipJpegDataset(shards)
        loader = iter(
            DataLoader(
                data,
                batch_size=batch,
                sampler=RandomSampler(data, replacement=True, num_samples=1 << 40),
                num_workers=workers,
                pin_memory=pin,
                prefetch_factor=4,
                persistent_workers=True,
            )
        )
    for _ in range(5):  # warm-up: workers, decoders
        next(loader)

    def thread_cpu():
        usage = resource.getrusage(resource.RUSAGE_THREAD)
        return usage.ru_utime + usage.ru_stime

    cpu = thread_cpu()
    start = time.perf_counter()
    for _ in range(batches):
        out = next(loader)
        assert out.shape == (batch, 3, 256, 256) and out.is_pinned() == pin
    wall = time.perf_counter() - start
    del loader
    return batch * batches / wall, 1000 * (thread_cpu() - cpu) / batches


def report(row: dict) -> None:
    rounded = {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}
    print(json.dumps(rounded), flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("shards")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 16, 32, 64])
    ap.add_argument("--crop", nargs="+", default=["random", "resized"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--no-dct", action="store_true")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--loaders", action="store_true")
    ap.add_argument("--skip-reader", action="store_true")
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--no-readahead", action="store_true")
    a = ap.parse_args()
    shards = sorted(glob.glob(os.path.join(a.shards, "*.zip")))

    for crop in [] if a.skip_reader else a.crop:
        for dct in [True] + ([False] if a.no_dct else []):
            for n in a.threads:
                batches = a.batches if n > 1 else max(4, a.batches // 16)
                if a.cold:
                    drop_cache(shards)
                wall, prof = run_reader(
                    shards, n, crop, dct, a.batch, batches, readahead=not a.no_readahead
                )
                images = batches * a.batch
                report(
                    dict(
                        kind="kohakuclip",
                        crop=crop,
                        dct_scale=dct,
                        cold=a.cold,
                        readahead=not a.no_readahead,
                        threads=n,
                        images_per_s=images / wall,
                        ms_per_image_per_core=1000 * wall * n / images,
                        **{f"{s}_ms": 1000 * prof[s] / images for s in STAGES},
                    )
                )

    if a.loaders:
        for kind in ("kohakuclip", "pillow-dataloader"):
            for n in a.threads:
                if n == 1:
                    continue
                images_s, main_ms = run_loader(kind, shards, n, a.batch, a.batches)
                report(
                    dict(
                        kind=f"loader:{kind}",
                        workers_or_threads=n,
                        images_per_s=images_s,
                        main_thread_cpu_ms_per_batch=main_ms,
                    )
                )

    if a.baselines:
        for kind in ("pillow", "torchvision"):
            for n in a.threads:
                count = 2000 if n > 1 else 1000
                jobs = [(kind, shards, count, seed) for seed in range(n)]
                with mp.get_context("spawn").Pool(n) as pool:
                    results = pool.map(baseline_worker, jobs)
                per_image = [wall / k for wall, k in results]
                report(
                    dict(
                        kind=kind,
                        crop="random",
                        processes=n,
                        images_per_s=sum(1 / t for t in per_image),
                        ms_per_image_per_core=1000 * float(np.mean(per_image)),
                    )
                )


if __name__ == "__main__":
    main()
