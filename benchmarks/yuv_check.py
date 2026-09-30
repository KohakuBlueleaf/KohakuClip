"""The yuv modes vs rgb on the same clips: agreement (PSNR) and the GPU cost per frame.

python benchmarks/yuv_check.py SHARD_DIR [--batch 64] [--frames 8]
"""

import argparse
import glob
import os
import random
import time

import numpy as np
import torch

from kohakuclip import Augment, Reader, clip
from kohakuclip.torch import yuv_to_rgb


def timed(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    shards = sorted(glob.glob(os.path.join(a.shards, "*.zip")))
    readers = {
        m: Reader(
            shards, size=256, mode=m, threads=a.threads, augment=Augment(crop="center")
        )
        for m in ("rgb", "yuv", "yuv_resized")
    }
    rgb = readers["rgb"]
    rng = random.Random(0)
    # "yuv" windows are square with the batch's smallest short side: pick videos sharing a size
    vids = [v for v in range(len(rgb)) if min(rgb.info(v).h, rgb.info(v).w) == 512]
    items = []
    for _ in range(a.batch):
        vid = rng.choice(vids)
        v = rgb.info(vid)
        items.append((vid, clip(v.n, v.fps, a.frames, 6.0, rng)))
    frames = a.batch * a.frames
    host = torch.from_numpy(rgb.read(items).video).pin_memory()
    ref = host.cuda().float()
    ms = timed(lambda: host.cuda(non_blocking=True).float() / 127.5 - 1)
    print(
        f"rgb: GPU copy + to float {ms / frames * 1000:.1f} us/frame, {3 * 256 * 256} bytes/frame"
    )
    for mode in ("yuv", "yuv_resized"):
        batch = readers[mode].read(items)
        pinned = type(batch)(
            torch.from_numpy(batch.video).pin_memory().numpy(),
            batch.window,
            batch.flips,
            batch.colorspace,
        )
        out = (yuv_to_rgb(pinned, 256) + 1) * 127.5
        psnr = 10 * np.log10(255**2 / ((out - ref) ** 2).mean().item())
        ms = timed(lambda: yuv_to_rgb(pinned, 256))
        print(
            f"{mode}: PSNR vs rgb {psnr:.1f} dB; GPU copy + convert (+ resize) "
            f"{ms / frames * 1000:.1f} us/frame, {batch.video.shape[-1]} bytes/frame"
        )


if __name__ == "__main__":
    main()
