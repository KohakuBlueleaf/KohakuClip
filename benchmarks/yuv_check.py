"""YUV mode vs RGB mode on the same clips: agreement (PSNR) and the GPU cost of ``yuv_to_rgb``.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    shards = sorted(glob.glob(os.path.join(a.shards, "*.zip")))
    rgb = Reader(shards, size=256, mode="rgb", threads=a.threads, augment=Augment(crop="center"))
    yuv = Reader(shards, size=256, mode="yuv", threads=a.threads, augment=Augment(crop="center"))
    rng = random.Random(0)
    # the YUV window is square with the batch's smallest short side: pick videos sharing a size
    vids = [v for v in range(len(rgb)) if min(rgb.info(v).h, rgb.info(v).w) == 512]
    items = []
    for _ in range(a.batch):
        vid = rng.choice(vids)
        v = rgb.info(vid)
        items.append((vid, clip(v.n, v.fps, a.frames, 6.0, rng)))
    ref = torch.from_numpy(rgb.read(items).video).cuda().float()
    batch = yuv.read(items)
    out = (yuv_to_rgb(batch, 256) + 1) * 127.5
    mse = ((out - ref) ** 2).mean().item()
    print(f"YUV (GPU convert + resize) vs RGB (CPU fused): PSNR {10 * np.log10(255 ** 2 / mse):.1f} dB, "
          f"mean |diff| {(out - ref).abs().mean().item():.2f}")
    # GPU cost per batch from pinned memory: copy + conversion (YUV) vs copy only (RGB)
    yuv_host = torch.from_numpy(batch.video).pin_memory()
    rgb_host = torch.from_numpy(rgb.read(items).video).pin_memory()
    frames = a.batch * a.frames

    def timed(fn, reps=20):
        fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / reps * 1000

    pinned = type(batch)(yuv_host.numpy(), batch.window, batch.flips, batch.colorspace)
    ms_yuv = timed(lambda: yuv_to_rgb(pinned, 256))
    ms_rgb = timed(lambda: (rgb_host.cuda(non_blocking=True).float() / 127.5 - 1))
    print(f"GPU per frame: yuv copy + convert + resize {ms_yuv / frames * 1000:.1f} us, rgb copy + to float "
          f"{ms_rgb / frames * 1000:.1f} us; bytes per frame yuv {batch.video.shape[-1]} vs rgb {3 * 256 * 256}")

if __name__ == "__main__":
    main()
