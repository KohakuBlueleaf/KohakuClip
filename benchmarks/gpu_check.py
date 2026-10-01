"""The GPU side: yuv -> rgb implementations (agreement and time) and host-to-GPU transfer costs.

    python benchmarks/gpu_check.py SHARD_DIR [--batch 16] [--frames 8]

1. ``yuv_to_rgb``: plain torch, ``torch.compile`` of it, and the Triton kernel, per output mode
   (agreement with the torch reference, and with rgb mode; microseconds per frame).
2. Transfers per batch: pageable vs pinned copies, pinned allocation per batch vs a reused
   buffer, and the host memcpy when decoding into pageable memory then pinning.
"""

import argparse
import glob
import os
import time

import numpy as np
import torch

from kohakuclip import Reader
from kohakuclip.gpu import yuv_to_rgb_triton
from kohakuclip.torch import yuv_to_rgb_torch


def cuda_time(fn, reps: int = 20) -> float:
    """Milliseconds per call (CUDA events, after a warm-up)."""
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def wall_time(fn, reps: int = 20) -> float:
    """Milliseconds per call (host wall clock, synchronized)."""
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR in dB of two [-1, 1] tensors, on the 0-255 scale."""
    mse = (((a.float() - b.float()) * 127.5) ** 2).mean().item()
    return 10 * np.log10(255**2 / max(mse, 1e-12))


def check_kernels(shards, batch, frames, size):
    rgb_reader = Reader(shards, size=size, crop="center", threads=16)
    first = Reader(shards, size=size, crop="center")
    # "yuv" needs one stored short side per batch: keep to 512p videos
    short = [
        v for v in range(len(first)) if min(first.info(v).h, first.info(v).w) == 512
    ]
    items = [(v, list(range(0, 8 * frames, 8))) for v in short[:batch]]
    rgb = torch.as_tensor(rgb_reader.read(items).video).cuda().float() / 127.5 - 1

    for mode in ("yuv_resized", "yuv"):
        reader = Reader(shards, size=size, mode=mode, crop="center", threads=16)
        check_mode(mode, reader.read(items), rgb, size)


def check_mode(mode, b, rgb, size):
    """Agreement and time per frame of the yuv -> rgb implementations on one batch."""
    video = torch.as_tensor(b.video).cuda()
    window = (int(b.window[0][0]), int(b.window[0][1]))
    flips = torch.as_tensor(b.flips)
    n = video.shape[0] * video.shape[1]

    def triton_fp32():
        return yuv_to_rgb_triton(video, window, b.colorspace, flips, size)

    def triton_bf16():
        return yuv_to_rgb_triton(
            video, window, b.colorspace, flips, size, torch.bfloat16
        )

    def eager():
        return yuv_to_rgb_torch(video, b, size)

    compiled_fn = torch.compile(yuv_to_rgb_torch, dynamic=False)

    def compiled():
        return compiled_fn(video, b, size)

    reference, fast = eager(), triton_fp32()
    diff = (fast - reference).abs().max().item() * 127.5
    print(
        f"{mode}: window {window}, {n} frames; triton vs torch max |diff| {diff:.3f} "
        f"(0-255), PSNR vs rgb mode: torch {psnr(reference, rgb):.1f} dB, "
        f"triton {psnr(fast, rgb):.1f} dB"
    )
    implementations = {
        "torch": eager,
        "torch.compile": compiled,
        "triton": triton_fp32,
        "triton bf16": triton_bf16,
    }
    for name, fn in implementations.items():
        print(f"  {name:14s} {cuda_time(fn) * 1000 / n:7.2f} us/frame")


def check_transfers(batch, frames, size):
    shapes = {
        "rgb": (batch, frames, 3, size, size),
        "yuv_resized": (batch, frames, size * size * 3 // 2),
        "yuv (512p)": (batch, frames, 512 * 512 * 3 // 2),
    }
    stream = torch.cuda.Stream()
    for name, shape in shapes.items():
        check_transfer(name, shape, stream)


def check_transfer(name, shape, stream):
    """Copy and pinning costs of one batch shape."""
    pageable = torch.randint(0, 255, shape, dtype=torch.uint8)
    pinned = pageable.pin_memory()

    def copy_pageable():
        pageable.cuda()

    def copy_pinned():
        with torch.cuda.stream(stream):
            pinned.cuda(non_blocking=True)
        stream.synchronize()

    def alloc_pinned():
        torch.empty(shape, dtype=torch.uint8, pin_memory=True)

    def repin():
        pageable.pin_memory()

    print(
        f"{name}: {pageable.numel() / 1e6:.1f} MB per batch; copy pageable "
        f"{wall_time(copy_pageable):.2f} ms, pinned {wall_time(copy_pinned):.2f} ms; "
        f"new pinned buffer {wall_time(alloc_pinned):.3f} ms (caching allocator); "
        f"pin a pageable batch {wall_time(repin):.2f} ms"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shards")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()
    shards = sorted(glob.glob(os.path.join(a.shards, "*.zip")))
    print(torch.cuda.get_device_name())
    check_kernels(shards, a.batch, a.frames, a.size)
    check_transfers(a.batch, a.frames, a.size)


if __name__ == "__main__":
    main()
