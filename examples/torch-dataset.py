#!/usr/bin/env python3
"""Demonstrate loading MP4 folders with KohakuClip + PyTorch."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.utils.data as data
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from torch_dataset_utils import FolderVideoDataset, save_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory that contains MP4 files")
    parser.add_argument(
        "--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(256, 256)
    )
    parser.add_argument(
        "--target-length", type=int, default=48, help="Frames per clip returned from __getitem__"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-sample", type=Path, default=None, help="Optional path to save a sample clip"
    )
    parser.add_argument(
        "--no-normalize", action="store_true", help="Return tensors in [0, 1] instead of [-1, 1]"
    )
    return parser.parse_args()


def default_transform(frames: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=frames.dtype, device=frames.device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=frames.dtype, device=frames.device).view(1, 3, 1, 1)
    return (frames - mean) / std


def main() -> None:
    args = parse_args()
    transform = None if args.no_normalize else default_transform

    dataset = FolderVideoDataset(
        args.root,
        resize=tuple(args.resize),
        target_length=args.target_length,
        transform=transform,
        seed=args.seed,
    )

    print(f"Found {len(dataset)} MP4 files under {args.root}")

    try:
        meta = dataset.metadata(0)
        print(
            f"Sample metadata: {meta.frame_count} frames @ {meta.width}x{meta.height}, "
            f"bitrate={meta.bit_rate}bps, file_size={meta.file_size / (1024 ** 2):.2f} MiB",
        )
    except Exception as exc:  # pragma: no cover - best effort logging
        print(f"Metadata lookup failed: {exc}")

    sample = dataset[random.randint(0, len(dataset) - 1)]
    print(f"Sample tensor shape: {tuple(sample.shape)}")

    if args.save_sample:
        save_video(sample, str(args.save_sample))
        print(f"Wrote sample clip to {args.save_sample}")

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    for batch_idx, batch in enumerate(tqdm(loader, desc="Streaming batches")):
        b, t, c, h, w = batch.shape
        tqdm.write(f"Batch {batch_idx}: {b} clips ({t}x{h}x{w})")
        if batch_idx >= 2:
            break


if __name__ == "__main__":
    main()
