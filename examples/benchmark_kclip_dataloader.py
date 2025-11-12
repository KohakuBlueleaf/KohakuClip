#!/usr/bin/env python3
"""Generate a synthetic video corpus and benchmark KohakuClip + PyTorch DataLoader."""

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import psutil
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import av

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from torch_dataset_utils import FolderVideoDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir", type=Path, help="Directory that will store generated MP4 files"
    )
    parser.add_argument(
        "--num-videos", type=int, default=10_000, help="How many videos to synthesize"
    )
    parser.add_argument(
        "--min-frames", type=int, default=64, help="Minimum frames per synthetic clip"
    )
    parser.add_argument(
        "--max-frames", type=int, default=512, help="Maximum frames per synthetic clip"
    )
    parser.add_argument("--size", type=int, default=384, help="Square frame resolution")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--sequence-length", type=int, default=64, help="Frames requested per dataset sample"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Stop after this many batches (for quick runs)",
    )
    parser.add_argument(
        "--metrics-csv", type=Path, default=None, help="Optional CSV path for timeline metrics"
    )
    parser.add_argument(
        "--leak-threshold-mb",
        type=float,
        default=256.0,
        help="Flag leak if memory grows beyond this",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Assume dataset_dir already has enough videos",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def synthesize_dataset(
    root: Path,
    *,
    num_videos: int,
    min_frames: int,
    max_frames: int,
    size: int,
    fps: int,
    seed: int,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(root.glob("*.mp4"))
    if len(existing) >= num_videos:
        print(f"Reusing {len(existing)} existing clips under {root}")
        return

    rng = random.Random(seed)
    print(f"Generating {num_videos - len(existing)} synthetic clips under {root} ...")
    for idx in tqdm(range(len(existing), num_videos), desc="Encoding videos"):
        frames = rng.randint(min_frames, max_frames)
        tensor = torch.randint(
            0,
            256,
            (frames, size, size, 3),
            dtype=torch.uint8,
        )
        path = root / f"clip_{idx:05d}.mp4"
        encode_video(tensor, path, fps)


def encode_video(frames: torch.Tensor, path: Path, fps: int) -> None:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = frames.shape[2]
    stream.height = frames.shape[1]
    stream.pix_fmt = "yuv420p"

    array = frames.numpy()
    for frame in array:
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def collect_memory_bytes(process: psutil.Process) -> int:
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return total


def run_benchmark(args: argparse.Namespace) -> None:
    if not args.skip_generation:
        synthesize_dataset(
            args.dataset_dir,
            num_videos=args.num_videos,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
            size=args.size,
            fps=args.fps,
            seed=args.seed,
        )

    dataset = FolderVideoDataset(
        args.dataset_dir,
        resize=(args.size, args.size),
        target_length=args.sequence_length,
        seed=args.seed,
    )
    print(f"Dataset contains {len(dataset)} videos")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    process = psutil.Process(os.getpid())
    ram_samples: list[int] = []
    records: list[dict[str, float]] = []
    total_batches = total_videos = total_frames = 0
    start = time.perf_counter()

    total_steps = args.max_batches or math.ceil(len(dataset) / args.batch_size)
    progress = tqdm(loader, total=total_steps, desc="Benchmarking")

    for batch_idx, batch in enumerate(progress):
        batch_videos = batch.size(0)
        batch_frames = batch_videos * batch.size(1)
        total_batches += 1
        total_videos += batch_videos
        total_frames += batch_frames

        elapsed = max(1e-9, time.perf_counter() - start)
        mem_bytes = collect_memory_bytes(process)
        ram_samples.append(mem_bytes)
        throughput_videos = total_videos / elapsed
        throughput_frames = total_frames / elapsed

        progress.set_postfix(
            vids=f"{throughput_videos:.1f} v/s",
            frames=f"{throughput_frames:.1f} f/s",
            mem=f"{mem_bytes / (1024 ** 3):.2f} GiB",
        )

        records.append(
            {
                "batch": total_batches,
                "time_sec": elapsed,
                "memory_bytes": mem_bytes,
                "memory_gib": mem_bytes / (1024**3),
                "videos": total_videos,
                "frames": total_frames,
                "throughput_videos": throughput_videos,
                "throughput_frames": throughput_frames,
            }
        )

        if args.max_batches and total_batches >= args.max_batches:
            break

    progress.close()
    duration = max(1e-9, time.perf_counter() - start)

    if not ram_samples:
        print("No batches were processed.")
        return

    peak_mem = max(ram_samples) / (1024**3)
    mem_growth_mb = (ram_samples[-1] - ram_samples[0]) / (1024**2)
    leak_detected = mem_growth_mb > args.leak_threshold_mb

    print("\n==== Benchmark Summary ====")
    print(f"Total batches   : {total_batches}")
    print(f"Videos processed: {total_videos}")
    print(f"Frames processed: {total_frames}")
    print(f"Elapsed time    : {duration:.2f}s")
    print(
        f"Throughput      : {total_videos / duration:.2f} videos/s, {total_frames / duration:.2f} frames/s"
    )
    print(f"Peak RSS        : {peak_mem:.2f} GiB")
    print(f"Mem growth      : {mem_growth_mb:.1f} MiB")
    print(
        f"Leak detected   : {'YES' if leak_detected else 'no'} (threshold={args.leak_threshold_mb} MiB)"
    )

    if args.metrics_csv:
        write_metrics_csv(args.metrics_csv, records)
        print(f"Wrote timeline metrics to {args.metrics_csv}")


def write_metrics_csv(path: Path, records: Iterable[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "batch",
                "time_sec",
                "memory_bytes",
                "memory_gib",
                "videos",
                "frames",
                "throughput_videos",
                "throughput_frames",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
