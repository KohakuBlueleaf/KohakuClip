#!/usr/bin/env python3
"""Generate a synthetic video corpus and benchmark KohakuClip + PyTorch DataLoader."""

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import numpy as np
import psutil
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
        "--metrics-csv", type=Path, default=None, help="Optional CSV path for per-batch metrics"
    )
    parser.add_argument(
        "--ram-csv", type=Path, default=None, help="Optional CSV path for RAM checkpoints"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Assume dataset_dir already has enough videos",
    )
    parser.add_argument(
        "--gen-workers", type=int, default=16, help="Processes used when encoding videos"
    )
    parser.add_argument(
        "--unique-videos",
        type=int,
        default=None,
        help="Number of unique clips to encode before reusing/copying (defaults to num-videos)",
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
    workers: int = 1,
    unique_videos: int | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(root.glob("*.mp4"))
    if len(existing) >= num_videos:
        print(f"Reusing {len(existing)} existing clips under {root}")
        return
    unique_target = min(unique_videos or num_videos, num_videos)
    encode_needed = max(0, unique_target - len(existing))
    if encode_needed > 0:
        print(f"Encoding {encode_needed} unique clips with {workers} workers ...")
        tasks = [
            (
                root,
                idx,
                min_frames,
                max_frames,
                size,
                fps,
                seed + idx,
            )
            for idx in range(len(existing), len(existing) + encode_needed)
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(
                tqdm(
                    pool.map(_encode_worker, tasks),
                    total=encode_needed,
                    desc="Encoding videos",
                )
            )

    existing = sorted(root.glob("*.mp4"))
    total = len(existing)
    if total >= num_videos:
        return

    print(f"Copying {num_videos - total} clips to reach target size ...")
    if total == 0:
        raise RuntimeError("No videos available to copy after generation")

    copy_missing_videos(existing, root, num_videos - total)


def copy_missing_videos(
    existing: list[Path], root: Path, copies_needed: int, *, batch_size: int = 8
) -> None:
    if copies_needed <= 0:
        return

    copy_plan: dict[Path, list[Path]] = {}
    start_index = len(existing)
    for offset in range(copies_needed):
        src = existing[offset % len(existing)]
        dst = root / f"clip_{start_index + offset:05d}.mp4"
        copy_plan.setdefault(src, []).append(dst)

    progress = tqdm(total=copies_needed, desc="Copying videos")
    sources = list(copy_plan.items())
    for batch_start in range(0, len(sources), batch_size):
        batch = sources[batch_start : batch_start + batch_size]
        buffers = {src: src.read_bytes() for src, _ in batch}
        for src, destinations in batch:
            data = buffers[src]
            for dst in destinations:
                dst.write_bytes(data)
                progress.update(1)
    progress.close()


def _encode_worker(args: tuple[Path, int, int, int, int, int, int]) -> None:
    root, idx, min_frames, max_frames, size, fps, seed = args
    rng = np.random.default_rng(seed)
    frame_count = int(rng.integers(min_frames, max_frames + 1))
    frames = generate_geometric_frames(rng, frame_count, size)
    path = root / f"clip_{idx:05d}.mp4"
    encode_video(frames, path, fps)


def generate_geometric_frames(rng: np.random.Generator, frame_count: int, size: int) -> np.ndarray:
    frames = np.empty((frame_count, size, size, 3), dtype=np.uint8)
    base_color = rng.integers(0, 128, size=3, dtype=np.int32)
    color_delta = rng.integers(-2, 3, size=3, dtype=np.int32)
    color_steps = np.clip(
        base_color + np.arange(frame_count, dtype=np.int32)[:, None] * color_delta,
        0,
        255,
    ).astype(np.uint8)
    shapes = _create_shapes(rng, size)

    for idx in range(frame_count):
        frame = frames[idx]
        frame[:] = color_steps[idx]
        for shape in shapes:
            if shape["kind"] == "rect":
                _render_rect(frame, shape)
            else:
                _render_disk(frame, shape)
            _advance_shape(shape, size)
    return frames


def _create_shapes(
    rng: np.random.Generator, size: int
) -> list[dict[str, float | np.ndarray | str]]:
    shape_count = int(rng.integers(2, 6))
    shapes: list[dict[str, float | np.ndarray | str]] = []
    for _ in range(shape_count):
        kind = rng.choice(["rect", "disk"])  # type: ignore[arg-type]
        color = rng.integers(64, 255, size=3, dtype=np.uint8)
        speed = rng.uniform(0.5, 2.5)
        angle = rng.uniform(0, 2 * math.pi)
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        if kind == "rect":
            min_half = max(2, size // 16)
            max_half = max(min_half + 1, size // 6)
            half_w = float(rng.integers(min_half, max_half))
            half_h = float(rng.integers(min_half, max_half))
            margin_x = half_w + 1
            margin_y = half_h + 1
            shapes.append(
                {
                    "kind": "rect",
                    "x": rng.uniform(margin_x, size - margin_x),
                    "y": rng.uniform(margin_y, size - margin_y),
                    "vx": vx,
                    "vy": vy,
                    "half_w": half_w,
                    "half_h": half_h,
                    "color": color,
                }
            )
        else:
            min_radius = max(2, size // 20)
            max_radius = max(min_radius + 1, size // 8)
            radius = float(rng.integers(min_radius, max_radius))
            margin = radius + 1
            shapes.append(
                {
                    "kind": "disk",
                    "x": rng.uniform(margin, size - margin),
                    "y": rng.uniform(margin, size - margin),
                    "vx": vx,
                    "vy": vy,
                    "radius": radius,
                    "color": color,
                }
            )
    return shapes


def _advance_shape(shape: dict[str, float | np.ndarray | str], size: int) -> None:
    shape["x"] = float(shape["x"]) + float(shape["vx"])
    shape["y"] = float(shape["y"]) + float(shape["vy"])

    if shape["kind"] == "rect":
        half_w = float(shape["half_w"])
        half_h = float(shape["half_h"])
        min_x = half_w
        max_x = size - half_w
        min_y = half_h
        max_y = size - half_h
    else:
        radius = float(shape["radius"])
        min_x = radius
        max_x = size - radius
        min_y = radius
        max_y = size - radius

    if shape["x"] <= min_x or shape["x"] >= max_x:
        shape["vx"] = -float(shape["vx"])
        shape["x"] = float(np.clip(shape["x"], min_x, max_x))
    if shape["y"] <= min_y or shape["y"] >= max_y:
        shape["vy"] = -float(shape["vy"])
        shape["y"] = float(np.clip(shape["y"], min_y, max_y))


def _render_rect(frame: np.ndarray, shape: dict[str, float | np.ndarray | str]) -> None:
    half_w = int(shape["half_w"])
    half_h = int(shape["half_h"])
    cx = float(shape["x"])
    cy = float(shape["y"])
    x0 = max(0, int(round(cx - half_w)))
    x1 = min(frame.shape[1], int(round(cx + half_w)))
    y0 = max(0, int(round(cy - half_h)))
    y1 = min(frame.shape[0], int(round(cy + half_h)))
    if x0 >= x1 or y0 >= y1:
        return
    frame[y0:y1, x0:x1] = shape["color"]


def _render_disk(frame: np.ndarray, shape: dict[str, float | np.ndarray | str]) -> None:
    radius = float(shape["radius"])
    cx = float(shape["x"])
    cy = float(shape["y"])
    x0 = max(0, int(math.floor(cx - radius)))
    x1 = min(frame.shape[1], int(math.ceil(cx + radius)))
    y0 = max(0, int(math.floor(cy - radius)))
    y1 = min(frame.shape[0], int(math.ceil(cy + radius)))
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    region = frame[y0:y1, x0:x1]
    region[mask] = shape["color"]


def encode_video(frames: np.ndarray, path: Path, fps: int) -> None:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = frames.shape[2]
    stream.height = frames.shape[1]
    stream.pix_fmt = "yuv420p"

    for frame in frames:
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
            workers=args.gen_workers,
            unique_videos=args.unique_videos,
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
    bench_clock = time.perf_counter()
    ram_samples: list[int] = []
    ram_checkpoints: list[dict[str, float | str]] = []

    def record_ram(stage: str) -> int:
        value = collect_memory_bytes(process)
        ram_samples.append(value)
        ram_checkpoints.append(
            {
                "stage": stage,
                "memory_bytes": value,
                "memory_gib": value / (1024**3),
                "time_sec": time.perf_counter() - bench_clock,
            }
        )
        return value

    record_ram("start")
    record_ram("after_dataset")

    record_ram("before_loop")

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
        mem_bytes = record_ram(f"batch_{total_batches}")
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
    record_ram("after_loop")
    duration = max(1e-9, time.perf_counter() - start)

    if not ram_samples:
        print("No batches were processed.")
        return

    print("\n==== Benchmark Summary ====")
    print(f"Total batches   : {total_batches}")
    print(f"Videos processed: {total_videos}")
    print(f"Frames processed: {total_frames}")
    print(f"Elapsed time    : {duration:.2f}s")
    print(
        f"Throughput      : {total_videos / duration:.2f} videos/s, {total_frames / duration:.2f} frames/s"
    )
    mem_min = min(ram_samples) / (1024**3)
    mem_max = max(ram_samples) / (1024**3)
    mem_mean = mean(ram_samples) / (1024**3)
    mem_std = (pstdev(ram_samples) if len(ram_samples) > 1 else 0.0) / (1024**3)
    print(
        f"RAM stats (GiB) : min={mem_min:.2f}  mean={mem_mean:.2f}  max={mem_max:.2f}  std={mem_std:.3f}"
    )
    print("Checkpoints:")
    for ckpt in ram_checkpoints:
        print(f"  {ckpt['stage']:>16}: {ckpt['memory_gib']:.2f} GiB @ {ckpt['time_sec']:.2f}s")

    if args.metrics_csv:
        write_metrics_csv(args.metrics_csv, records)
        print(f"Wrote timeline metrics to {args.metrics_csv}")
    if args.ram_csv:
        write_ram_csv(args.ram_csv, ram_checkpoints)
        print(f"Wrote RAM checkpoints to {args.ram_csv}")


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


def write_ram_csv(path: Path, records: Iterable[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stage", "time_sec", "memory_bytes", "memory_gib"],
        )
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
