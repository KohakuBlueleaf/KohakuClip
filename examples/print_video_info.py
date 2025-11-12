#!/usr/bin/env python3
"""
Summarize the basic properties of a video as seen by kohakuclip.

Usage:
    python examples/print_video_info.py /path/to/video.mp4
"""

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from kohakuclip import KClip


def _format_bytes(num_bytes: int) -> str:
    suffixes: Iterable[str] = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(num_bytes)
    for suffix in suffixes:
        if value < 1024.0 or suffix == "TiB":
            return f"{value:.2f} {suffix}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a video using kohakuclip.")
    parser.add_argument("video", type=Path, help="Path to the input video file.")
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="First frame index to include (default: 0).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Exclusive end frame index.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Frame stride.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Resize frames to WIDTH x HEIGHT via native scaler.",
    )
    parser.add_argument(
        "--sample-frame",
        type=int,
        default=None,
        metavar="INDEX",
        help="Optionally print basic statistics for a specific frame index.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clip = KClip(args.video)
    if args.start is not None or args.end is not None or args.step != 1:
        clip = clip.range(args.start, args.end, args.step)
    if args.resize:
        width, height = args.resize
        clip = clip.resize(width, height)
    frames = clip.to_array()

    frame_count, height, width, channels = frames.shape
    print(f"Video      : {args.video}")
    print(f"Frames     : {frame_count}")
    print(f"Resolution : {width}x{height}")
    print(f"Channels   : {channels}")
    print(f"Dtype      : {frames.dtype}")
    print(f"Buffer size: {_format_bytes(frames.nbytes)} (numpy)")

    if args.sample_frame is not None:
        idx = max(0, min(args.sample_frame, frame_count - 1))
        sample = frames[idx].astype(np.float32)
        print(f"\nFrame {idx}:")
        print(f"  Min pixel value : {sample.min():.2f}")
        print(f"  Max pixel value : {sample.max():.2f}")
        print(f"  Mean per channel: {sample.mean(axis=(0, 1))}")


if __name__ == "__main__":
    main()
