#!/usr/bin/env python3
"""
Demonstrate that kohakuclip produces identical results when loading from bytes or a file path.

Usage:
    python examples/compare_path_vs_bytes.py clip.mp4
"""

import argparse
from pathlib import Path

import numpy as np

from kohakuclip import KClip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare path vs. bytes loading modes.")
    parser.add_argument("video", type=Path, help="Video to load.")
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
        help="Resize frames prior to comparison.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output (still exits non-zero on mismatch).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video

    clip = KClip(video_path)
    if args.start is not None or args.end is not None or args.step != 1:
        clip = clip.range(args.start, args.end, args.step)
    if args.resize:
        width, height = args.resize
        clip = clip.resize(width, height)

    if not args.quiet:
        print(f"Reading frames from path: {video_path}")
    frames_from_path = clip.to_array()

    if not args.quiet:
        print("Reading frames from bytes (memory-only)...")
    raw = video_path.read_bytes()
    clip_bytes = KClip(raw)
    if args.start is not None or args.end is not None or args.step != 1:
        clip_bytes = clip_bytes.range(args.start, args.end, args.step)
    if args.resize:
        width, height = args.resize
        clip_bytes = clip_bytes.resize(width, height)
    frames_from_bytes = clip_bytes.to_array()

    identical = np.array_equal(frames_from_path, frames_from_bytes)

    if identical:
        if not args.quiet:
            print("✅ Frames are identical in both modes.")
    else:
        diff = np.abs(frames_from_path.astype(np.int16) - frames_from_bytes.astype(np.int16))
        max_diff = diff.max()
        raise SystemExit(f"❌ Frame buffers differ! Maximum absolute difference: {max_diff}")


if __name__ == "__main__":
    main()
