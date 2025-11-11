#!/usr/bin/env python3
"""
Save a single frame from a video to disk using Pillow.

Usage:
    python examples/save_first_frame.py input.mp4 output.png --frame 0

Requires Pillow (pip install pillow).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kohakuclip import KClip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a frame as an image file.")
    parser.add_argument("video", type=Path, help="Input video file.")
    parser.add_argument("output", type=Path, help="Destination image (e.g. frame.png).")
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Zero-based frame index to export (defaults to the first frame).",
    )
    return parser.parse_args()


def main() -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "This example requires Pillow. Install it via 'pip install pillow' and retry."
        ) from exc

    args = parse_args()
    start = max(0, args.frame)
    clip = KClip(args.video).range(start, start + 1)
    frames = clip.to_array()

    if frames.size == 0:
        raise SystemExit("The requested frame was not available.")

    image = Image.fromarray(frames[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Saved frame {idx} to {args.output}")


if __name__ == "__main__":
    main()
