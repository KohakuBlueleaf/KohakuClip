#!/usr/bin/env python3
"""
Demonstrate range-limited loading and native resizing with KClip.

Usage:
    python examples/range_and_resize.py video.mp4 --start 0 --end 120 --step 3 --size 224 224
"""
import argparse
from pathlib import Path

import numpy as np

from kohakuclip import KClip


def save_video(frames: np.ndarray, path: Path, fps: int) -> None:
    if frames.size == 0:
        raise SystemExit("No frames available to write. Adjust your range settings.")

    try:
        import av
    except ImportError as exc:  # pragma: no cover - PyAV is optional
        raise SystemExit("PyAV is required to save MP4 files. Install `av` and retry.") from exc

    height, width = frames.shape[1], frames.shape[2]
    if height % 2 != 0 or width % 2 != 0:
        even_height = height - (height % 2)
        even_width = width - (width % 2)
        frames = frames[:, :even_height, :even_width, :]
        height, width = even_height, even_width

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = int(width)
        stream.height = int(height)
        stream.pix_fmt = "yuv420p"

        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Range + resize demo for KohakuClip.")
    parser.add_argument("video", type=Path, help="Path to the video to sample.")
    parser.add_argument("--start", type=int, default=0, help="First frame to include.")
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Exclusive end frame. Default: read to the end.",
    )
    parser.add_argument("--step", type=int, default=2, help="Frame stride.")
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(224, 224),
        help="Resize frames to WIDTH HEIGHT via native ffmpeg scaling.",
    )
    parser.add_argument(
        "--tensor",
        action="store_true",
        help="Convert the result to a PyTorch tensor for inspection.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional MP4 path to save the resulting frames (requires PyAV).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frame rate to use when writing the MP4 output (default: 24).",
    )
    return parser.parse_args()


def describe(arr: np.ndarray) -> None:
    print(f"Frames shape: {arr.shape}")
    print(f"Dtype       : {arr.dtype}")
    print(f"Range       : [{arr.min()}, {arr.max()}]")


def main() -> None:
    args = parse_args()
    width, height = args.size

    clip = KClip(args.video).range(args.start, args.end, args.step).resize(width, height)
    frames = clip.to_array()
    describe(frames)

    if args.tensor:
        tensor = clip.to_tensor(dtype=None)
        print(f"\nTensor shape: {tuple(tensor.shape)}  dtype: {tensor.dtype}")
        print(f"Tensor device: {tensor.device}  contiguous: {tensor.is_contiguous()}")

    if args.output:
        save_video(frames, args.output, fps=args.fps)
        print(f"Wrote {frames.shape[0]} frames to {args.output}")


if __name__ == "__main__":
    main()
