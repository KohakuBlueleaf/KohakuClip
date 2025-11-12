from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
av = pytest.importorskip("av")
import torch.nn.functional as F

from kohakuclip import KClip, VideoMetadata, load_frames, load_frames_from_bytes


@dataclass
class SyntheticClip:
    path: Path
    data: bytes
    frames: np.ndarray


@pytest.fixture(scope="module")
def synthetic_clip(tmp_path_factory: pytest.TempPathFactory) -> SyntheticClip:
    tmp_dir = tmp_path_factory.mktemp("clips")
    path = tmp_dir / "synthetic.mp4"
    frames = _generate_frames(num_frames=16, size=64, square=18)
    _encode_video(frames, path)
    data = path.read_bytes()
    return SyntheticClip(path=path, data=data, frames=frames)


def test_eager_helpers_round_trip(synthetic_clip: SyntheticClip) -> None:
    frames_path = load_frames(synthetic_clip.path)
    frames_bytes = load_frames_from_bytes(synthetic_clip.data)

    assert frames_path.shape == frames_bytes.shape
    np.testing.assert_allclose(frames_path, frames_bytes, atol=1)


def test_range_and_native_resize_match_numpy(synthetic_clip: SyntheticClip) -> None:
    clip = KClip(synthetic_clip.path).range(2, 10, 2).resize(32, 24)
    ranged_resized = clip.to_array()

    baseline = load_frames(synthetic_clip.path)[2:10:2]
    baseline_t = torch.from_numpy(baseline).permute(0, 3, 1, 2).float()
    resized = (
        F.interpolate(baseline_t, size=(24, 32), mode="bilinear", align_corners=False)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .numpy()
    )

    assert ranged_resized.shape == resized.shape

    diff = np.abs(ranged_resized.astype(np.int16) - resized.astype(np.int16))
    # Native FFmpeg scaling and PyTorch's bilinear implementation differ slightly.
    # Most pixels match exactly, so enforce tight average error while allowing rare spikes.
    assert diff.mean() <= 5
    assert diff.max() <= 80


def test_to_tensor_matches_numpy(synthetic_clip: SyntheticClip) -> None:
    clip = KClip(synthetic_clip.path).range(0, 5)
    frames = clip.to_array()
    tensor = clip.to_tensor()

    assert tensor.shape == (frames.shape[0], frames.shape[1], frames.shape[2], 3)
    torch.testing.assert_close(tensor.cpu(), torch.from_numpy(frames), atol=0, rtol=0)


def test_crop_and_python_resize(synthetic_clip: SyntheticClip) -> None:
    clip = (
        KClip(synthetic_clip.path)
        .crop(top=4, left=8, bottom=40, right=52)
        .resize(24, 24, backend="python")
    )
    frames = clip.to_array()
    assert frames.shape[1:3] == (24, 24)
    assert frames.shape[0] > 0


def test_metadata_for_path_and_bytes(synthetic_clip: SyntheticClip) -> None:
    clip = KClip(synthetic_clip.path)
    meta = clip.meta()

    assert isinstance(meta, VideoMetadata)
    assert meta.width == synthetic_clip.frames.shape[2]
    assert meta.height == synthetic_clip.frames.shape[1]
    assert meta.frame_count == synthetic_clip.frames.shape[0]
    assert meta.resolution == (meta.width, meta.height)
    assert clip.meta() is meta  # cached object

    clip_bytes = KClip(synthetic_clip.data)
    meta_bytes = clip_bytes.meta()

    assert meta_bytes.frame_count == meta.frame_count
    assert meta_bytes.width == meta.width
    assert meta_bytes.height == meta.height


def _generate_frames(num_frames: int, size: int, square: int) -> np.ndarray:
    frames = torch.zeros(num_frames, size, size, 3, dtype=torch.uint8)
    colors = torch.linspace(50, 220, num_frames, dtype=torch.uint8)

    for idx in range(num_frames):
        offset = (idx * 2) % (size - square)
        frame = frames[idx]
        frame[:, :, 0] = 20  # background tint
        frame[:, :, 1] = 10
        frame[offset : offset + square, offset : offset + square] = torch.tensor(
            [colors[idx], 255 - colors[idx], 120], dtype=torch.uint8
        )

    return frames.numpy()


def _encode_video(frames: np.ndarray, path: Path) -> None:
    with av.open(path, mode="w") as container:
        stream = container.add_stream("mpeg4", rate=24)
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = "yuv420p"

        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
