"""Reusable PyTorch dataset utilities built on top of KohakuClip."""

import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from kohakuclip import KClip, VideoMetadata

__all__ = ["FolderVideoDataset", "random_truncate", "save_video"]


class FolderVideoDataset(Dataset):
    """Dataset that lazily loads MP4 files via KohakuClip."""

    def __init__(
        self,
        folder_path: Path | str,
        *,
        resize: tuple[int, int] | None = (256, 256),
        target_length: int | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        cache_name: str = "dataset.npy",
        max_attempts: int = 8,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"{self.folder_path} does not exist")

        self.resize = resize
        self.target_length = target_length
        self.transform = transform
        self.cache_name = cache_name
        self.cache_path = self.folder_path / self.cache_name
        self.max_attempts = max_attempts
        self.dtype = dtype
        self._rng = random.Random(seed)
        self._files: np.ndarray | None = None
        self._file_count = self._ensure_file_cache()
        if self._file_count == 0:
            raise ValueError(f"No MP4 files found under {self.folder_path}")

    def _ensure_file_cache(self) -> int:
        if self.cache_path.exists():
            files = np.load(self.cache_path, allow_pickle=True)
            count = int(files.size)
            del files
            return count

        files = np.array(
            sorted(str(path) for path in self.folder_path.rglob("*.mp4") if path.is_file()),
            dtype=object,
        )
        np.save(self.cache_path, files)
        count = int(files.size)
        del files
        return count

    def __len__(self) -> int:
        return self._file_count

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._file_count == 0:
            raise RuntimeError("Dataset is empty")

        attempts = 0
        index = idx % self._file_count
        while attempts < self.max_attempts:
            file_path = Path(self.files[index])
            try:
                return self._load_clip(file_path)
            except Exception:
                attempts += 1
                index = self._rng.randint(0, self._file_count - 1)
        raise RuntimeError(f"Failed to load video after {self.max_attempts} attempts")

    def metadata(self, idx: int) -> VideoMetadata:
        file_path = Path(self.files[idx % self._file_count])
        return self._metadata_for(file_path)

    def _load_clip(self, file_path: Path) -> torch.Tensor:
        clip = KClip(file_path)
        if self.resize:
            clip = clip.resize(*self.resize)
        metadata = self._metadata_for(file_path, clip)
        frame_count = metadata.frame_count

        if self.target_length is not None:
            frame_count = frame_count or self._probe_frame_count(file_path)
            if frame_count is None or frame_count < self.target_length:
                raise ValueError(
                    f"Video {file_path} shorter ({frame_count}) than target length ({self.target_length})"
                )
            start = self._rng.randint(0, frame_count - self.target_length)
            clip = clip.range(start, start + self.target_length)

        frames = clip.to_array()
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        tensor = tensor.to(self.dtype) / 255.0
        if self.transform:
            tensor = self.transform(tensor)
        return tensor

    def _metadata_for(self, file_path: Path, clip: KClip | None = None) -> VideoMetadata:
        clip = clip or KClip(file_path)
        meta = clip.meta()
        return meta

    def _probe_frame_count(self, file_path: Path) -> int | None:
        clip = KClip(file_path)
        frames = clip.to_array()
        return frames.shape[0]

    @property
    def files(self) -> np.ndarray:
        if self._files is None:
            self._files = np.load(self.cache_path, allow_pickle=True)
        return self._files


def random_truncate(length: int):
    def inner(video: torch.Tensor) -> torch.Tensor:
        t, _, _, _ = video.shape
        if t <= length:
            return video
        start = torch.randint(0, t - length + 1, (1,)).item()
        return video[start : start + length]

    return inner


def save_video(video: torch.Tensor, path: str, fps: int = 30) -> None:
    """Persist a tensor shaped [T, 3, H, W] in the range [-1, 1] to disk."""

    import av

    assert video.ndim == 4 and video.shape[1] == 3, "expected [T, 3, H, W]"

    video = (video.clamp(-1, 1) * 127.5 + 127.5).byte()
    video = video.permute(0, 2, 3, 1).cpu().numpy()

    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = video.shape[2]
    stream.height = video.shape[1]
    stream.pix_fmt = "yuv420p"

    for frame in video:
        frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()
