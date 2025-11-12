"""Implementation of the public KohakuClip Python API."""

from dataclasses import dataclass
from os import PathLike as _PathLike
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray

from ._kclip import (
    load_numpy_frames_from_bytes as _rust_load_bytes,
    load_numpy_frames_from_path as _rust_load_path,
    load_video_metadata_from_bytes as _rust_meta_bytes,
    load_video_metadata_from_path as _rust_meta_path,
)

PathLike = str | _PathLike[str] | Path
FrameArray = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class _VideoSource:
    kind: Literal["path", "bytes"]
    value: str | bytes

    @classmethod
    def normalize(
        cls,
        source: "PathLike | bytes | bytearray | memoryview | _VideoSource",
    ) -> "_VideoSource":
        if isinstance(source, cls):
            return source
        if isinstance(source, (bytes, bytearray, memoryview)):
            return cls("bytes", bytes(source))
        if isinstance(source, (str, Path, _PathLike)):
            return cls("path", str(Path(source)))
        raise TypeError(
            "KClip source must be a filesystem path, bytes-like object, or another KClip source."
        )


@dataclass(frozen=True, slots=True)
class _PostOp:
    kind: Literal["crop", "numpy_resize", "callable"]
    params: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Lightweight container describing a video source."""

    file_size: int
    format: str
    width: int
    height: int
    frame_count: int | None
    bit_rate: int | None

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    def as_dict(self) -> dict[str, Any]:
        return {
            "file_size": self.file_size,
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "bit_rate": self.bit_rate,
            "resolution": self.resolution,
        }


_MISSING = object()


class KClip:
    """Stateless video loading pipeline."""

    __slots__ = ("_source", "_frame_range", "_resize", "_post_ops", "_metadata")

    def __init__(
        self,
        source: PathLike | bytes | bytearray | memoryview | _VideoSource,
        *,
        frame_range: slice | None = None,
        resize: tuple[int, int] | None = None,
        _post_ops: tuple[_PostOp, ...] | None = None,
        _metadata: VideoMetadata | None = None,
    ) -> None:
        self._source = _VideoSource.normalize(source)
        self._frame_range = frame_range
        self._resize = resize
        self._post_ops = _post_ops or ()
        self._metadata = _metadata

    def range(
        self,
        start: int | None = None,
        end: int | None = None,
        step: int = 1,
    ) -> "KClip":
        """Select a subset of frames using zero-based indexing."""

        start = _ensure_optional_non_negative(start, "start")
        end = _ensure_optional_non_negative(end, "end")
        step = _ensure_positive(step, "step")
        return self._replace(frame_range=slice(start, end, step))

    def resize(
        self,
        width: int,
        height: int,
        *,
        backend: Literal["auto", "native", "python", "rust"] = "auto",
    ) -> "KClip":
        """Request resized frames (width × height) before materializing them."""

        width = _ensure_positive(width, "width")
        height = _ensure_positive(height, "height")
        backend = backend.lower()
        if backend not in {"auto", "native", "rust", "python"}:
            raise ValueError("backend must be one of 'auto', 'native', 'rust', 'python'")

        if backend in {"auto", "native", "rust"}:
            return self._replace(resize=(width, height))

        # Python fallback (post-process)
        op = _PostOp("numpy_resize", (height, width))
        return self._replace(resize=None, post_ops=self._post_ops + (op,))

    def crop(
        self,
        *,
        top: int = 0,
        bottom: int | None = None,
        left: int = 0,
        right: int | None = None,
    ) -> "KClip":
        """Crop frames after loading."""

        top = _ensure_non_negative(top, "top")
        left = _ensure_non_negative(left, "left")
        bottom = _ensure_optional_non_negative(bottom, "bottom")
        right = _ensure_optional_non_negative(right, "right")
        op = _PostOp("crop", (top, bottom, left, right))
        return self._replace(post_ops=self._post_ops + (op,))

    def to_array(self) -> FrameArray:
        """Materialize the configured pipeline as a NumPy array."""

        frames = self._load_native()
        return self._apply_post(frames)

    def load(self) -> FrameArray:
        """Alias for :meth:`to_array`."""

        return self.to_array()

    def to_tensor(
        self,
        *,
        dtype: object | None = None,
        device: object | None = None,
        copy: bool = False,
    ):
        """Convert the frames into a PyTorch tensor."""

        frames = self.to_array()
        if copy:
            frames = np.ascontiguousarray(frames)

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("torch is required for KClip.to_tensor()") from exc

        tensor = torch.from_numpy(frames)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor

    def apply(self, func: Callable[[np.ndarray], np.ndarray]) -> "KClip":
        """Append a custom NumPy transformation executed after built-in post ops."""

        if not callable(func):
            raise TypeError("func must be callable")
        op = _PostOp("callable", (func,))
        return self._replace(post_ops=self._post_ops + (op,))

    def meta(self) -> VideoMetadata:
        """Return metadata for the underlying video source."""

        if self._metadata is None:
            raw = self._load_metadata()
            self._metadata = VideoMetadata(
                file_size=int(raw["file_size"]),
                format=str(raw["format"]),
                width=int(raw["width"]),
                height=int(raw["height"]),
                frame_count=(int(raw["frame_count"]) if raw["frame_count"] is not None else None),
                bit_rate=(int(raw["bit_rate"]) if raw["bit_rate"] is not None else None),
            )
        return self._metadata

    def _load_native(self) -> FrameArray:
        if self._source.kind == "path":
            return _rust_load_path(self._source.value, self._frame_range, self._resize)
        return _rust_load_bytes(self._source.value, self._frame_range, self._resize)

    def _load_metadata(self) -> dict[str, Any]:
        if self._source.kind == "path":
            return _rust_meta_path(self._source.value)
        return _rust_meta_bytes(self._source.value)

    def _apply_post(self, frames: FrameArray) -> FrameArray:
        if not self._post_ops:
            return frames

        result = frames
        for op in self._post_ops:
            if op.kind == "crop":
                top, bottom, left, right = op.params
                result = result[:, top:bottom, left:right, :]
            elif op.kind == "numpy_resize":
                height, width = op.params
                result = _resize_numpy(result, height, width)
            elif op.kind == "callable":
                (func,) = op.params
                produced = func(result)
                result = _ensure_frame_array(produced)
            else:  # pragma: no cover - defensive
                raise RuntimeError(f"Unknown post operation: {op.kind}")
        return result

    def _replace(
        self,
        *,
        frame_range: slice | None | object = _MISSING,
        resize: tuple[int, int] | None | object = _MISSING,
        post_ops: tuple[_PostOp, ...] | None | object = _MISSING,
        metadata: VideoMetadata | None | object = _MISSING,
    ) -> "KClip":
        frame_range_value = self._frame_range if frame_range is _MISSING else frame_range
        resize_value = self._resize if resize is _MISSING else resize
        post_ops_value = self._post_ops if post_ops is _MISSING else post_ops
        metadata_value = self._metadata if metadata is _MISSING else metadata
        return self.__class__(
            self._source,
            frame_range=frame_range_value,
            resize=resize_value,
            _post_ops=post_ops_value,
            _metadata=metadata_value,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"KClip(source={self._source}, range={self._frame_range}, "
            f"resize={self._resize}, post_ops={self._post_ops})"
        )

    def __getstate__(
        self,
    ) -> tuple[_VideoSource, slice | None, tuple[int, int] | None, tuple[_PostOp, ...]]:
        return (self._source, self._frame_range, self._resize, self._post_ops)

    def __setstate__(self, state) -> None:
        source, frame_range, resize, post_ops = state
        self.__init__(source, frame_range=frame_range, resize=resize, _post_ops=post_ops)


def load_frames(path: PathLike) -> FrameArray:
    """Load all RGB frames from a video file into a NumPy array."""

    return KClip(path).to_array()


def load_frames_from_bytes(data: bytes) -> FrameArray:
    """Load frames from raw video bytes."""

    return KClip(data).to_array()


def _ensure_frame_array(value: np.ndarray | Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 4:
        raise ValueError(
            "Custom post-processing functions must return an array shaped (frames, height, width, channels)"
        )
    return array


def _resize_numpy(frames: FrameArray, height: int, width: int) -> FrameArray:
    """Nearest-neighbor resize implemented in NumPy for post-processing."""

    if frames.shape[1] == height and frames.shape[2] == width:
        return frames

    if frames.shape[1] == 0 or frames.shape[2] == 0:
        raise ValueError("cannot resize an empty frame tensor")

    y_idx = np.linspace(0, frames.shape[1] - 1, height).round().astype(np.intp)
    x_idx = np.linspace(0, frames.shape[2] - 1, width).round().astype(np.intp)
    return frames[:, y_idx][:, :, x_idx]


def _ensure_positive(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return int(value)


def _ensure_non_negative(value: int, name: str) -> int:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return int(value)


def _ensure_optional_non_negative(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _ensure_non_negative(value, name)


__all__ = [
    "FrameArray",
    "PathLike",
    "KClip",
    "VideoMetadata",
    "load_frames",
    "load_frames_from_bytes",
]
