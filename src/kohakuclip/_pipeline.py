"""Implementation of the public KohakuClip Python API."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike as _PathLike
from pathlib import Path
from typing import Literal, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ._kclip import (
    load_numpy_frames_from_bytes as _rust_load_bytes,
    load_numpy_frames_from_path as _rust_load_path,
)

PathLike = Union[str, _PathLike[str], Path]
FrameArray = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class _VideoSource:
    kind: Literal["path", "bytes"]
    value: Union[str, bytes]

    @classmethod
    def normalize(
        cls,
        source: Union[PathLike, bytes, bytearray, memoryview, "_VideoSource"],
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
    kind: Literal["crop", "numpy_resize"]
    params: Tuple[int | None, ...]


_MISSING = object()


class KClip:
    """Stateless video loading pipeline."""

    __slots__ = ("_source", "_frame_range", "_resize", "_post_ops")

    def __init__(
        self,
        source: Union[PathLike, bytes, bytearray, memoryview, _VideoSource],
        *,
        frame_range: slice | None = None,
        resize: tuple[int, int] | None = None,
        _post_ops: Tuple[_PostOp, ...] | None = None,
    ) -> None:
        self._source = _VideoSource.normalize(source)
        self._frame_range = frame_range
        self._resize = resize
        self._post_ops = _post_ops or ()

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

    def _load_native(self) -> FrameArray:
        if self._source.kind == "path":
            return _rust_load_path(self._source.value, self._frame_range, self._resize)
        return _rust_load_bytes(self._source.value, self._frame_range, self._resize)

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
            else:  # pragma: no cover - defensive
                raise RuntimeError(f"Unknown post operation: {op.kind}")
        return result

    def _replace(
        self,
        *,
        frame_range: slice | None | object = _MISSING,
        resize: tuple[int, int] | None | object = _MISSING,
        post_ops: Tuple[_PostOp, ...] | None | object = _MISSING,
    ) -> "KClip":
        frame_range_value = self._frame_range if frame_range is _MISSING else frame_range
        resize_value = self._resize if resize is _MISSING else resize
        post_ops_value = self._post_ops if post_ops is _MISSING else post_ops
        return self.__class__(
            self._source,
            frame_range=frame_range_value,
            resize=resize_value,
            _post_ops=post_ops_value,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"KClip(source={self._source}, range={self._frame_range}, "
            f"resize={self._resize}, post_ops={self._post_ops})"
        )

    def __getstate__(
        self,
    ) -> Tuple[_VideoSource, slice | None, tuple[int, int] | None, Tuple[_PostOp, ...]]:
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
    "load_frames",
    "load_frames_from_bytes",
]
