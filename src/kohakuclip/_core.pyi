"""Type stubs of the native module (src/kclip_rs)."""

from os import PathLike
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

class VideoInfo:
    n: int
    fps: float
    h: int
    w: int
    codec: Literal["av1", "h264", "hevc"]
    colorspace: str
    source: str
    name: str

class Batch:
    video: npt.NDArray[np.uint8]
    window: npt.NDArray[np.uint32] | None
    flips: npt.NDArray[np.int8] | None
    colorspace: list[str] | None

class ImageInfo:
    h: int
    w: int
    codec: Literal["jpeg", "av1", "jxl", "webp", "png"]
    source: str
    index: int
    name: str

class ImageBatch:
    image: npt.NDArray[np.uint8]
    window: npt.NDArray[np.uint32] | None
    flips: npt.NDArray[np.int8] | None
    colorspace: list[str] | None

class Pending:
    def result(self) -> Any: ...  # Batch (Reader) or ImageBatch (ImageReader)

class Reader:
    def __init__(
        self,
        sources: list[str | PathLike[str]],
        size: int = 256,
        mode: Literal["rgb", "yuv", "yuv_resized"] = "rgb",
        threads: int = 1,
        crop: Literal["random", "center"] = "random",
        hflip: float = 0.0,
        vflip: float = 0.0,
        seed: int | None = None,
        skip: bool = True,
        use_index: bool = True,
        open_files: int = 4096,
        moov_cache: int = 4096,
    ) -> None: ...
    @property
    def size(self) -> int: ...
    @property
    def mode(self) -> Literal["rgb", "yuv", "yuv_resized"]: ...
    def __len__(self) -> int: ...
    def info(self, vid: int) -> VideoInfo: ...
    def sample(
        self,
        n: int,
        frames: int,
        fps: float | None = None,
        spread: bool = False,
        seed: int | None = None,
        videos: list[int] | None = None,
    ) -> list[tuple[int, list[int]]]: ...
    def submit(
        self,
        items: list[tuple[int, list[int]]],
        out: npt.NDArray[np.uint8] | None = None,
        seed: int | None = None,
    ) -> Pending: ...
    def read(
        self,
        items: list[tuple[int, list[int]]],
        out: npt.NDArray[np.uint8] | None = None,
        seed: int | None = None,
    ) -> Batch: ...

class ImageReader:
    def __init__(
        self,
        sources: list[str | PathLike[str]],
        size: int = 256,
        mode: Literal["rgb", "yuv", "yuv_resized"] = "rgb",
        threads: int = 1,
        crop: Literal["random", "center", "resized"] = "random",
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (0.75, 4 / 3),
        hflip: float = 0.0,
        vflip: float = 0.0,
        seed: int | None = None,
        dct_scale: bool = True,
        readahead: bool = True,
        use_index: bool = True,
        open_files: int = 4096,
    ) -> None: ...
    @property
    def size(self) -> int: ...
    @property
    def mode(self) -> Literal["rgb", "yuv", "yuv_resized"]: ...
    def __len__(self) -> int: ...
    def info(self, image: int) -> ImageInfo: ...
    def sample(self, n: int, seed: int | None = None) -> list[int]: ...
    def submit(
        self,
        images: list[int],
        out: npt.NDArray[np.uint8] | None = None,
        seed: int | None = None,
    ) -> Pending: ...
    def read(
        self,
        images: list[int],
        out: npt.NDArray[np.uint8] | None = None,
        seed: int | None = None,
    ) -> ImageBatch: ...

def permute(n: int, seed: int, positions: list[int]) -> list[int]: ...
def profile(reset: bool = False) -> dict[str, float]: ...
def encode(
    source: str | PathLike[str] | bytes,
    codec: str = "av1",
    crf: int = 36,
    gop: int = 16,
    preset: int = 6,
    max_short: int = 512,
    loop_filters: bool = False,
) -> bytes: ...
def pack(path: str | PathLike[str], members: list[tuple[str, bytes]]) -> None: ...
def encode_image(
    source: str | PathLike[str] | bytes,
    codec: Literal["jpeg", "av1", "jxl", "webp"] = "jpeg",
    quality: int = 70,
    chroma: Literal["420", "444"] = "444",
    crf: int = 30,
    distance: float = 1.0,
    effort: int | None = None,
    max_short: int = 512,
    min_short: int = 384,
    dct_scale: bool = False,
    jpeg_encoder: Literal["mozjpeg", "turbo"] = "mozjpeg",
) -> tuple[bytes | None, int, int, int, int]: ...
def pack_images(
    path: str | PathLike[str],
    members: list[tuple[str, bytes, int, int, str]],
    header: str = "{}",
) -> None: ...
