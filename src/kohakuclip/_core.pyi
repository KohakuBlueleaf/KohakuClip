"""Type stubs of the native module (src/kclip_rs)."""

from os import PathLike
from typing import Literal

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

class Pending:
    def result(self) -> Batch: ...

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
