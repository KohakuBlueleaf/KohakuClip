from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

FrameArray = NDArray[np.uint8]

class VideoMetadataDict(TypedDict):
    file_size: int
    format: str
    width: int
    height: int
    frame_count: int | None
    bit_rate: int | None

def load_numpy_frames_from_path(
    path: str,
    frame_range: slice | None = ...,
    resize: tuple[int, int] | None = ...,
) -> FrameArray: ...
def load_numpy_frames_from_bytes(
    data: bytes,
    frame_range: slice | None = ...,
    resize: tuple[int, int] | None = ...,
) -> FrameArray: ...
def load_video_metadata_from_path(path: str) -> VideoMetadataDict: ...
def load_video_metadata_from_bytes(data: bytes) -> VideoMetadataDict: ...
