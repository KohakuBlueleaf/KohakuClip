"""KohakuClip: fast random-access video clips for training.

Storage: mp4 files (AV1 / H.264 / HEVC) in zip or tar shards with an in-archive frame index
(``kohakuclip.writer``), or plain folders of mp4 files.
Reading: ``Reader`` plans each batch and decodes it natively on a persistent thread pool (pread
per GOP, libavcodec / libdav1d, antialiased resize fused with the crop), without the GIL.
"""

from ._core import Batch, Pending, Reader, VideoInfo, profile
from .sampling import clip, random_frames

__all__ = [
    "Batch",
    "Pending",
    "Reader",
    "VideoInfo",
    "clip",
    "profile",
    "random_frames",
]
