"""KohakuClip: fast random-access video clips and images for training.

Storage: mp4 files (AV1 / H.264 / HEVC) in zip or tar shards with an in-archive frame index
(``kohakuclip.writer``), or plain folders of mp4 files; images (JPEG by default; AV1, JPEG XL,
WebP) in zip shards with a fixed-size per-image index, or plain archives and folders of images.
Reading: ``Reader`` / ``ImageReader`` plan each batch and decode it natively on a persistent
thread pool (one pread per GOP or image, libavcodec / libdav1d / libjpeg-turbo, antialiased
resize fused with the crop), without the GIL.
"""

from ._core import (
    Batch,
    ImageBatch,
    ImageInfo,
    ImageReader,
    Pending,
    Reader,
    VideoInfo,
    permute,
    profile,
)
from .sampling import clip, random_frames

__all__ = [
    "Batch",
    "ImageBatch",
    "ImageInfo",
    "ImageReader",
    "Pending",
    "Reader",
    "VideoInfo",
    "clip",
    "permute",
    "profile",
    "random_frames",
]
