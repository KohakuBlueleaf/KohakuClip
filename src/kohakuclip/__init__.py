"""KohakuClip: fast random-access video clips for training.

Storage: faststart mp4 (AV1 / H.264) in zip shards with an in-zip frame index (``writer``).
Reading: plan in Python, read + decode + resize/crop natively in parallel (``Reader``).
"""

from .reader import Augment, Batch, Reader
from .sampling import clip, random_frames
from .shard import Shard, Video
from ._native import profile

__all__ = ["Augment", "Batch", "Reader", "Shard", "Video", "clip", "random_frames", "profile"]
