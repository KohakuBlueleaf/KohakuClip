"""Which frames to read. All return frame indices (display order) of one video with n frames."""

import random


def clip(
    n: int, fps: float, frames: int, target_fps: float | None, rng: random.Random
) -> list[int]:
    """A contiguous clip of ``frames`` frames resampled to ``target_fps`` (None: every frame) at a
    random start; frames repeat when the video is shorter than the clip."""
    step = 1.0 if target_fps is None else fps / target_fps
    span = step * (frames - 1)
    start = rng.uniform(0, max(0.0, n - 1 - span))
    return [min(n - 1, round(start + k * step)) for k in range(frames)]


def random_frames(n: int, frames: int, rng: random.Random) -> list[int]:
    """``frames`` distinct frames drawn uniformly from the whole video, in display order."""
    return sorted(rng.sample(range(n), min(frames, n)))
