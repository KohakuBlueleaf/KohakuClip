"""Smoke test of an installed wheel without any system FFmpeg: encode a synthetic video with
the bundled libraries, pack a shard, read clips back."""

import os
import tempfile

import numpy as np

import kohakuclip
from kohakuclip import Reader, _core


def synthetic_y4m(path: str, w: int = 320, h: int = 240, frames: int = 48) -> None:
    """A moving gradient as raw YUV4MPEG2 (readable by FFmpeg without any decoder library)."""
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F24:1 Ip A1:1 C420jpeg\n".encode())
        yy, xx = np.mgrid[0:h, 0:w]
        for t in range(frames):
            y = ((xx + yy + 4 * t) % 256).astype(np.uint8)
            u = np.full((h // 2, w // 2), 128, np.uint8)
            f.write(b"FRAME\n" + y.tobytes() + u.tobytes() + u.tobytes())


def main() -> None:
    print("kohakuclip", kohakuclip.__file__)
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src.y4m")
        synthetic_y4m(src)
        mp4 = _core.encode(src, crf=40, preset=12)
        shard = os.path.join(tmp, "shard.zip")
        _core.pack(shard, [("a.mp4", mp4)])
        reader = Reader([shard], size=128, threads=2, crop="center")
        video = reader.read([(0, [0, 10, 20, 30])]).video
        assert video.shape == (1, 4, 3, 128, 128), video.shape
        assert video.std() > 5, "decoded frames are flat"
    print("ok", video.shape)


if __name__ == "__main__":
    main()
