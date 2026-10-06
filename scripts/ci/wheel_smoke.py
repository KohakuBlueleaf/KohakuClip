"""Smoke test of an installed wheel without any system FFmpeg or libjpeg-turbo: encode a
synthetic video and a synthetic image with the bundled libraries, pack shards, read clips and
images back."""

import os
import tempfile

import numpy as np

import kohakuclip
from kohakuclip import ImageReader, Reader, _core


def synthetic_y4m(path: str, w: int = 320, h: int = 240, frames: int = 48) -> None:
    """A moving gradient as raw YUV4MPEG2 (readable by FFmpeg without any decoder library)."""
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F24:1 Ip A1:1 C420jpeg\n".encode())
        yy, xx = np.mgrid[0:h, 0:w]
        for t in range(frames):
            y = ((xx + yy + 4 * t) % 256).astype(np.uint8)
            u = np.full((h // 2, w // 2), 128, np.uint8)
            f.write(b"FRAME\n" + y.tobytes() + u.tobytes() + u.tobytes())


def synthetic_bmp(w: int = 600, h: int = 400) -> bytes:
    """A color gradient as a 24-bit BMP (decoded by FFmpeg itself, found from its magic)."""
    yy, xx = np.mgrid[0:h, 0:w]
    bgr = np.stack([(xx + yy) % 256, yy * 255 // h, xx * 255 // w], -1).astype(np.uint8)
    row = (3 * w + 3) // 4 * 4  # rows padded to 4 bytes, stored bottom-up
    pixels = np.zeros((h, row), np.uint8)
    pixels[:, : 3 * w] = bgr[::-1].reshape(h, 3 * w)
    header = b"BM" + (54 + pixels.size).to_bytes(4, "little") + bytes(4)
    header += (54).to_bytes(4, "little") + (40).to_bytes(4, "little")
    header += w.to_bytes(4, "little") + h.to_bytes(4, "little")
    header += (1).to_bytes(2, "little") + (24).to_bytes(2, "little") + bytes(24)
    return header + pixels.tobytes()


def image_round_trip(tmp: str) -> None:
    """BMP -> stored JPEG (mozjpeg, built in; libjpeg-turbo, bundled) -> image shard -> a
    128 x 128 crop."""
    data, w, h, _, _ = _core.encode_image(synthetic_bmp())
    assert data is not None and (w, h) == (600, 400), (w, h)
    turbo, *_ = _core.encode_image(synthetic_bmp(), jpeg_encoder="turbo")
    assert turbo is not None and turbo != data
    shard = os.path.join(tmp, "images.zip")
    _core.pack_images(shard, [("a.jpg", data, w, h, "jpeg")])
    image = ImageReader([shard], size=128, crop="center").read([0]).image
    assert image.shape == (1, 3, 128, 128), image.shape
    assert image.std() > 5, "decoded image is flat"
    print("ok", image.shape)


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
        image_round_trip(tmp)


if __name__ == "__main__":
    main()
