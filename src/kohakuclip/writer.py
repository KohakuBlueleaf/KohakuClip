"""Write shards: encode videos to the storage format, pack them into zip shards with an index.

    python -m kohakuclip.writer out_dir video1.mp4 video2.mp4 ... [--codec av1 --crf 36 --per-shard 1000]

Storage format (defaults, see README for the measurements behind them): native fps, short side
capped at 512 (never upscaled), closed GOP of 16, AV1 (SVT-AV1 preset 6) with the in-loop filters
(deblocking, CDEF, loop restoration) off for faster decoding, faststart mp4, zip (stored) or tar.
Encoding uses the ffmpeg CLI (with libsvtav1 / libx264); reading back uses PyAV (write time only).
"""

import argparse
import io
import json
import os
import struct
import subprocess
import tarfile
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from .shard import FRAME, INDEX, KEY, MAGIC, codec_prefix, data_offset


@dataclass
class Encoding:
    codec: str = "av1"            # "av1" (SVT-AV1) | "h264" (x264, tune=fastdecode)
    crf: int = 36
    gop: int = 16
    preset: int = 6               # SVT-AV1 preset (h264: x264 preset "medium")
    max_short: int = 512          # short side cap (never upscaled)
    loop_filters: bool = False    # AV1 deblocking / CDEF / loop restoration
    ffmpeg: str = "ffmpeg"

    def args(self) -> list[str]:
        if self.codec == "av1":
            lf = int(self.loop_filters)
            params = f"enable-dlf={lf}:enable-cdef={lf}:enable-restoration={lf}:scd=0:irefresh-type=2:lp=1"
            return ["-c:v", "libsvtav1", "-preset", str(self.preset), "-crf", str(self.crf), "-g", str(self.gop),
                    "-svtav1-params", params]
        if self.codec == "h264":
            x264 = f"keyint={self.gop}:min-keyint={self.gop}:scenecut=0:bframes=0:threads=1"
            return ["-c:v", "libx264", "-preset", "medium", "-tune", "fastdecode", "-crf", str(self.crf),
                    "-x264-params", x264]
        raise ValueError(f"unknown codec {self.codec}")


def encode(src: str, dst: str, enc: Encoding) -> None:
    """One video -> faststart mp4 at native fps, short side <= enc.max_short, closed GOPs."""
    s = enc.max_short
    scale = f"scale='if(lt(iw,ih),min(iw,{s}),-2)':'if(lt(iw,ih),-2,min(ih,{s}))':flags=area"
    cmd = [enc.ffmpeg, "-v", "error", "-y", "-i", src, "-vf", scale, "-pix_fmt", "yuv420p", "-an",
           *enc.args(), "-threads", "1", "-color_range", "tv", "-colorspace", "bt709",
           "-movflags", "+faststart", dst]
    subprocess.run(cmd, check=True, capture_output=True)


def frame_table(path: str) -> tuple[list, dict]:
    """(offset in file, size, keep bytes, flags) per frame and the video metadata, read with PyAV."""
    import av

    with av.open(path) as c:
        s = c.streams.video[0]
        rows = [(p.pos, p.size, p.size, KEY if p.is_keyframe else 0) for p in c.demux(s) if p.size]
        ctx = s.codec_context
        codec = {"av1": "av1", "libdav1d": "av1", "h264": "h264", "hevc": "hevc"}[ctx.name]
        prefix = codec_prefix(codec, bytes(ctx.extradata or b"")).hex()
        meta = dict(n=len(rows), fps=float(s.average_rate or s.guessed_rate), h=ctx.height, w=ctx.width,
                    codec=codec, prefix=prefix, colorspace="bt709")
    if codec == "av1":  # bytes later frames depend on (see kohakuclip.av1)
        from .av1 import keep_bytes

        with open(path, "rb") as f:
            samples = [(f.seek(off), f.read(size))[1] for off, size, _, _ in rows]
        keep = keep_bytes(bytes.fromhex(prefix), samples)
        rows = [(off, size, k, flags) for (off, size, _, flags), k in zip(rows, keep)]
    return rows, meta


def pack(mp4s: list[str], path: str) -> None:
    """Archive the mp4 files (``.tar``, or zip in stored mode), then append the index member with the
    absolute offset of every frame in the archive."""
    names = [os.path.basename(p) for p in mp4s]
    if path.endswith(".tar"):
        with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as t:
            for p, n in zip(mp4s, names):
                t.add(p, n)
        with tarfile.open(path) as t:
            starts = [t.getmember(n).offset_data for n in names]
    else:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
            for p, n in zip(mp4s, names):
                z.write(p, n)
        with zipfile.ZipFile(path) as z, open(path, "rb") as f:
            starts = [data_offset(f, z.getinfo(n).header_offset) for n in names]
    videos, records = [], []
    for p, n, base in zip(mp4s, names, starts):
        rows, meta = frame_table(p)
        videos.append(dict(member=n, row=len(records), **meta))
        records += [(base + off, size, keep, flags, (0,) * 7) for off, size, keep, flags in rows]
    meta = json.dumps(dict(videos=videos)).encode()
    head = MAGIC + struct.pack("<Q", len(meta)) + meta
    head += b"\0" * (-len(head) % 8)
    blob = head + np.array(records, FRAME).tobytes()
    if path.endswith(".tar"):
        with tarfile.open(path, "a", format=tarfile.PAX_FORMAT) as t:
            info = tarfile.TarInfo(INDEX)
            info.size = len(blob)
            t.addfile(info, io.BytesIO(blob))
    else:
        with zipfile.ZipFile(path, "a", zipfile.ZIP_STORED) as z:
            z.writestr(zipfile.ZipInfo(INDEX), blob)


def _encode_one(args):
    src, dst, enc = args
    try:
        encode(src, dst, enc)
        return dst
    except subprocess.CalledProcessError:
        return None


def write(sources: list[str], out_dir: str, enc: Encoding = Encoding(), per_shard: int = 1000,
          workers: int = os.cpu_count() or 1, container: str = "zip") -> list[str]:
    """Encode ``sources`` in parallel and pack them into ``out_dir/shard_XXXXX.{zip,tar}``."""
    os.makedirs(out_dir, exist_ok=True)
    shards = []
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp, ProcessPoolExecutor(workers) as pool:
        for k in range(0, len(sources), per_shard):
            jobs = [(src, os.path.join(tmp, f"{k + i:08d}.mp4"), enc) for i, src in enumerate(sources[k:k + per_shard])]
            done = [p for p in pool.map(_encode_one, jobs) if p]
            path = os.path.join(out_dir, f"shard_{k // per_shard:05d}.{container}")
            pack(done, path)
            shards.append(path)
            for p in done:
                os.remove(p)
    with open(os.path.join(out_dir, "encoding.json"), "w") as f:
        json.dump(asdict(enc), f, indent=2)
    return shards


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("sources", nargs="+")
    for field, default in asdict(Encoding()).items():
        kind = (lambda v: v.lower() in ("1", "true", "yes")) if isinstance(default, bool) else type(default)
        ap.add_argument(f"--{field.replace('_', '-')}", type=kind, default=default)
    ap.add_argument("--per-shard", type=int, default=1000)
    ap.add_argument("--container", choices=("zip", "tar"), default="zip")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = vars(ap.parse_args())
    enc = Encoding(**{k: a[k] for k in asdict(Encoding())})
    for path in write(a["sources"], a["out_dir"], enc, a["per_shard"], a["workers"], a["container"]):
        print(path)


if __name__ == "__main__":
    main()
