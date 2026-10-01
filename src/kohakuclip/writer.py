"""Write shards: re-encode videos to the storage format and pack them into zip / tar shards.

    kohakuclip-write out_dir videos/ more.tar clip.mp4 [--codec av1 --crf 36 --per-shard 1000]

Sources: video files, folders (every video file below them) and tar archives of videos (members
are read into memory, nothing is extracted). Encoding runs on the FFmpeg libraries KohakuClip is
built against (``backend="native"``, GIL released, one encode per thread), or on the ``ffmpeg``
command line (``backend="ffmpeg"``). Packing and indexing are native either way.

Storage format (defaults, see README for the measurements behind them): native fps, short side
capped at 512 (never upscaled), closed GOP of 16, AV1 (SVT-AV1 preset 6) with the in-loop filters
off for faster decoding, faststart mp4, zip (stored) or tar.
"""

import argparse
import json
import os
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

from . import _core

# SVT-AV1 prints its configuration per encode unless told otherwise (1: errors only)
os.environ.setdefault("SVT_LOG", "1")

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".ts"}


@dataclass
class Encoding:
    codec: str = "av1"  # "av1" (SVT-AV1), "h264" (x264), "hevc" (x265)
    crf: int = 36
    gop: int = 16
    preset: int = 6  # SVT-AV1 preset (x264 / x265: "medium")
    max_short: int = 512  # short side cap (never upscaled)
    loop_filters: bool = False  # AV1 deblocking / CDEF / loop restoration

    def ffmpeg_args(self) -> list[str]:
        """Encoder arguments for the ``ffmpeg`` command line (same settings as the native path)."""
        g = self.gop
        if self.codec == "av1":
            lf = int(self.loop_filters)
            params = (
                f"enable-dlf={lf}:enable-cdef={lf}:enable-restoration={lf}"
                ":scd=0:irefresh-type=2:lp=1"
            )
            return [
                "-c:v",
                "libsvtav1",
                "-preset",
                str(self.preset),
                "-crf",
                str(self.crf),
                "-g",
                str(g),
                "-svtav1-params",
                params,
            ]
        if self.codec == "h264":
            params = f"keyint={g}:min-keyint={g}:scenecut=0:bframes=0:threads=1"
            return [
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-tune",
                "fastdecode",
                "-crf",
                str(self.crf),
                "-x264-params",
                params,
            ]
        if self.codec == "hevc":
            params = (
                f"keyint={g}:min-keyint={g}:scenecut=0:bframes=0"
                ":frame-threads=1:pools=none"
            )
            return [
                "-c:v",
                "libx265",
                "-preset",
                "medium",
                "-crf",
                str(self.crf),
                "-x265-params",
                params,
            ]
        raise ValueError(f"unknown codec {self.codec!r}")


# ------------------------------------------------------------------ sources
@dataclass
class Video:
    """One input video: a path, or the bytes of a tar member."""

    name: str
    source: str | bytes  # a path, or the bytes of a video file

    @property
    def data(self) -> bytes | None:
        """The bytes to pipe into ffmpeg (None for a path)."""
        return self.source if isinstance(self.source, bytes) else None

    @property
    def url(self) -> str:
        """What ffmpeg opens: the path, or stdin."""
        return self.source if isinstance(self.source, str) else "pipe:0"


def expand(sources: list[str]) -> Iterator[Video]:
    """Video files, every video file below folders, and the video members of tar archives."""
    for source in sources:
        path = Path(source)

        if path.is_dir():
            files = sorted(
                p for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS
            )
            for p in files:
                yield Video(str(p.relative_to(path)), str(p))

        elif tarfile.is_tarfile(path):
            with tarfile.open(path) as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    if Path(member.name).suffix.lower() not in VIDEO_EXTENSIONS:
                        continue
                    file = tar.extractfile(member)
                    if file is not None:
                        yield Video(member.name, file.read())

        else:
            yield Video(path.name, str(path))


# ------------------------------------------------------------------ encoding
def encode_native(video: Video, enc: Encoding) -> bytes:
    return _core.encode(video.source, **asdict(enc))


def source_color(video: Video, ffprobe: str) -> list[str]:
    """ffmpeg arguments tagging an untagged source as the native path does: SD BT.601, HD BT.709."""
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,color_space",
        "-of",
        "json",
        video.url,
    ]
    probe = subprocess.run(cmd, input=video.data, check=True, capture_output=True)
    stream = json.loads(probe.stdout)["streams"][0]
    if stream.get("color_space", "unknown") not in ("unknown", "unspecified"):
        return []

    hd = min(stream["width"], stream["height"]) >= 720
    tag = "bt709" if hd else "smpte170m"
    return ["-colorspace", tag, "-color_primaries", tag, "-color_trc", tag]


def encode_ffmpeg(video: Video, enc: Encoding, ffmpeg: str = "ffmpeg") -> bytes:
    """The same encode on the ffmpeg command line (input bytes go through stdin)."""
    s = enc.max_short
    scale = (
        f"scale='if(lt(iw,ih),min(iw,{s}),-2)':'if(lt(iw,ih),-2,min(ih,{s}))'"
        ":flags=area"
    )
    ffprobe = str(Path(ffmpeg).with_name("ffprobe")) if os.sep in ffmpeg else "ffprobe"
    color = source_color(video, ffprobe)

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out.mp4")
        cmd = [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            video.url,
            "-vf",
            scale,
            "-pix_fmt",
            "yuv420p",
            "-an",
            *enc.ffmpeg_args(),
            "-threads",
            "1",
            "-color_range",
            "tv",
            *color,
            "-movflags",
            "+faststart",
            out,
        ]
        subprocess.run(cmd, input=video.data, check=True, capture_output=True)
        return Path(out).read_bytes()


def write(
    sources: list[str],
    out_dir: str,
    enc: Encoding | None = None,
    per_shard: int = 1000,
    workers: int = os.cpu_count() or 1,
    container: str = "zip",
    backend: str = "native",
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    """Encode ``sources`` with ``workers`` threads and pack them into
    ``out_dir/shard_XXXXX.{zip,tar}``. Videos that fail to encode are skipped (and reported).
    """
    enc = enc or Encoding()
    if backend == "native":
        encode = partial(encode_native, enc=enc)
    elif backend == "ffmpeg":
        encode = partial(encode_ffmpeg, enc=enc, ffmpeg=ffmpeg)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    def try_encode(video: Video) -> tuple[str, bytes | None]:
        try:
            return video.name, encode(video)
        except (RuntimeError, subprocess.CalledProcessError) as e:
            print(f"skipped {video.name}: {e}")
            return video.name, None

    os.makedirs(out_dir, exist_ok=True)
    shards: list[str] = []
    batch: list[Video] = []

    def flush(pool: ThreadPoolExecutor) -> None:
        encoded = [(name, data) for name, data in pool.map(try_encode, batch) if data]
        names = unique_names([name for name, _ in encoded])
        path = os.path.join(out_dir, f"shard_{len(shards):05d}.{container}")
        _core.pack(path, [(n, data) for n, (_, data) in zip(names, encoded)])
        shards.append(path)
        batch.clear()

    with ThreadPoolExecutor(workers) as pool:
        for video in expand(sources):
            batch.append(video)
            if len(batch) == per_shard:
                flush(pool)
        if batch:
            flush(pool)

    with open(os.path.join(out_dir, "encoding.json"), "w") as f:
        json.dump(asdict(enc), f, indent=2)
    return shards


def unique_names(names: list[str]) -> list[str]:
    """Member names: ``<stem>.mp4``, numbered when two videos share a stem."""
    seen: dict[str, int] = {}
    out = []
    for name in names:
        stem = str(Path(name).with_suffix(""))
        count = seen.get(stem, 0)
        seen[stem] = count + 1
        out.append(f"{stem}.mp4" if count == 0 else f"{stem}_{count}.mp4")
    return out


def parse_bool(text: str) -> bool:
    return text.lower() in ("1", "true", "yes")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("out_dir")
    ap.add_argument("sources", nargs="+")
    for field, default in asdict(Encoding()).items():
        kind = parse_bool if isinstance(default, bool) else type(default)
        ap.add_argument(f"--{field.replace('_', '-')}", type=kind, default=default)
    ap.add_argument("--per-shard", type=int, default=1000)
    ap.add_argument("--container", choices=("zip", "tar"), default="zip")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--backend", choices=("native", "ffmpeg"), default="native")
    ap.add_argument(
        "--ffmpeg", default="ffmpeg", help="ffmpeg binary for --backend ffmpeg"
    )
    a = vars(ap.parse_args())

    enc = Encoding(**{k: a[k] for k in asdict(Encoding())})
    shards = write(
        a["sources"],
        a["out_dir"],
        enc,
        per_shard=a["per_shard"],
        workers=a["workers"],
        container=a["container"],
        backend=a["backend"],
        ffmpeg=a["ffmpeg"],
    )
    for path in shards:
        print(path)


if __name__ == "__main__":
    main()
