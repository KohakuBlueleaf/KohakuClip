"""Write image shards: encode images to the storage format and pack them into zip shards.

    kohakuclip-write-images out_dir images/ data.tar photo.jpg [--quality 70 --chroma 444]

Sources: image files, folders (every image below them, with ``<stem>.txt`` captions and
``<stem>.json`` metadata next to them) and tar archives (WebDataset layout: members grouped by
key, the image member plus optional ``.txt`` / ``.json``; members are read into memory, nothing
is extracted). Each image is decoded (JPEG through libjpeg-turbo, anything else through FFmpeg),
its short side capped at ``max_short`` (Lanczos-3, never upscaled; images whose short side is
under ``min_short`` are dropped, ``min_short=0`` keeps all), turned upright (EXIF orientation),
and encoded, natively without the GIL, one image per thread (``workers``). Images the native
path cannot read (e.g. CMYK JPEGs) are decoded by Pillow and encoded natively (``fallback``);
``backend="pillow"`` uses Pillow for all (libjpeg-turbo, not mozjpeg).

Output: ``out_dir/shard_XXXXX.zip`` (stored members ``<key>.<ext>`` and the image index),
``manifest.parquet`` (``manifest.jsonl`` without pyarrow: one row per stored image with its
shard, index in the shard, key, stored and source sizes, ``native_res`` (stored at the source
size), caption, metadata), ``skipped.jsonl`` (dropped and failed inputs) and ``encoding.json``.
Sources may also be ``ImageSource`` objects (bytes or a path, with caption and metadata).

Storage format (defaults, see README for the measurements behind them): JPEG through mozjpeg
(trellis quantization, baseline scans) quality 70, 4:4:4: 41-42 dB PSNR on 256 x 256 random
resized crops at ~43 KB per image; short side 512 at most, 384 at least.
"""

import argparse
import io
import json
import os
import tarfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from . import _core

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".avif",
    ".jxl",
}
SIDECARS = {".txt", ".json"}
STORED_EXTENSION = {"jpeg": "jpg", "av1": "obu", "jxl": "jxl", "webp": "webp"}


@dataclass
class ImageEncoding:
    """How images are stored (the keyword arguments of ``_core.encode_image``)."""

    # "jpeg", "av1" (libaom, intra), "jxl", "webp"
    codec: str = "jpeg"
    # JPEG / WebP quality
    quality: int = 70
    # JPEG / AV1 chroma subsampling: "444" or "420"
    chroma: str = "444"
    # AV1 constant quality (0-63)
    crf: int = 30
    # JPEG XL distance
    distance: float = 1.0
    # AV1 cpu-used / JPEG XL effort / WebP method; None: the codec's default
    effort: int | None = None
    # short side cap (never upscaled)
    max_short: int = 512
    # images with a shorter short side are dropped
    min_short: int = 384
    # shrink JPEG inputs in the IDCT when the stored size allows
    dct_scale: bool = False
    # "mozjpeg" or "turbo" (libjpeg-turbo, ~8x faster to encode)
    jpeg_encoder: str = "mozjpeg"


# ------------------------------------------------------------------ sources
@dataclass
class ImageSource:
    """One input image: a path, or the bytes of a tar member; its caption and metadata."""

    key: str
    source: str | bytes
    caption: str | None = None
    meta: str | None = None  # the json sidecar's text

    def data(self) -> bytes:
        if isinstance(self.source, bytes):
            return self.source
        with open(self.source, "rb") as f:
            return f.read()


def _read_text(path: Path) -> str | None:
    return path.read_text(errors="replace").strip() if path.exists() else None


def _folder(root: Path) -> Iterator[ImageSource]:
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    for p in files:
        key = str(p.relative_to(root).with_suffix(""))
        caption = _read_text(p.with_suffix(".txt"))
        meta = _read_text(p.with_suffix(".json"))
        yield ImageSource(key, str(p), caption, meta)


def _tar(path: Path) -> Iterator[ImageSource]:
    """WebDataset groups: consecutive members sharing a key (the path up to the first dot of
    the file name). The image is the member with an image extension, else the group's only
    non-sidecar member (e.g. ``.raw`` bytes: the format is read from the bytes)."""
    group: dict[str, bytes] = {}
    current = None

    def flush() -> Iterator[ImageSource]:
        if current is None:
            return
        images = [s for s in group if s.lower() in IMAGE_EXTENSIONS]
        others = [s for s in group if s.lower() not in SIDECARS]
        suffix = images[0] if images else (others[0] if len(others) == 1 else None)
        if suffix is None:
            return
        caption = group.get(".txt")
        meta = group.get(".json")
        yield ImageSource(
            current,
            group[suffix],
            caption.decode(errors="replace").strip() if caption else None,
            meta.decode(errors="replace") if meta else None,
        )

    with tarfile.open(path) as tar:
        for member in tar:
            if not member.isfile():
                continue
            directory, _, name = member.name.rpartition("/")
            stem, dot, rest = name.partition(".")
            key = f"{directory}/{stem}" if directory else stem
            if key != current:
                yield from flush()
                group, current = {}, key
            file = tar.extractfile(member)
            if file is not None:
                group[dot + rest] = file.read()
        yield from flush()


def expand(sources: "list[str | ImageSource]") -> Iterator[ImageSource]:
    """Image files, every image below folders, the images of tar archives, and
    ``ImageSource`` objects as they are."""
    for source in sources:
        if isinstance(source, ImageSource):
            yield source
            continue
        path = Path(source)
        if path.is_dir():
            yield from _folder(path)
        elif tarfile.is_tarfile(path):
            yield from _tar(path)
        else:
            yield ImageSource(path.stem, str(path))


# ------------------------------------------------------------------ encoding
# (bytes or None when too small, stored w, h, source w, h)
Encoded = tuple[bytes | None, int, int, int, int]


def encode_native(image: ImageSource, enc: ImageEncoding) -> Encoded:
    return _core.encode_image(image.data(), **asdict(enc))


def _pillow_rgb(data: bytes):
    """Decode with Pillow: upright (EXIF orientation), alpha onto white, RGB."""
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        img = Image.new("RGB", rgba.size, (255, 255, 255))
        img.paste(rgba, mask=rgba.getchannel("A"))
    return img.convert("RGB")


def encode_pillow_decoded(image: ImageSource, enc: ImageEncoding) -> Encoded:
    """Decode with Pillow (inputs the native decoders refuse, e.g. CMYK JPEGs), then the native
    encode of that picture (passed on as an uncompressed PNG)."""
    out = io.BytesIO()
    _pillow_rgb(image.data()).save(out, "PNG", compress_level=0)
    return _core.encode_image(out.getvalue(), **asdict(enc))


def encode_pillow(image: ImageSource, enc: ImageEncoding) -> Encoded:
    """The same encode with Pillow (JPEG and WebP): EXIF orientation, alpha onto white, the
    Lanczos resize, libjpeg-turbo / libwebp at the same settings."""
    from PIL import Image

    if enc.codec not in ("jpeg", "webp"):
        raise ValueError(f"the Pillow backend writes jpeg and webp, not {enc.codec}")
    img = _pillow_rgb(image.data())

    sw, sh = img.size
    short = min(sw, sh)
    if short < enc.min_short:
        return None, 0, 0, sw, sh
    if short > enc.max_short:
        scale = enc.max_short / short
        size = (
            max(enc.max_short, round(sw * scale)),
            max(enc.max_short, round(sh * scale)),
        )
        img = img.resize(size, Image.Resampling.LANCZOS)

    out = io.BytesIO()
    if enc.codec == "jpeg":
        subsampling = {"420": 2, "444": 0}[enc.chroma]
        img.save(
            out, "JPEG", quality=enc.quality, subsampling=subsampling, optimize=True
        )
    else:
        method = 4 if enc.effort is None else enc.effort
        img.save(out, "WEBP", quality=enc.quality, method=method)
    w, h = img.size
    return out.getvalue(), w, h, sw, sh


def _write_manifest(out_dir: str, rows: list[dict]) -> str:
    """manifest.parquet (pyarrow), else manifest.jsonl."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        path = os.path.join(out_dir, "manifest.jsonl")
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return path
    path = os.path.join(out_dir, "manifest.parquet")
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def unique_keys(keys: list[str]) -> list[str]:
    """Keys of one shard, numbered when two images share one."""
    seen: dict[str, int] = {}
    out = []
    for key in keys:
        count = seen.get(key, 0)
        seen[key] = count + 1
        out.append(key if count == 0 else f"{key}_{count}")
    return out


def write_images(
    sources: "list[str | ImageSource]",
    out_dir: str,
    enc: ImageEncoding | None = None,
    per_shard: int = 10000,
    workers: int = os.cpu_count() or 1,
    backend: str = "native",
    fallback: bool = True,
) -> list[str]:
    """Encode ``sources`` with ``workers`` threads and pack them into
    ``out_dir/shard_XXXXX.zip``; writes the manifest and the skipped list. Returns the shards.
    """
    enc = enc or ImageEncoding()
    if backend not in ("native", "pillow"):
        raise ValueError(f"unknown backend {backend!r}")
    primary = encode_native if backend == "native" else encode_pillow
    extension = STORED_EXTENSION[enc.codec]

    def try_encode(image: ImageSource) -> tuple[ImageSource, Encoded | None, str]:
        try:
            return image, primary(image, enc), ""
        except (RuntimeError, ValueError, OSError) as e:
            error = str(e)
        if fallback and backend == "native":
            try:
                return image, encode_pillow_decoded(image, enc), ""
            except Exception as e:  # noqa: BLE001 (any undecodable input is skipped)
                error += f"; pillow: {e}"
        return image, None, error

    os.makedirs(out_dir, exist_ok=True)
    shards: list[str] = []
    manifest: list[dict] = []
    skipped: list[dict] = []
    batch: list[ImageSource] = []
    header = json.dumps({"encoding": asdict(enc)})

    def flush(pool: ThreadPoolExecutor) -> None:
        # (source, stored bytes, stored w, h, source w, h) of the images that made it
        stored: list[tuple[ImageSource, bytes, int, int, int, int]] = []
        for image, result, error in pool.map(try_encode, batch):
            if result is None:
                skipped.append({"key": image.key, "reason": "failed", "error": error})
                continue
            data, w, h, sw, sh = result
            if data is None:
                skipped.append(
                    {"key": image.key, "reason": "too_small", "w": sw, "h": sh}
                )
                continue
            stored.append((image, data, w, h, sw, sh))
        batch.clear()
        if not stored:
            return

        name = f"shard_{len(shards):05d}.zip"
        keys = unique_keys([image.key for image, *_ in stored])
        members = []
        for k, (key, (image, data, w, h, sw, sh)) in enumerate(zip(keys, stored)):
            members.append((f"{key}.{extension}", data, w, h, enc.codec))
            manifest.append(
                {
                    "shard": name,
                    "index": k,
                    "key": key,
                    "w": w,
                    "h": h,
                    "source_w": sw,
                    "source_h": sh,
                    "native_res": (w, h) == (sw, sh),
                    "caption": image.caption,
                    "meta": image.meta,
                }
            )
        path = os.path.join(out_dir, name)
        _core.pack_images(path, members, header)
        shards.append(path)

    with ThreadPoolExecutor(workers) as pool:
        for image in expand(sources):
            batch.append(image)
            if len(batch) == per_shard:
                flush(pool)
        if batch:
            flush(pool)

    _write_manifest(out_dir, manifest)
    with open(os.path.join(out_dir, "skipped.jsonl"), "w") as f:
        for row in skipped:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(out_dir, "encoding.json"), "w") as f:
        json.dump(asdict(enc), f, indent=2)
    return shards


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("out_dir")
    ap.add_argument("sources", nargs="+")
    for name, default in asdict(ImageEncoding()).items():
        flag = f"--{name.replace('_', '-')}"
        if isinstance(default, bool):
            ap.add_argument(
                flag, action=argparse.BooleanOptionalAction, default=default
            )
        elif name == "jpeg_encoder":
            ap.add_argument(flag, choices=("mozjpeg", "turbo"), default=default)
        elif default is None:
            ap.add_argument(flag, type=int, default=None)
        else:
            ap.add_argument(flag, type=type(default), default=default)
    ap.add_argument("--per-shard", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--backend", choices=("native", "pillow"), default="native")
    ap.add_argument("--fallback", action=argparse.BooleanOptionalAction, default=True)
    a = vars(ap.parse_args())

    enc = ImageEncoding(**{k: a[k] for k in asdict(ImageEncoding())})
    shards = write_images(
        a["sources"],
        a["out_dir"],
        enc,
        per_shard=a["per_shard"],
        workers=a["workers"],
        backend=a["backend"],
        fallback=a["fallback"],
    )
    for path in shards:
        print(path)


if __name__ == "__main__":
    main()
