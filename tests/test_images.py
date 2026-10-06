"""Images end to end: write shards from synthetic images (folders, tars, odd inputs), read
batches back, compare with a Pillow + torch reference, across sources, codecs and modes.
"""

import io
import json
import os
import tarfile
import zipfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL")
import torch.nn.functional as F
from PIL import Image

from kohakuclip import ImageReader, _core, permute
from kohakuclip.image_writer import ImageEncoding, write_images

SIZE = 128
# (width, height): above the cap, between floor and cap, under the floor, portrait
SOURCES = [(1200, 800), (700, 450), (300, 200), (480, 900)]


def picture(w: int, h: int, seed: int) -> Image.Image:
    """A smooth color picture with some detail (compresses like a photo, not like noise)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rng = np.random.default_rng(seed)
    channels = []
    for c in range(3):
        fx, fy = rng.uniform(0.005, 0.03, 2)
        phase = rng.uniform(0, 6.28)
        channels.append(127 + 100 * np.sin(fx * xx + phase) * np.cos(fy * yy + c))
    rgb = np.stack(channels, -1) + rng.normal(0, 4, (h, w, 3))
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def jpeg_bytes(img: Image.Image, **kwargs) -> bytes:
    out = io.BytesIO()
    img.save(out, "JPEG", quality=95, **kwargs)
    return out.getvalue()


@pytest.fixture(scope="module")
def folder(tmp_path_factory) -> str:
    root = tmp_path_factory.mktemp("images") / "src"
    (root / "nested").mkdir(parents=True)
    for i, (w, h) in enumerate(SOURCES):
        sub = root / "nested" if i % 2 else root
        (sub / f"img{i}.jpg").write_bytes(jpeg_bytes(picture(w, h, i)))
        (sub / f"img{i}.txt").write_text(f"caption {i}")
    return str(root)


@pytest.fixture(scope="module")
def shards(folder, tmp_path_factory) -> list[str]:
    out = tmp_path_factory.mktemp("shards")
    return write_images([folder], str(out), ImageEncoding(), per_shard=2, workers=2)


def manifest(shard: str) -> list[dict]:
    path = os.path.join(os.path.dirname(shard), "manifest.parquet")
    if os.path.exists(path):
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    with open(os.path.join(os.path.dirname(shard), "manifest.jsonl")) as f:
        return [json.loads(line) for line in f]


def stored_image(reader: ImageReader, i: int) -> Image.Image:
    info = reader.info(i)
    with zipfile.ZipFile(info.source) as z:
        return Image.open(io.BytesIO(z.read(info.name))).convert("RGB")


def reference(img: Image.Image, size: int = SIZE) -> np.ndarray:
    """Short side -> size (torch antialiased bilinear), center crop, CHW uint8."""
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None].float()
    h, w = x.shape[2:]
    scale = size / min(h, w)
    nh, nw = max(size, round(h * scale)), max(size, round(w * scale))
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", antialias=True)
    top, left = (nh - size) // 2, (nw - size) // 2
    x = x[0, :, top : top + size, left : left + size]
    return x.round().clamp(0, 255).byte().numpy()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return 10 * np.log10(255**2 / max(mse, 1e-10))


def test_storage_format(shards):
    """Short sides capped at 512, 384-511 kept, under 384 dropped; manifest rows match."""
    reader = ImageReader(shards)
    sizes = sorted((reader.info(i).w, reader.info(i).h) for i in range(len(reader)))
    assert sizes == sorted([(768, 512), (700, 450), (480, 900)])
    rows = manifest(shards[0])
    assert len(rows) == len(reader)
    assert {r["caption"] for r in rows} == {"caption 0", "caption 1", "caption 3"}
    for i in range(len(reader)):
        info = reader.info(i)
        row = next(r for r in rows if r["key"] + ".jpg" == info.name)
        assert (row["w"], row["h"]) == (info.w, info.h)
        assert info.codec == "jpeg"
    with open(os.path.join(os.path.dirname(shards[0]), "skipped.jsonl")) as f:
        skipped = [json.loads(line) for line in f]
    assert [s["reason"] for s in skipped] == ["too_small"]


def test_rgb_matches_reference(shards):
    """Center crops (no DCT shrinking) agree with Pillow decode + torch resize."""
    reader = ImageReader(shards, size=SIZE, threads=2, crop="center", dct_scale=False)
    out = reader.read(list(range(len(reader)))).image
    for i, got in enumerate(out):
        want = reference(stored_image(reader, i))
        diff = np.abs(got.astype(int) - want.astype(int))
        # chroma: upsampled before (Pillow) vs resized as planes (here)
        assert psnr(got, want) > 38 and diff.mean() < 1.5, (i, psnr(got, want))


def test_dct_scale_is_close(shards):
    """Shrinking in the IDCT (1/2, 1/4) is a different low-pass than the resize: close."""
    fast = ImageReader(shards, size=SIZE, crop="center")
    full = ImageReader(shards, size=SIZE, crop="center", dct_scale=False)
    ids = list(range(len(fast)))
    assert psnr(fast.read(ids).image, full.read(ids).image) > 34


def test_sources_agree(shards, tmp_path):
    """The indexed zip, the zip listed without its index, a tar written with an index, and a
    folder of the stored files return the same batches."""
    members = []
    for path in shards:
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name != "__index__.bin":
                    members.append((name, z.read(name)))
                    (tmp_path / "tree" / name).parent.mkdir(parents=True, exist_ok=True)
                    (tmp_path / "tree" / name).write_bytes(z.read(name))
    reader = ImageReader(shards)
    sizes = {
        reader.info(i).name: (reader.info(i).w, reader.info(i).h) for i in range(3)
    }
    tar = str(tmp_path / "all.tar")
    _core.pack_images(tar, [(n, d, *sizes[n], "jpeg") for n, d in members])

    outs = {}
    for label, kwargs in {
        "zip": {"sources": shards},
        "zip-listed": {"sources": shards, "use_index": False},
        "tar": {"sources": [tar]},
        "folder": {"sources": [str(tmp_path / "tree")]},
    }.items():
        r = ImageReader(size=SIZE, crop="center", hflip=0.5, **kwargs)
        by_name = {r.info(i).name: i for i in range(len(r))}
        ids = [by_name[n] for n in sorted(by_name)]
        outs[label] = r.read(ids, seed=3).image
    for label, out in outs.items():
        assert np.array_equal(out, outs["zip"]), label


def test_augmentations(shards):
    """Seeded batches repeat; flips are flips; random resized crops cover the output."""
    reader = ImageReader(shards, size=SIZE, crop="resized", scale=(0.2, 1.0), hflip=0.5)
    ids = [0, 1, 2, 0, 1, 2]
    a, b = reader.read(ids, seed=1).image, reader.read(ids, seed=1).image
    assert np.array_equal(a, b)
    assert not np.array_equal(a, reader.read(ids, seed=2).image)

    plain = ImageReader(shards, size=SIZE, crop="center")
    flipped = ImageReader(shards, size=SIZE, crop="center", hflip=1.0, vflip=1.0)
    x, y = plain.read([0, 1, 2]).image, flipped.read([0, 1, 2]).image
    assert np.array_equal(x[..., ::-1, ::-1], y)


def test_yuv_modes_match_rgb(shards):
    """GPU-side conversion (run on the CPU here) of full-range JPEG planes (4:4:4 stored,
    resampled to 4:2:0) vs rgb."""
    from kohakuclip.torch import yuv_to_rgb

    rgb = ImageReader(shards, size=SIZE, crop="center", dct_scale=False)
    for mode in ("yuv_resized", "yuv"):
        yuv = ImageReader(shards, size=SIZE, crop="center", mode=mode)
        ids = [0, 1, 2]
        if mode == "yuv":
            # "yuv" windows the batch's smallest stored short side: only images with that
            # short side cover the rgb crop's region
            ids = [i for i in ids if min(yuv.info(i).h, yuv.info(i).w) == 450]
        batch = yuv.read(ids)
        assert batch.colorspace == ["bt601-full"] * len(ids)
        side = 450 if mode == "yuv" else SIZE
        assert batch.window.tolist() == [[side, side]] * len(ids)

        ref = torch.from_numpy(rgb.read(ids).image).float()
        got = (yuv_to_rgb(batch, SIZE, device="cpu") + 1) * 127.5
        assert got.shape == (len(ids), 3, SIZE, SIZE)
        p = 10 * np.log10(255**2 / ((got - ref) ** 2).mean().item())
        assert p > 36, (mode, p)


@pytest.mark.parametrize("codec", ["av1", "jxl", "webp"])
def test_other_codecs(folder, tmp_path, codec):
    """AV1 (intra OBUs), JPEG XL and WebP shards read back close to the JPEG ones."""
    jpeg = write_images([folder], str(tmp_path / "jpeg"), ImageEncoding(), workers=2)
    other = write_images(
        [folder], str(tmp_path / codec), ImageEncoding(codec=codec), workers=2
    )
    a = ImageReader(jpeg, size=SIZE, crop="center").read([0, 1, 2]).image
    reader = ImageReader(other, size=SIZE, crop="center")
    assert reader.info(0).codec == codec
    b = reader.read([0, 1, 2]).image
    assert psnr(a, b) > 33, psnr(a, b)


def test_odd_inputs(tmp_path):
    """EXIF orientation is applied, alpha lands on white, gray stays gray, CMYK JPEGs go
    through the Pillow fallback, a WebDataset tar keeps captions."""
    src = tmp_path / "src"
    src.mkdir()
    base = picture(900, 600, 7)

    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways: rotate 90 degrees clockwise to display
    (src / "rotated.jpg").write_bytes(jpeg_bytes(base, exif=exif))
    rgba = base.convert("RGBA")
    alpha = np.full((600, 900), 255, np.uint8)
    alpha[:, :450] = 0  # left half transparent
    rgba.putalpha(Image.fromarray(alpha))
    rgba.save(src / "alpha.png")
    (src / "gray.jpg").write_bytes(jpeg_bytes(base.convert("L")))
    (src / "cmyk.jpg").write_bytes(jpeg_bytes(base.convert("CMYK")))

    shards = write_images([str(src)], str(tmp_path / "out"), workers=2)
    reader = ImageReader(shards, size=SIZE, crop="center", dct_scale=False)
    by_name = {reader.info(i).name: i for i in range(len(reader))}
    assert sorted(by_name) == ["alpha.jpg", "cmyk.jpg", "gray.jpg", "rotated.jpg"]

    rotated = reader.info(by_name["rotated.jpg"])
    assert (rotated.w, rotated.h) == (512, 768)
    got = reader.read([by_name["rotated.jpg"]]).image[0]
    want = reference(base.transpose(Image.Transpose.ROTATE_270).resize((512, 768)))
    assert psnr(got, want) > 30

    white = reader.read([by_name["alpha.jpg"]]).image[0]
    assert white[:, :, : SIZE // 3].min() > 245  # the transparent left side
    gray = reader.read([by_name["gray.jpg"]]).image[0]
    assert np.array_equal(gray[0], gray[1]) and np.array_equal(gray[0], gray[2])
    cmyk = reader.read([by_name["cmyk.jpg"]]).image[0]
    assert psnr(cmyk, reference(base.resize((768, 512)))) > 25

    tar = tmp_path / "wds.tar"
    with tarfile.open(tar, "w") as t:
        for key, data in [("a", jpeg_bytes(base)), ("b", jpeg_bytes(base))]:
            for suffix, payload in [(".jpg", data), (".txt", f"about {key}".encode())]:
                info = tarfile.TarInfo(f"part/{key}{suffix}")
                info.size = len(payload)
                t.addfile(info, io.BytesIO(payload))
    out = write_images([str(tar)], str(tmp_path / "wds"), workers=2)
    rows = manifest(out[0])
    assert [(r["key"], r["caption"]) for r in rows] == [
        ("part/a", "about a"),
        ("part/b", "about b"),
    ]


def test_pillow_backend_agrees(folder, tmp_path):
    native = write_images([folder], str(tmp_path / "native"), workers=2)
    pillow = write_images(
        [folder], str(tmp_path / "pillow"), workers=2, backend="pillow"
    )
    a = ImageReader(native, size=SIZE, crop="center").read([0, 1, 2]).image
    b = ImageReader(pillow, size=SIZE, crop="center").read([0, 1, 2]).image
    assert psnr(a, b) > 38, psnr(a, b)


def test_permute():
    n = 1000
    order = permute(n, 5, list(range(n)))
    assert sorted(order) == list(range(n)) and order != list(range(n))
    assert permute(n, 5, [3, 7]) == [order[3], order[7]]
    with pytest.raises(IndexError):
        permute(n, 5, [n])


def test_dataset_batches(shards):
    """ImageDataset: the same batches for any worker count, exact resumption, epochs that
    visit every image once."""
    from torch.utils.data import DataLoader

    from kohakuclip.torch import ClipLoader, ImageDataset

    def make(**kwargs):
        args = {"batch_size": 2, "size": 64, "threads": 2, "hflip": 0.5, "seed": 7}
        return ImageDataset(shards, **{**args, **kwargs})

    def batches(dataset, n, workers=0):
        loader = DataLoader(dataset, batch_size=None, num_workers=workers)
        out = []
        for image in loader:
            out.append(image.clone())
            if len(out) == n:
                break
        return out

    direct = batches(make(), 6)
    assert direct[0].shape == (2, 3, 64, 64)
    for a, b in zip(direct, batches(make(), 6, workers=2)):
        assert torch.equal(a, b)

    loader = ClipLoader(make(), workers=0)
    it = iter(loader)
    for _ in range(3):
        next(it)
    resumed = ClipLoader(make(seed=0))
    resumed.load_state_dict(loader.state_dict())
    for a, b in zip(direct[3:], resumed):
        assert torch.equal(a, b)
        break

    epochs = make(epochs=True, batch_size=1)
    n = len(epochs._reader())
    seen = [i for index in range(n) for i in epochs._images(0, 1, index)]
    assert sorted(seen) == list(range(n))


def ycbcr_planes(data: bytes) -> np.ndarray:
    """A JPEG's decoded Y, Cb, Cr (libjpeg's accurate IDCT, chroma upsampled with libjpeg's
    fancy upsampling when subsampled), HWC uint8."""
    img = Image.open(io.BytesIO(data))
    img.draft("YCbCr", img.size)
    assert img.mode == "YCbCr"
    return np.asarray(img)


def full_range_rgb(ycc: np.ndarray) -> np.ndarray:
    """KohakuClip's full-range BT.601 YUV -> RGB (Q14 coefficients, the same integer math),
    CHW uint8."""
    q14 = {
        k: round(v * 16384)
        for k, v in {"rv": 1.402, "gu": -0.344136, "gv": -0.714136, "bu": 1.772}.items()
    }
    y = ycc[..., 0].astype(np.int64) * 16384 * 16
    cb = (ycc[..., 1].astype(np.int64) - 128) * 16
    cr = (ycc[..., 2].astype(np.int64) - 128) * 16

    def pixel(c):
        return np.clip((c + (1 << 17)) >> 18, 0, 255).astype(np.uint8)

    r = pixel(y + q14["rv"] * cr)
    g = pixel(y + q14["gu"] * cb + q14["gv"] * cr)
    b = pixel(y + q14["bu"] * cb)
    return np.stack([r, g, b])


def test_mozjpeg_encoding():
    """The default JPEG encoder is mozjpeg: baseline (sequential) scans, 4:4:4, smaller than
    libjpeg-turbo at the same quality, close to the same picture."""
    source = jpeg_bytes(picture(900, 700, 11))
    moz, w, h, _, _ = _core.encode_image(source, quality=70)
    turbo, *_ = _core.encode_image(source, quality=70, jpeg_encoder="turbo")
    assert (w, h) == (658, 512)
    assert b"\xff\xc0" in moz[:2000] and b"\xff\xc2" not in moz  # SOF0, not progressive
    assert Image.open(io.BytesIO(moz)).layer == [
        (1, 1, 1, 0),
        (2, 1, 1, 1),
        (3, 1, 1, 1),
    ]
    assert len(moz) < len(turbo)
    a = np.asarray(Image.open(io.BytesIO(moz)).convert("RGB"))
    b = np.asarray(Image.open(io.BytesIO(turbo)).convert("RGB"))
    assert psnr(a, b) > 36
    gray, *_ = _core.encode_image(jpeg_bytes(picture(600, 500, 3).convert("L")))
    assert Image.open(io.BytesIO(gray)).mode == "L"
    with pytest.raises(ValueError):
        _core.encode_image(source, jpeg_encoder="libjpeg")


def test_mozjpeg_decodes_bit_exact(tmp_path):
    """Reading mozjpeg files at their stored size (no resize) is bit-for-bit libjpeg's
    decode + the conversion, for even and odd sizes, with flips."""
    src = tmp_path / "src"
    src.mkdir()
    for i, (w, h) in enumerate([(700, 450), (453, 391), (512, 777)]):
        (src / f"img{i}.png").write_bytes(b"")
        picture(w, h, 20 + i).save(src / f"img{i}.png")
    shards = write_images([str(src)], str(tmp_path / "out"), workers=2)
    probe = ImageReader(shards)
    for i in range(len(probe)):
        info = probe.info(i)
        with zipfile.ZipFile(info.source) as z:
            data = z.read(info.name)
        ycc = ycbcr_planes(data)
        side = min(info.w, info.h)
        top, left = (info.h - side) // 2, (info.w - side) // 2
        want = full_range_rgb(ycc[top : top + side, left : left + side])
        for hflip in (0.0, 1.0):
            reader = ImageReader(
                shards, size=side, crop="center", hflip=hflip, dct_scale=False
            )
            got = reader.read([i]).image[0]
            expected = want[:, :, ::-1] if hflip else want
            assert np.array_equal(got, expected), (info.name, hflip)


def test_plain_jpeg_sources(tmp_path):
    """JPEG files read straight from a plain tar, a zip (stored and deflated members) and a
    folder, with other files around: the same batches as a KohakuClip shard of the same
    bytes, and odd-sized 4:2:0 JPEGs close to libjpeg's own decode."""
    files = {}
    for i, (w, h) in enumerate([(640, 467), (501, 389), (800, 600)]):
        files[f"dir/img{i}.jpg"] = jpeg_bytes(picture(w, h, 30 + i), subsampling=2)
    files["dir/IMG3.JPEG"] = jpeg_bytes(picture(450, 600, 33), subsampling=0)
    extras = {"dir/readme.txt": b"not an image", "dir/img0.json": b"{}"}

    folder = tmp_path / "folder"
    for name, data in {**files, **extras}.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)
    tar = tmp_path / "plain.tar"
    with tarfile.open(tar, "w") as t:
        for name, data in {**files, **extras}.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    zips = {}
    for label, method in (
        ("stored", zipfile.ZIP_STORED),
        ("deflated", zipfile.ZIP_DEFLATED),
    ):
        zips[label] = tmp_path / f"{label}.zip"
        with zipfile.ZipFile(zips[label], "w", method) as z:
            for name, data in {**files, **extras}.items():
                z.writestr(name, data)
    shard = str(tmp_path / "shard.zip")
    sizes = {n: Image.open(io.BytesIO(d)).size for n, d in files.items()}
    _core.pack_images(shard, [(n, d, *sizes[n], "jpeg") for n, d in files.items()])

    sources = {
        "shard": [shard],
        "tar": [str(tar)],
        "zip-stored": [str(zips["stored"])],
        "zip-deflated": [str(zips["deflated"])],
        "folder": [str(folder)],
    }
    outs = {}
    for label, paths in sources.items():
        for crop in ("center", "resized"):
            reader = ImageReader(paths, size=SIZE, crop=crop, hflip=0.5)
            by_name = {reader.info(i).name: i for i in range(len(reader))}
            assert sorted(by_name) == sorted(files), label
            ids = [by_name[n] for n in sorted(files)]
            outs[label, crop] = reader.read(ids, seed=5).image
    for (label, crop), out in outs.items():
        assert np.array_equal(out, outs["shard", crop]), (label, crop)

    # odd-sized 4:2:0 at the stored size: chroma stays aligned with libjpeg's upsampling
    reader = ImageReader([str(tar)], size=467, crop="center", dct_scale=False)
    i = next(i for i in range(len(reader)) if reader.info(i).name == "dir/img0.jpg")
    got = reader.read([i]).image[0]
    want = np.asarray(Image.open(io.BytesIO(files["dir/img0.jpg"])).convert("RGB"))
    left = (640 - 467) // 2
    want = want[:, left : left + 467].transpose(2, 0, 1)
    assert psnr(got, want) > 40, psnr(got, want)


def test_writer_keeps_odd_chroma_aligned():
    """An odd-sized 4:2:0 JPEG stored as 4:4:4 at its own size: chroma lines up with
    libjpeg's upsampled chroma (a ratio-based resize drifts by up to half a sample)."""
    source = jpeg_bytes(picture(640, 467, 41), subsampling=2)
    stored, w, h, _, _ = _core.encode_image(source, quality=100)
    assert (w, h) == (640, 467)
    a = ycbcr_planes(stored).astype(np.float64)
    b = ycbcr_planes(source).astype(np.float64)
    for c in (1, 2):
        assert psnr(a[..., c], b[..., c]) > 48, (c, psnr(a[..., c], b[..., c]))
