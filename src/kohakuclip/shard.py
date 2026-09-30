"""Shards: zip archives (stored, no compression) of faststart mp4 files, plus an optional index member.

The index (``__index__.bin``, written by ``kohakuclip.writer``) holds, per video, the absolute byte
offset, size and flags of every frame, so a clip is read with one pread per GOP. It is memory-mapped
straight from the zip: the OS page cache shares it between workers and nothing extra is stored.
Without it, each video's frame table is parsed from its own ``moov`` box (faststart puts it at the
front) on first use and cached: one extra small read per video.

Index layout: b"KCIDX1\\0\\0" | u64 json length | json | pad to 8 | frame records (``FRAME``).
"""

import json
import mmap
import os
import struct
import zipfile
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

INDEX = "__index__.bin"
MAGIC = b"KCIDX1\0\0"
FRAME = np.dtype([("off", "<i8"), ("size", "<i4"), ("flags", "u1"), ("pad", "u1", (3,))])
KEY = 1   # flags bit 0: keyframe (closed-GOP start)
SKIP = 2  # flags bit 1: nothing later depends on this sample (AV1 top-layer frame): skip unless wanted


@dataclass
class Video:
    n: int                # frames
    fps: float
    h: int
    w: int
    codec: str            # "av1" | "h264" | "hevc"
    off: np.ndarray       # [n] absolute byte offset of each frame in the shard
    size: np.ndarray      # [n] bytes
    keys: np.ndarray      # sorted keyframe indices
    skip: np.ndarray | None = None  # [n] bool, samples that may be left out when not wanted
    prefix: bytes = b""   # prepended to each GOP's first packet (see ``codec_prefix``)
    colorspace: str = "bt709"


def data_offset(f, header_offset: int) -> int:
    """Absolute offset of a zip member's data (after its local header)."""
    f.seek(header_offset)
    head = f.read(30)
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    return header_offset + 30 + name_len + extra_len


class Shard:
    def __init__(self, path: str, moov_cache: int = 4096):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
        self.members = [(i.filename, i.header_offset) for i in infos if i.filename != INDEX]
        index = next((i for i in infos if i.filename == INDEX), None)
        self.meta, self.frames = self._open_index(index) if index is not None else (None, None)
        self._moov = lru_cache(maxsize=moov_cache)(self._parse_moov)

    def __len__(self) -> int:
        return len(self.members)

    def _open_index(self, info):
        with open(self.path, "rb") as f:
            start = data_offset(f, info.header_offset)
        mm = mmap.mmap(self.fd, 0, prot=mmap.PROT_READ)
        if mm[start:start + 8] != MAGIC:
            raise ValueError(f"{self.path}: bad index magic")
        jlen = struct.unpack("<Q", mm[start + 8:start + 16])[0]
        meta = json.loads(mm[start + 16:start + 16 + jlen])["videos"]
        rec = start + 16 + jlen + (-(16 + jlen) % 8)  # records are 8-aligned within the member
        total = sum(v["n"] for v in meta)
        frames = np.frombuffer(mm, FRAME, count=total, offset=rec)
        return meta, frames

    def video(self, i: int) -> Video:
        if self.meta is None:
            return self._moov(i)
        m = self.meta[i]
        fr = self.frames[m["row"]:m["row"] + m["n"]]
        return Video(m["n"], m["fps"], m["h"], m["w"], m["codec"], fr["off"], fr["size"],
                     np.flatnonzero(fr["flags"] & KEY), (fr["flags"] & SKIP) != 0,
                     bytes.fromhex(m.get("prefix", "")), m.get("colorspace", "bt709"))

    def _parse_moov(self, i: int) -> Video:
        _, header_offset = self.members[i]
        buf = os.pread(self.fd, 64 * 1024, header_offset)
        base = 30 + sum(struct.unpack("<HH", buf[26:30]))
        pos = base
        while True:  # ftyp, then moov (faststart); read the whole moov if it is larger
            size, typ = struct.unpack(">I4s", buf[pos:pos + 8])
            if typ == b"moov":
                if pos + size > len(buf):
                    buf = os.pread(self.fd, pos + size, header_offset)
                return parse_moov(memoryview(buf)[pos:pos + size], header_offset + base)
            if typ == b"mdat":
                raise ValueError(f"{self.path}:{self.members[i][0]} is not faststart (moov after mdat)")
            pos += size


def codec_prefix(codec: str, config: bytes) -> bytes:
    """Decoder prefix from an mp4 codec configuration record (av1C / avcC / hvcC payload): the AV1
    sequence header, or the H.264 / HEVC parameter sets in Annex B. mp4 keeps these out of band."""
    if codec == "av1":
        return config[4:]  # 4-byte av1C header, then configOBUs
    start, nals = b"\0\0\0\1", []
    if codec == "h264":  # avcC: [5] & 31 SPS, each u16 length + NAL; then u8 PPS count, same layout
        if config[4] & 3 != 3:
            raise ValueError("only 4-byte NAL lengths are supported")
        i, count = 6, config[5] & 31
        for _ in range(2):
            for _ in range(count):
                n = struct.unpack(">H", config[i:i + 2])[0]
                nals.append(config[i + 2:i + 2 + n])
                i += 2 + n
            count, i = config[i] if i < len(config) else 0, i + 1
    else:  # hvcC: 22-byte header (lengthSizeMinusOne in [21] & 3), then arrays of NAL units
        if config[21] & 3 != 3:
            raise ValueError("only 4-byte NAL lengths are supported")
        i = 23
        for _ in range(config[22]):
            count = struct.unpack(">H", config[i + 1:i + 3])[0]
            i += 3
            for _ in range(count):
                n = struct.unpack(">H", config[i:i + 2])[0]
                nals.append(config[i + 2:i + 2 + n])
                i += 2 + n
    return b"".join(start + n for n in nals)


# ------------------------------------------------------------------ minimal single-track mp4 moov parser
def _boxes(buf, start, end):
    i = start
    while i + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[i:i + 8])
        head = 8
        if size == 1:
            size, head = struct.unpack(">Q", buf[i + 8:i + 16])[0], 16
        yield typ, i + head, i + size
        i += size


def _find(buf, path, start, end):
    for typ, s, e in _boxes(buf, start, end):
        if typ == path[0]:
            return (s, e) if len(path) == 1 else _find(buf, path[1:], s, e)
    return None


def _u32(buf, at, count):
    return np.frombuffer(buf[at:at + 4 * count], ">u4").astype(np.int64)


def parse_moov(buf, base: int) -> Video:
    """Frame table of the first video track: sizes (stsz), chunk offsets (stco/co64) + samples per
    chunk (stsc) -> absolute offsets, sync samples (stss), fps (mdhd + stts), size and codec (stsd)."""
    trak = _find(buf, [b"trak"], 8, len(buf))
    stbl = _find(buf, [b"mdia", b"minf", b"stbl"], *trak)
    box = {t: (s, e) for t, s, e in _boxes(buf, *stbl)}
    s = box[b"stsz"][0]
    fixed, n = struct.unpack(">II", buf[s + 4:s + 12])
    size = np.full(n, fixed, np.int64) if fixed else _u32(buf, s + 12, n)
    if b"stco" in box:
        s = box[b"stco"][0]
        chunks = _u32(buf, s + 8, struct.unpack(">I", buf[s + 4:s + 8])[0])
    else:
        s = box[b"co64"][0]
        k = struct.unpack(">I", buf[s + 4:s + 8])[0]
        chunks = np.frombuffer(buf[s + 8:s + 8 + 8 * k], ">u8").astype(np.int64)
    s = box[b"stsc"][0]
    k = struct.unpack(">I", buf[s + 4:s + 8])[0]
    stsc = _u32(buf, s + 8, 3 * k).reshape(k, 3)
    per_chunk = np.empty(len(chunks), np.int64)
    for j in range(k):
        per_chunk[stsc[j, 0] - 1:(stsc[j + 1, 0] - 1 if j + 1 < k else len(chunks))] = stsc[j, 1]
    first = np.concatenate([[0], np.cumsum(per_chunk)[:-1]])
    chunk_of = np.repeat(np.arange(len(chunks)), per_chunk)
    csum = np.concatenate([[0], np.cumsum(size)])
    off = base + chunks[chunk_of] + (csum[:-1] - csum[first[chunk_of]])
    if b"stss" in box:
        s = box[b"stss"][0]
        keys = _u32(buf, s + 8, struct.unpack(">I", buf[s + 4:s + 8])[0]) - 1
    else:
        keys = np.arange(n)
    s, _ = _find(buf, [b"mdia", b"mdhd"], *trak)
    timescale = struct.unpack(">I", buf[s + (20 if buf[s] == 1 else 12):][:4])[0]
    s = box[b"stts"][0]
    stts = _u32(buf, s + 8, 2 * struct.unpack(">I", buf[s + 4:s + 8])[0]).reshape(-1, 2)
    fps = timescale * stts[:, 0].sum() / max(1, (stts[:, 0] * stts[:, 1]).sum())
    s, e = box[b"stsd"]
    entry = bytes(buf[s + 8:e])
    codec = {b"av01": "av1", b"avc1": "h264", b"hvc1": "hevc", b"hev1": "hevc"}[entry[4:8]]
    h, w = struct.unpack(">HH", entry[8 + 24:8 + 28])[::-1]
    tag = {"av1": b"av1C", "h264": b"avcC", "hevc": b"hvcC"}[codec]
    j = entry.find(tag)
    prefix = codec_prefix(codec, entry[j + 4:j - 4 + struct.unpack(">I", entry[j - 4:j])[0]])
    return Video(n, float(fps), h, w, codec, off, size, np.asarray(keys, np.int64), None, prefix)
