"""Batch reader: plans each request from the shard index, then decodes the whole batch natively.

    reader = Reader(["data/shard_000.zip", ...], size=256, threads=16)
    video = reader.read([(vid, frames), ...])            # uint8 [B, T, 3, 256, 256]

``mode="rgb"`` (default): the native core resizes the short side to ``size`` (antialiased bilinear)
fused with a ``size`` x ``size`` crop and flips, so only output pixels are produced on the CPU.
``mode="yuv"``: the core returns the stored-resolution crop window as raw YUV 4:2:0 and
``kohakuclip.torch.yuv_to_rgb`` converts and resizes it on the GPU (half the bytes, no CPU resize).
"""

import ctypes as C
import random
from dataclasses import dataclass

import numpy as np

from . import _native
from .shard import Shard, Video


@dataclass
class Augment:
    crop: str = "random"   # "random" | "center"
    hflip: float = 0.0     # probability of a horizontal flip
    vflip: float = 0.0


@dataclass
class Batch:
    video: np.ndarray               # rgb: uint8 [B, T, 3, S, S]; yuv: uint8 [B, T, H*W*3/2]
    window: np.ndarray | None = None  # yuv: [B, 2] window (h, w) in stored pixels
    flips: np.ndarray | None = None   # yuv: [B, 2] (hflip, vflip), applied on the GPU
    colorspace: list | None = None    # yuv: per clip "bt709" / "bt601"


class Reader:
    def __init__(self, shards, size: int = 256, mode: str = "rgb", threads: int = 1,
                 augment: Augment | None = None, seed: int | None = None):
        self.shards = [s if isinstance(s, Shard) else Shard(s) for s in shards]
        self.videos = [(s, i) for s in self.shards for i in range(len(s))]
        self.size, self.mode, self.threads = size, mode, threads
        self.augment = augment or Augment()
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.videos)

    def info(self, vid: int) -> Video:
        shard, i = self.videos[vid]
        return shard.video(i)

    def read(self, items: list[tuple[int, list[int]]], out: np.ndarray | None = None) -> Batch:
        """items: (video id, frame indices) per clip, all with the same number of frames."""
        t = len(items[0][1])
        s = self.size
        yuv = self.mode == "yuv"
        keep, reqs, dups, windows, flips, spaces = [], [], [], [], [], []
        infos = [self.info(vid) for vid, _ in items]
        if yuv:  # window side = stored short side (the crop spans the resized short side)
            side = min(min(v.h, v.w) for v in infos) // 2 * 2
            out = np.empty((len(items), t, side * side * 3 // 2), np.uint8) if out is None else out
        else:
            out = np.empty((len(items), t, 3, s, s), np.uint8) if out is None else out
        for b, ((vid, frames), v) in enumerate(zip(items, infos)):
            want, inverse = np.unique(np.asarray(frames, np.int32), return_inverse=True)
            target = out[b] if len(want) == t else np.empty((len(want),) + out.shape[2:], np.uint8)
            r, arrays = self._plan(v, self.videos[vid][0].fd, want, target, side if yuv else None)
            reqs.append(r)
            keep.append(arrays)
            dups.append(None if len(want) == t else (b, target, inverse))
            if yuv:
                windows.append((side, side))
                flips.append((r.hflip, r.vflip))
                spaces.append(v.colorspace)
                r.hflip = r.vflip = 0
        status = _native.decode(reqs, self.threads)
        if status.any():
            raise RuntimeError(f"decode failed for clips {np.flatnonzero(status).tolist()} (status {status[status != 0].tolist()})")
        for d in dups:
            if d is not None:
                b, target, inverse = d
                out[b] = target[inverse]
        if yuv:
            return Batch(out, np.array(windows), np.array(flips, np.int8), spaces)
        return Batch(out)

    def _plan(self, v: Video, fd: int, want: np.ndarray, target: np.ndarray, side: int | None):
        # one group per GOP that holds wanted frames: bytes from its keyframe to its last wanted frame
        key_of = v.keys[np.searchsorted(v.keys, want, side="right") - 1]
        starts, first = np.unique(key_of, return_index=True)
        ends = np.maximum.reduceat(want, first) + 1
        pk = np.concatenate([np.arange(a, e, dtype=np.int32) for a, e in zip(starts, ends)])
        arrays = dict(
            group_off=np.ascontiguousarray(v.off[starts], np.int64),
            group_npk=(ends - starts).astype(np.int32),
            pk_len=np.ascontiguousarray(v.size[pk], np.int32),
            pk_idx=pk,
            want=want,
            prefix=np.frombuffer(v.prefix, np.uint8) if v.prefix else None,
        )
        a = self.augment
        rng = self.rng
        if side is None:  # rgb: resize short side to size, crop size x size
            scale = self.size / min(v.h, v.w)
            nh, nw = max(self.size, round(v.h * scale)), max(self.size, round(v.w * scale))
            oh = ow = self.size
        else:  # yuv: stored-resolution window, even origin
            nh, nw, oh, ow = v.h, v.w, side, side
        if a.crop == "random":
            top, left = rng.randint(0, nh - oh), rng.randint(0, nw - ow)
        else:
            top, left = (nh - oh) // 2, (nw - ow) // 2
        if side is not None:
            top, left = top & ~1, left & ~1
        hflip, vflip = rng.random() < a.hflip, rng.random() < a.vflip
        r = _native.Request(
            fd=fd, codec=_native.CODECS[v.codec], mode=_native.MODES[self.mode], ngroups=len(starts),
            group_off=arrays["group_off"].ctypes.data, group_npk=arrays["group_npk"].ctypes.data,
            pk_len=arrays["pk_len"].ctypes.data, pk_idx=arrays["pk_idx"].ctypes.data,
            prefix=arrays["prefix"].ctypes.data if arrays["prefix"] is not None else None,
            nprefix=len(v.prefix), nwant=len(want), want=want.ctypes.data,
            nh=nh, nw=nw, top=top, left=left, oh=oh, ow=ow, hflip=int(hflip), vflip=int(vflip),
            out=target.ctypes.data,
        )
        arrays["target"] = target
        return r, arrays
