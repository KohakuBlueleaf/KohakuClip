"""Batch reader: plans each request from the shard index, then decodes the whole batch natively.

    reader = Reader(["data/shard_000.zip", ...], size=256, threads=16)
    video = reader.read([(vid, frames), ...])            # uint8 [B, T, 3, 256, 256]

``mode="rgb"`` (default): the native core resizes the short side to ``size`` (antialiased bilinear)
fused with a ``size`` x ``size`` crop and flips, so only output pixels are produced on the CPU.
``mode="yuv"``: the core returns the stored-resolution window as raw YUV 4:2:0 and
``kohakuclip.torch.yuv_to_rgb`` converts and resizes it on the GPU (no CPU resize or conversion;
more bytes to the GPU when stored frames are larger than the output).
``mode="yuv_resized"``: the core resizes the planes to the output crop (4:2:0, half the bytes of
rgb) and the GPU only converts.
"""

import ctypes as C
import random
from dataclasses import dataclass

import numpy as np

from . import _native
from .shard import Shard, Video


@dataclass
class Augment:
    crop: str = "random"  # "random" | "center"
    hflip: float = 0.0  # probability of a horizontal flip
    vflip: float = 0.0


@dataclass
class Batch:
    video: (
        np.ndarray
    )  # rgb: uint8 [B, T, 3, S, S]; yuv*: uint8 [B, T, H*W*3/2] (Y, U, V)
    window: np.ndarray | None = None  # yuv*: [B, 2] plane size (h, w)
    flips: np.ndarray | None = None  # yuv: [B, 2] (hflip, vflip), applied on the GPU
    colorspace: list | None = None  # yuv: per clip "bt709" / "bt601"


class Pending:
    """A submitted batch: ``result()`` waits for the native decode and returns the ``Batch``."""

    def __init__(self, job, keep, dups, batch: Batch):
        self.job, self.keep, self.dups, self.batch = job, keep, dups, batch

    def result(self) -> Batch:
        status = self.job.wait()
        if status.any():
            bad = np.flatnonzero(status)
            raise RuntimeError(
                f"decode failed for clips {bad.tolist()} (status {status[bad].tolist()})"
            )
        for (
            d
        ) in (
            self.dups
        ):  # clips with repeated frames were decoded once per frame, then expanded
            if d is not None:
                b, target, inverse = d
                self.batch.video[b] = target[inverse]
        self.keep = None
        return self.batch


class Reader:
    def __init__(
        self,
        shards,
        size: int = 256,
        mode: str = "rgb",
        threads: int = 1,
        augment: Augment | None = None,
        seed: int | None = None,
        skip: bool = True,
    ):
        self.shards = [s if isinstance(s, Shard) else Shard(s) for s in shards]
        self.skip = skip  # decode only what later frames need of unwanted samples (AV1, needs the index)
        self.videos = [(s, i) for s in self.shards for i in range(len(s))]
        self.size, self.mode, self.threads = size, mode, threads
        self.augment = augment or Augment()
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.videos)

    def info(self, vid: int) -> Video:
        shard, i = self.videos[vid]
        return shard.video(i)

    def read(
        self, items: list[tuple[int, list[int]]], out: np.ndarray | None = None
    ) -> Batch:
        """items: (video id, frame indices) per clip, all with the same number of frames."""
        return self.submit(items, out).result()

    def submit(
        self, items: list[tuple[int, list[int]]], out: np.ndarray | None = None
    ) -> "Pending":
        """Queue a batch for decoding and return at once; ``.result()`` waits for it. Batches queued
        back to back share the native pool with no barrier in between."""
        t, s = len(items[0][1]), self.size
        infos = [self.info(vid) for vid, _ in items]
        # per-frame output. rgb: [3, S, S]. yuv 4:2:0 (1.5 bytes per pixel): the stored-resolution
        # square window spanning the short side ("yuv"; side = the batch's smallest short side) or
        # the S x S crop resized on the CPU ("yuv_resized", same geometry as rgb; S even)
        if self.mode == "rgb":
            side, shape = None, (3, s, s)
        else:
            side = (
                s
                if self.mode == "yuv_resized"
                else min(min(v.h, v.w) for v in infos) // 2 * 2
            )
            shape = (side * side * 3 // 2,)
        out = np.empty((len(items), t, *shape), np.uint8) if out is None else out
        keep, reqs, dups, flips = [], [], [], []
        for b, ((vid, frames), v) in enumerate(zip(items, infos)):
            want, inverse = np.unique(np.asarray(frames, np.int32), return_inverse=True)
            unique = len(want) == t
            target = out[b] if unique else np.empty((len(want), *shape), np.uint8)
            shard, i = self.videos[vid]
            r, arrays = self._plan(v, shard.fd(i), want, target, side)
            reqs.append(r)
            keep.append(arrays)
            dups.append(None if unique else (b, target, inverse))
            if side is not None:  # flips happen on the GPU
                flips.append((r.hflip, r.vflip))
                r.hflip = r.vflip = 0
        batch = Batch(out)
        if side is not None:
            batch = Batch(
                out,
                np.full((len(items), 2), side),
                np.array(flips, np.int8),
                [v.colorspace for v in infos],
            )
        return Pending(_native.Job(reqs, self.threads), keep, dups, batch)

    def _plan(
        self, v: Video, fd: int, want: np.ndarray, target: np.ndarray, side: int | None
    ):
        # one group per GOP that holds wanted frames: bytes from its keyframe to its last wanted frame
        # unwanted samples: only the bytes later frames depend on (``keep``), left out if none
        key_of = v.keys[np.searchsorted(v.keys, want, side="right") - 1]
        starts, first = np.unique(key_of, return_index=True)
        ends = np.maximum.reduceat(want, first) + 1
        groups = [np.arange(a, e, dtype=np.int32) for a, e in zip(starts, ends)]
        size = v.size
        if self.skip and v.keep is not None:
            size = np.where(np.isin(np.arange(v.n), want), v.size, v.keep)
            groups = [g[size[g] > 0] for g in groups]
        pk = np.concatenate(groups)
        base = np.repeat(v.off[starts], [len(g) for g in groups])
        arrays = dict(
            group_off=np.ascontiguousarray(v.off[starts], np.int64),
            group_len=np.ascontiguousarray(
                v.off[ends - 1] + v.size[ends - 1] - v.off[starts], np.int64
            ),
            group_npk=np.array([len(g) for g in groups], np.int32),
            pk_off=np.ascontiguousarray(v.off[pk] - base, np.int64),
            pk_len=np.ascontiguousarray(size[pk], np.int32),
            pk_idx=pk,
            want=want,
            prefix=np.frombuffer(v.prefix, np.uint8) if v.prefix else None,
        )
        a, rng = self.augment, self.rng
        if self.mode == "yuv":  # stored-resolution window
            nh, nw, oh, ow = v.h, v.w, side, side
        else:  # short side resized to size (even for 4:2:0 output), size x size crop
            scale = self.size / min(v.h, v.w)
            nh, nw = (max(self.size, round(x * scale)) for x in (v.h, v.w))
            oh = ow = self.size
        if a.crop == "random":
            top, left = rng.randint(0, nh - oh), rng.randint(0, nw - ow)
        else:
            top, left = (nh - oh) // 2, (nw - ow) // 2
        if (
            self.mode == "yuv"
        ):  # the window is copied: its chroma must start on a chroma sample
            top, left = top & ~1, left & ~1
        hflip, vflip = rng.random() < a.hflip, rng.random() < a.vflip
        r = _native.Request(
            fd=fd,
            codec=_native.CODECS[v.codec],
            mode=_native.MODES[self.mode],
            ngroups=len(starts),
            group_off=arrays["group_off"].ctypes.data,
            group_len=arrays["group_len"].ctypes.data,
            group_npk=arrays["group_npk"].ctypes.data,
            pk_off=arrays["pk_off"].ctypes.data,
            pk_len=arrays["pk_len"].ctypes.data,
            pk_idx=arrays["pk_idx"].ctypes.data,
            prefix=(
                arrays["prefix"].ctypes.data if arrays["prefix"] is not None else None
            ),
            nprefix=len(v.prefix),
            nwant=len(want),
            want=want.ctypes.data,
            nh=nh,
            nw=nw,
            top=top,
            left=left,
            oh=oh,
            ow=ow,
            hflip=int(hflip),
            vflip=int(vflip),
            out=target.ctypes.data,
        )
        arrays["target"] = target
        return r, arrays
