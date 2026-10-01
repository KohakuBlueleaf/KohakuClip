"""PyTorch side: ``ClipDataset`` (batches of clips as an ``IterableDataset``) and the GPU half
of the yuv modes.

    dataset = ClipDataset(sources, batch_size=16, frames=8, fps=6.0, threads=8)
    for video in dataset:  # uint8 [B, T, 3, S, S], pinned
        video = video.cuda(non_blocking=True)

or ``DataLoader(dataset, batch_size=None, num_workers=N, pin_memory=True)``.
"""

import collections
import hashlib
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from ._core import Batch, Reader

# YUV (limited range) -> RGB, per colorspace: (Kr, Kb)
_KRKB = {"bt709": (0.2126, 0.0722), "bt601": (0.299, 0.114)}


def yuv_to_rgb(
    batch: Batch,
    size: int,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """A yuv-mode batch -> RGB in [-1, 1], [B, T, 3, size, size], on ``device``: chroma
    upsampling, limited-range conversion, antialiased bilinear resize to ``size`` (``"yuv"``
    mode), flips. One Triton kernel on CUDA devices, plain torch otherwise."""
    if batch.window is None or batch.colorspace is None or batch.flips is None:
        raise ValueError("not a yuv-mode batch")
    device = torch.device(device)
    video = torch.as_tensor(batch.video).to(device, non_blocking=True)
    if device.type == "cuda":
        from .gpu import yuv_to_rgb_triton

        window = (int(batch.window[0][0]), int(batch.window[0][1]))
        flips = torch.as_tensor(batch.flips)
        return yuv_to_rgb_triton(video, window, batch.colorspace, flips, size, dtype)
    return yuv_to_rgb_torch(video, batch, size).to(dtype)


def yuv_to_rgb_torch(video: torch.Tensor, batch: Batch, size: int) -> torch.Tensor:
    """The reference implementation of ``yuv_to_rgb`` in plain torch (float32)."""
    if batch.window is None or batch.colorspace is None or batch.flips is None:
        raise ValueError("not a yuv-mode batch")
    device = video.device
    b, t, _ = video.shape
    h, w = (int(v) for v in batch.window[0])
    x = video.view(b * t, -1)
    y = x[:, : h * w].view(-1, 1, h, w).float()
    uv = x[:, h * w :].view(-1, 2, h // 2, w // 2).float()
    uv = F.interpolate(uv, size=(h, w), mode="bilinear", align_corners=False)

    kr, kb = (
        torch.tensor([_KRKB[c][i] for c in batch.colorspace], device=device)
        .repeat_interleave(t)
        .view(-1, 1, 1, 1)
        for i in (0, 1)
    )
    kg = 1 - kr - kb
    yy = (y - 16) / 219
    u, v = (uv[:, :1] - 128) / 224, (uv[:, 1:] - 128) / 224
    r = yy + 2 * (1 - kr) * v
    bb = yy + 2 * (1 - kb) * u
    g = (yy - kr * r - kb * bb) / kg
    rgb = torch.cat([r, g, bb], 1).clamp_(0, 1)

    # "yuv": the stored-resolution window; "yuv_resized": already the output size
    if (h, w) != (size, size):
        rgb = F.interpolate(
            rgb, size=(size, size), mode="bilinear", antialias=True, align_corners=False
        )
    rgb = rgb.view(b, t, 3, size, size)

    flips = torch.as_tensor(batch.flips, device=device).bool()
    rgb = torch.where(flips[:, 0].view(b, 1, 1, 1, 1), rgb.flip(-1), rgb)
    rgb = torch.where(flips[:, 1].view(b, 1, 1, 1, 1), rgb.flip(-2), rgb)
    return rgb * 2 - 1


def batch_seed(seed: int, rank: int, index: int, stream: str) -> int:
    """A 64-bit seed for one batch of one rank (``stream``: "sample" or "augment")."""
    key = f"{seed}:{rank}:{index}:{stream}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


def distributed_rank() -> tuple[int, int]:
    """(rank, world size) of this process: torch.distributed if initialized, else the
    RANK / WORLD_SIZE variables of the launcher, else (0, 1)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


class ClipDataset(IterableDataset):
    """An endless stream of decoded batches of clips (one item = one batch).

    Iterate it directly (decodes in this process, into pinned memory, ``inflight`` batches
    queued on the native pool ahead of the consumer), or through
    ``DataLoader(dataset, batch_size=None, num_workers=N, pin_memory=True)``.

    Clips: ``frames`` frames resampled to ``fps`` (None: native rate) from a random start, or
    ``frames`` random frames of the whole video (``spread``). Videos: drawn at random
    (``epochs=False``), or every video once per epoch in a seeded order, split over ranks
    (``epochs=True``; the last partial batch of an epoch is dropped).

    Batch ``i`` of a rank is a pure function of (``seed``, rank, ``i``): the same whatever the
    number of DataLoader workers, and a run resumed with ``start=i`` (``state_dict()`` /
    ``load_state_dict()``, or the trainer's step count) continues with exactly the batches it
    would have seen. Ranks come from torch.distributed (or RANK / WORLD_SIZE).

    RGB batches are uint8 tensors [B, T, 3, S, S]; yuv modes yield ``Batch`` for ``yuv_to_rgb``.
    """

    def __init__(
        self,
        sources: list[str],
        batch_size: int,
        frames: int = 8,
        fps: float | None = None,
        spread: bool = False,
        epochs: bool = False,
        seed: int = 0,
        start: int = 0,
        inflight: int = 2,
        pin: bool | None = None,
        **reader: object,
    ):
        self.sources, self.batch_size = sources, batch_size
        self.frames, self.fps, self.spread = frames, fps, spread
        self.epochs, self.seed, self.start = epochs, seed, start
        self.inflight, self.pin = inflight, pin
        self.reader_args = (
            reader  # Reader keyword arguments: size, mode, threads, crop, ...
        )
        self.reader: Reader | None = None
        self.consumed = 0  # batches handed out by this process (direct iteration)

    # ------------------------------------------------------------------ resumption
    def state_dict(self) -> dict:
        """Where a run is: resume with ``load_state_dict`` (or ``start=``)."""
        return {"seed": self.seed, "start": self.start + self.consumed}

    def load_state_dict(self, state: dict) -> None:
        self.seed, self.start, self.consumed = state["seed"], state["start"], 0

    # ------------------------------------------------------------------ batches
    def _reader(self) -> Reader:
        if self.reader is None:
            self.reader = Reader(self.sources, **self.reader_args)  # type: ignore[arg-type]
        return self.reader

    def _videos(self, rank: int, world: int, index: int) -> list[int] | None:
        """Epoch mode: the videos of batch ``index`` of ``rank``."""
        if not self.epochs:
            return None
        n = len(self._reader())
        per_rank = n // world // self.batch_size  # batches per rank per epoch
        if per_rank == 0:
            raise ValueError(f"{n} videos make no full batch per rank per epoch")
        epoch, position = divmod(index, per_rank)
        order = torch.randperm(
            n, generator=torch.Generator().manual_seed(self.seed + epoch)
        )
        mine = order[rank::world]
        return mine[
            position * self.batch_size : (position + 1) * self.batch_size
        ].tolist()

    def _submit(self, rank: int, world: int, index: int, pin: bool):
        reader = self._reader()
        items = reader.sample(
            self.batch_size,
            self.frames,
            self.fps,
            self.spread,
            seed=batch_seed(self.seed, rank, index, "sample"),
            videos=self._videos(rank, world, index),
        )
        out = None
        if reader.mode == "rgb":
            s = reader.size
            shape = (self.batch_size, self.frames, 3, s, s)
            out = torch.empty(shape, dtype=torch.uint8, pin_memory=pin)
        augment = batch_seed(self.seed, rank, index, "augment")
        pending = reader.submit(
            items, None if out is None else out.numpy(), seed=augment
        )
        return pending, out

    def __iter__(self):
        rank, world = distributed_rank()
        info = get_worker_info()
        worker, workers = (0, 1) if info is None else (info.id, info.num_workers)
        # pinned memory only in the process that copies to the GPU
        pin = (
            (info is None and torch.cuda.is_available())
            if self.pin is None
            else self.pin
        )

        # this worker's batches: DataLoader takes them round-robin from its workers
        first = self.start + worker
        indices = iter(range(first, 1 << 62, workers))

        pending: collections.deque = collections.deque()
        while True:
            while len(pending) <= self.inflight:
                pending.append(self._submit(rank, world, next(indices), pin))
            job, out = pending.popleft()
            batch = job.result()
            if info is None:
                self.consumed += 1
            yield out if out is not None else batch


class ClipLoader:
    """Iterates a ``ClipDataset`` in this process (``workers=0``) or through a ``DataLoader``
    with ``workers`` worker processes (pinning in this process), and counts the batches it
    handed out: ``state_dict()`` / ``load_state_dict()`` resume exactly.

    ``lookahead``: batches the consumer fetches ahead of the one it trains on, left out of the
    saved state. Lightning's training loop fetches one batch ahead (``lookahead=1``); Lightning
    saves and restores the state of a train loader that has these methods in its checkpoints.
    """

    def __init__(
        self,
        dataset: ClipDataset,
        workers: int = 0,
        prefetch: int = 2,
        lookahead: int = 0,
    ):
        self.dataset, self.workers, self.prefetch = dataset, workers, prefetch
        self.lookahead = lookahead
        self.consumed = 0

    def state_dict(self) -> dict:
        used = max(0, self.consumed - self.lookahead)
        return {"seed": self.dataset.seed, "start": self.dataset.start + used}

    def load_state_dict(self, state: dict) -> None:
        self.dataset.load_state_dict(state)
        self.consumed = 0

    def __iter__(self):
        if self.workers == 0:
            batches = iter(self.dataset)
        else:
            batches = iter(
                DataLoader(
                    self.dataset,
                    batch_size=None,
                    num_workers=self.workers,
                    pin_memory=torch.cuda.is_available(),
                    prefetch_factor=self.prefetch,
                )
            )
        for batch in batches:
            self.consumed += 1
            yield batch
