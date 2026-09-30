"""PyTorch side: a background loader (decode overlaps training) and the GPU half of ``mode="yuv"``.

    loader = Loader(reader, sample, batch_size=16, prefetch=2)   # sample(rng) -> (video id, frames)
    for video in loader:                                         # uint8 [B, T, 3, S, S], pinned
        video = video.cuda(non_blocking=True)

The native decode releases the GIL, so one background thread with ``Reader(threads=N)`` feeds a GPU
without DataLoader worker processes, pickling or collation.
"""

import collections
import queue
import random
import threading

import numpy as np
import torch
import torch.nn.functional as F

from .reader import Batch, Reader

# YUV (limited range) -> RGB, per colorspace: (Kr, Kb)
_KRKB = {"bt709": (0.2126, 0.0722), "bt601": (0.299, 0.114)}


def yuv_to_rgb(batch: Batch, size: int, device: torch.device | str = "cuda") -> torch.Tensor:
    """``Reader(mode="yuv")`` output -> float RGB in [-1, 1], [B, T, 3, size, size], on ``device``:
    chroma upsampling, limited-range conversion, antialiased bilinear resize to ``size``, flips."""
    b, t, _ = batch.video.shape
    h, w = (int(v) for v in batch.window[0])
    x = torch.as_tensor(batch.video).to(device, non_blocking=True).view(b * t, -1)
    y = x[:, :h * w].view(-1, 1, h, w).float()
    uv = x[:, h * w:].view(-1, 2, h // 2, w // 2).float()
    uv = F.interpolate(uv, size=(h, w), mode="bilinear", align_corners=False)
    kr, kb = (torch.tensor([_KRKB[c][i] for c in batch.colorspace], device=device).repeat_interleave(t).view(-1, 1, 1, 1)
              for i in (0, 1))
    kg = 1 - kr - kb
    yy = (y - 16) / 219
    u, v = (uv[:, :1] - 128) / 224, (uv[:, 1:] - 128) / 224
    r = yy + 2 * (1 - kr) * v
    bb = yy + 2 * (1 - kb) * u
    g = (yy - kr * r - kb * bb) / kg
    rgb = torch.cat([r, g, bb], 1).clamp_(0, 1)
    rgb = F.interpolate(rgb, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    rgb = rgb.view(b, t, 3, size, size)
    flips = torch.as_tensor(batch.flips, device=device).bool()
    rgb = torch.where(flips[:, 0].view(b, 1, 1, 1, 1), rgb.flip(-1), rgb)
    rgb = torch.where(flips[:, 1].view(b, 1, 1, 1, 1), rgb.flip(-2), rgb)
    return rgb * 2 - 1


class Loader:
    """Iterates batches decoded by ``reader`` on a background thread: ``inflight`` batches queued on
    the native pool at once (planning overlaps decoding, no per-batch barrier), ``prefetch`` finished
    batches buffered.

    ``sample(rng) -> (video id, frame indices)`` chooses each clip. RGB batches come back as pinned
    uint8 tensors; YUV batches as ``Batch`` objects for ``yuv_to_rgb``.
    """

    def __init__(self, reader: Reader, sample, batch_size: int, prefetch: int = 2, inflight: int = 2,
                 steps: int | None = None, seed: int = 0, pin: bool = True):
        self.reader, self.sample, self.batch_size = reader, sample, batch_size
        self.steps, self.pin, self.inflight = steps, pin, inflight
        self.rng = random.Random(seed)
        self.queue: queue.Queue = queue.Queue(maxsize=prefetch)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _buffer(self, t: int) -> np.ndarray | None:
        if self.reader.mode != "rgb":
            return None
        s = self.reader.size
        buf = torch.empty((self.batch_size, t, 3, s, s), dtype=torch.uint8, pin_memory=self.pin)
        return buf

    def _run(self):
        pending = collections.deque()  # batches in flight on the native pool
        step = 0
        try:
            while self.steps is None or step < self.steps or pending:
                if self.steps is None or step < self.steps:
                    items = [self.sample(self.rng) for _ in range(self.batch_size)]
                    buf = self._buffer(len(items[0][1]))
                    pending.append((self.reader.submit(items, None if buf is None else buf.numpy()), buf))
                    step += 1
                if len(pending) > self.inflight or (self.steps is not None and step >= self.steps):
                    job, buf = pending.popleft()
                    batch = job.result()
                    self.queue.put(buf if buf is not None else batch)
        except Exception as e:  # surface decode errors in the consumer
            self.queue.put(e)
            return
        self.queue.put(None)

    def __iter__(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item
