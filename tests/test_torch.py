"""ClipDataset: the same batches with any number of DataLoader workers, exact resumption,
epoch mode covering every video once."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from kohakuclip.torch import ClipDataset, ClipLoader

from .test_reader import sources  # noqa: F401  (module fixture)


@pytest.fixture(scope="module")
def shards(sources, tmp_path_factory):  # noqa: F811
    from kohakuclip.writer import Encoding, write

    out = tmp_path_factory.mktemp("torch")
    return write(sources * 2, str(out), Encoding(crf=30), workers=3)


def batches(dataset, n, workers=0):
    loader = DataLoader(dataset, batch_size=None, num_workers=workers)
    out = []
    for video in loader:
        out.append(video.clone())
        if len(out) == n:
            return out
    return out


def make(shards, **kwargs):
    defaults = {
        "batch_size": 2,
        "frames": 4,
        "fps": 6.0,
        "size": 64,
        "threads": 2,
        "hflip": 0.5,
        "seed": 7,
    }
    return ClipDataset(shards, **{**defaults, **kwargs})


def test_same_batches_for_any_worker_count(shards):
    direct = batches(make(shards), 6)
    two = batches(make(shards), 6, workers=2)
    three = batches(make(shards), 6, workers=3)
    for a, b, c in zip(direct, two, three):
        assert torch.equal(a, b) and torch.equal(a, c)


def test_exact_resume(shards):
    full = batches(make(shards), 8)

    first = make(shards)
    it = iter(first)
    for _ in range(3):
        next(it)
    state = first.state_dict()
    assert state["start"] == 3

    resumed = make(shards, seed=0)
    resumed.load_state_dict(state)
    rest = batches(resumed, 5, workers=2)
    for a, b in zip(full[3:], rest):
        assert torch.equal(a, b)


def test_epochs_cover_every_video_once(shards):
    dataset = make(shards, epochs=True, batch_size=2)
    n = len(dataset._reader())
    per_epoch = n // 2
    seen = []
    for index in range(per_epoch):
        seen += dataset._videos(0, 1, index)
    assert sorted(seen) == list(range(n))
    assert dataset._videos(0, 1, per_epoch) != dataset._videos(
        0, 1, 0
    )  # next epoch reshuffles
    assert np.array_equal(
        sorted(dataset._videos(0, 1, 0)), sorted(dataset._videos(0, 1, 0))
    )


@pytest.mark.parametrize("workers", [0, 2])
def test_loader_state(shards, workers):
    full = batches(make(shards), 7)

    loader = ClipLoader(make(shards), workers=workers)
    it = iter(loader)
    for _ in range(4):
        next(it)
    state = loader.state_dict()

    resumed = ClipLoader(make(shards, seed=1), workers=workers)
    resumed.load_state_dict(state)
    rest = [v for _, v in zip(range(3), resumed)]
    for a, b in zip(full[4:], rest):
        assert torch.equal(a, b)
