"""Lightning on KohakuClip shards: ``ClipLoader`` as the train loader. Lightning stores its
``state_dict()`` in every checkpoint and restores it on ``ckpt_path=...``, so a resumed run
sees exactly the batches the original run would have seen next (``lookahead=1``: Lightning
fetches one batch ahead of the one it trains on). Under DDP each rank draws its
own batches (ranks come from torch.distributed).

    python examples/lightning_module.py SHARD_DIR [--workers 0] [--devices 2] [--resume last.ckpt]
"""

import argparse
import glob
import os

import lightning.pytorch as pl
import torch
from torch import nn
from torch_loop import TinyVideoNet

from kohakuclip.torch import ClipDataset, ClipLoader


class NextFrameColor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = TinyVideoNet()

    def training_step(self, video, batch_idx):
        video = video.float() / 127.5 - 1
        clip, target = video[:, :-1], video[:, -1].mean(dim=(2, 3))
        loss = nn.functional.mse_loss(self.model(clip), target)
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, fused=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shards")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--resume")
    a = ap.parse_args()

    dataset = ClipDataset(
        sorted(glob.glob(os.path.join(a.shards, "*.zip"))),
        batch_size=16,
        frames=8,
        fps=6.0,
        size=128,
        threads=8,
        hflip=0.5,
        seed=0,
    )
    trainer = pl.Trainer(
        devices=a.devices,
        max_steps=a.steps,
        precision="bf16-mixed",
        use_distributed_sampler=False,  # ClipDataset splits by rank itself
        enable_checkpointing=True,
    )
    trainer.fit(
        NextFrameColor(),
        train_dataloaders=ClipLoader(dataset, workers=a.workers, lookahead=1),
        ckpt_path=a.resume,
    )


if __name__ == "__main__":
    main()
