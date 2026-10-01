"""A plain PyTorch training loop on KohakuClip shards, with exact resumption.

    python examples/torch_loop.py SHARD_DIR [--workers 0] [--resume ckpt.pt]

The loader state (seed and batches consumed) goes into the checkpoint next to the model; a
resumed run continues with exactly the batches the original run would have seen next.
"""

import argparse
import glob
import os

import torch
from torch import nn

from kohakuclip.torch import ClipDataset, ClipLoader


class TinyVideoNet(nn.Module):
    """A stand-in model: 3D convolutions, predicts the next frame's mean color."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 32, 3, stride=(1, 2, 2), padding=1),
            nn.GELU(),
            nn.Conv3d(32, 64, 3, stride=(1, 2, 2), padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, 3),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: [B, T, 3, H, W] in [-1, 1] -> [B, 3, T, H, W]
        return self.net(video.transpose(1, 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shards")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--resume")
    ap.add_argument("--save", default="ckpt.pt")
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
    loader = ClipLoader(dataset, workers=a.workers)

    model = TinyVideoNet().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    step = 0
    if a.resume:
        state = torch.load(a.resume)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        loader.load_state_dict(state["loader"])
        step = state["step"]

    for video in loader:
        video = video.cuda(non_blocking=True).float() / 127.5 - 1
        clip, target = video[:, :-1], video[:, -1].mean(dim=(2, 3))
        loss = nn.functional.mse_loss(model(clip), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step += 1
        if step % 50 == 0:
            print(f"step {step}: loss {loss.item():.4f}")
        if step == a.steps:
            break

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loader": loader.state_dict(),
            "step": step,
        },
        a.save,
    )


if __name__ == "__main__":
    main()
