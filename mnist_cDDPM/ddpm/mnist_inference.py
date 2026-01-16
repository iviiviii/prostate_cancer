#!/usr/bin/env python3
"""
Inference helpers for MNIST conditional DDPM.

python infer_cDDPM0114.py \
  --ckpt-path runs_cDDPM0114/train1/ckpt_step050000.pt \
  --output-dir runs_cDDPM0114/train1/infer \
  --num-samples 60000 --split-ratios 0.7,0.0,0.3 --split-seed 42 \
  --batch-size 128 --max-shift-px 6 --padding-mode zeros \
  --use-mnist-split test
  
"""

from __future__ import annotations

from pathlib import Path
import torch

from .mnist_model import ConditionalUNet
from .mnist_scheduler import NoiseScheduler


def load_model_and_sched(ckpt_path: Path, device: torch.device) -> tuple[ConditionalUNet, NoiseScheduler, dict]:
    # use str(ckpt_path) to avoid torch serialization Path restriction
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    args = ckpt.get("args", {})

    model = ConditionalUNet(
        in_channels=2,
        base_channels=args.get("base_channels", 64),
        time_dim=args.get("time_dim", 128),
        out_channels=1,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diff = ckpt.get("diffusion", {})
    sched = NoiseScheduler(
        num_steps=diff.get("num_steps", 1000),
        beta_start=diff.get("beta_start", 1e-4),
        beta_end=diff.get("beta_end", 0.02),
    )
    return model, sched, args


@torch.no_grad()
def sample_from_cond(ckpt_path: Path, cond: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Generate clean prediction given cond batch in [-1,1]."""
    model, sched, _ = load_model_and_sched(ckpt_path, device)
    cond = cond.to(device)
    return sched.sample(model, cond, device=device)
