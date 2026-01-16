#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import torch

from .mnist_utils import RowShiftMNIST
from .mnist_viz import save_pair_grid


def parse_args():
    p = argparse.ArgumentParser("Precompute MNIST row-shift cond/target tensors")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mnist-root", type=Path, default=Path.home() / ".cache" / "mnist")
    p.add_argument("--split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--num-samples", type=int, default=60000)
    p.add_argument("--max-shift-px", type=int, default=6)
    p.add_argument("--padding-mode", type=str, default="zeros", choices=["zeros", "border", "reflection"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-grid", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grid-n", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ds = RowShiftMNIST(
        mnist_root=args.mnist_root,
        split=args.split,
        num_samples=args.num_samples,
        max_shift_px=args.max_shift_px,
        padding_mode=args.padding_mode,
        seed=args.seed,
    )

    conds, targets = [], []
    for cond, tgt in ds:
        conds.append(cond)
        targets.append(tgt)

    cond_tensor = torch.stack(conds, dim=0)
    target_tensor = torch.stack(targets, dim=0)

    out_path = args.output_dir / f"mnist_rowshift_{args.split}.pt"
    torch.save(
        {
            "cond": cond_tensor,
            "target": target_tensor,
            "config": vars(args),
        },
        out_path,
    )
    print(f"[save] {out_path} (shape={tuple(cond_tensor.shape)})")

    if args.save_grid:
        n = min(int(args.grid_n), cond_tensor.shape[0])
        save_pair_grid(cond_tensor[:n], target_tensor[:n], args.output_dir / "preview.png", nrow=min(4, n))


if __name__ == "__main__":
    main()
