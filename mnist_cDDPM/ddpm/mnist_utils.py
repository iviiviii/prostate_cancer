#!/usr/bin/env python3
"""Utility helpers for MNIST row-shift conditional DDPM.

Provides:
- set_seed
- normalization helpers
- row-wise shift augmentation (no smoothing)
- RowShiftMNIST dataset
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


# ----------------------------
# Repro / normalization
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_ddpm_range(x01: torch.Tensor) -> torch.Tensor:
    """[0,1] -> [-1,1]"""
    return x01 * 2.0 - 1.0


def to_01_range(x11: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1]"""
    return x11.add(1.0).mul(0.5).clamp(0.0, 1.0)


# ----------------------------
# Row-wise shift augmentation
# ----------------------------
def _sample_row_offsets(h: int, max_shift_px: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Uniform integer offsets in [-max_shift_px, max_shift_px] for each row."""
    if max_shift_px <= 0:
        return torch.zeros(h, dtype=torch.float32)
    m = int(max_shift_px)
    return torch.randint(-m, m + 1, (h,), dtype=torch.float32, generator=generator)


def apply_row_shift_no_smooth(
    img: torch.Tensor,  # (C,H,W) in [0,1]
    offsets_px: torch.Tensor,  # (H,) integer px
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Row-wise horizontal shift via grid_sample, no smoothing."""
    _, H, W = img.shape

    ys = torch.arange(H, device=img.device, dtype=torch.float32)
    xs = torch.arange(W, device=img.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    grid_x = grid_x + offsets_px.view(H, 1)

    grid_x_norm = 2.0 * grid_x / max(1, (W - 1)) - 1.0
    grid_y_norm = 2.0 * grid_y / max(1, (H - 1)) - 1.0
    grid = torch.stack([grid_x_norm, grid_y_norm], dim=-1)  # (H,W,2)

    shifted = F.grid_sample(
        img.unsqueeze(0),
        grid.unsqueeze(0),
        mode="nearest",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return shifted.squeeze(0)


# ----------------------------
# Dataset
# ----------------------------
class RowShiftMNIST(Dataset):
    """Returns (cond, target) where
    - target: clean MNIST in [-1,1]
    - cond: row-shifted MNIST in [-1,1] (no smoothing)
    Row offsets are deterministic per index via a stored seed for reproducibility.
    """

    def __init__(
        self,
        mnist_root: Path,
        split: str = "train",
        num_samples: Optional[int] = None,
        max_shift_px: int = 6,
        padding_mode: str = "zeros",
        seed: int = 42,
        indices: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.max_shift_px = int(max_shift_px)
        self.padding_mode = padding_mode

        ds = MNIST(
            root=str(mnist_root),
            train=(split == "train"),
            download=True,
            transform=transforms.ToTensor(),
        )

        total = len(ds)
        rng = np.random.default_rng(seed)

        if indices is None:
            n = total if num_samples is None else min(total, int(num_samples))
            indices = rng.choice(total, size=n, replace=False)
        else:
            indices = np.array(indices, dtype=int)
            if num_samples is not None:
                indices = indices[: int(num_samples)]

        # seed per sample -> deterministic offsets per index
        self.offset_seeds = rng.integers(low=0, high=2**31 - 1, size=len(indices), dtype=np.int64)
        self.data = [ds[int(i)] for i in indices]  # list of (PIL->Tensor, label)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img01, _ = self.data[idx]  # (1,28,28) in [0,1]

        g = torch.Generator()
        g.manual_seed(int(self.offset_seeds[idx]))
        offsets = _sample_row_offsets(img01.shape[1], self.max_shift_px, generator=g)

        shifted = apply_row_shift_no_smooth(
            img=img01,
            offsets_px=offsets,
            padding_mode=self.padding_mode,
            align_corners=True,
        ).clamp(0.0, 1.0)

        target = to_ddpm_range(img01)
        cond = to_ddpm_range(shifted)
        return cond, target


def build_dataloader(
    mnist_root: Path,
    split: str,
    batch_size: int,
    num_samples: Optional[int],
    max_shift_px: int,
    padding_mode: str,
    seed: int,
    num_workers: int = 0,
    indices: Optional[Sequence[int]] = None,
) -> DataLoader:
    ds = RowShiftMNIST(
        mnist_root=mnist_root,
        split=split,
        num_samples=num_samples,
        max_shift_px=max_shift_px,
        padding_mode=padding_mode,
        seed=seed,
        indices=indices,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)


def _split_indices(total: int, ratios: Sequence[float], seed: int) -> tuple[list[int], list[int], list[int]]:
    ratios = [float(r) for r in ratios]
    s = sum(ratios)
    ratios = [r / s for r in ratios]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)

    sizes = [int(total * r) for r in ratios[:-1]]
    used = sum(sizes)
    sizes.append(total - used)

    t_size, v_size, te_size = sizes
    train_idx = perm[:t_size].tolist()
    val_idx = perm[t_size : t_size + v_size].tolist()
    test_idx = perm[t_size + v_size : t_size + v_size + te_size].tolist()
    return train_idx, val_idx, test_idx


def build_threeway_dataloaders(
    mnist_root: Path,
    batch_size: int,
    ratios: Optional[Sequence[float]],
    split_seed: int,
    num_samples: Optional[int],
    max_shift_px: int,
    padding_mode: str,
    seed: int,
    num_workers: int = 0,
    use_mnist_split: str = "train",
    no_split: bool = False,
) -> tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Create train/val/test dataloaders from MNIST.

    - If no_split=True: returns a single loader covering the whole chosen MNIST split
      (train -> train_loader, test -> test_loader), val=None, the other None. ratios are ignored.
    - Otherwise: splits according to ratios; zero-sized splits return None.
    """
    dummy_ds = MNIST(
        root=str(mnist_root),
        train=(use_mnist_split == "train"),
        download=True,
        transform=transforms.ToTensor(),
    )
    total = len(dummy_ds)
    n = total if num_samples is None else min(total, int(num_samples))

    if no_split:
        indices = list(range(n))
        loader = build_dataloader(
            mnist_root=mnist_root,
            split=use_mnist_split,
            batch_size=batch_size,
            num_samples=None,
            max_shift_px=max_shift_px,
            padding_mode=padding_mode,
            seed=seed,
            num_workers=num_workers,
            indices=indices,
        )
        if use_mnist_split == "train":
            return loader, None, None
        else:
            return None, None, loader

    if ratios is None:
        raise ValueError("ratios must be provided when no_split=False")

    train_idx, val_idx, test_idx = _split_indices(n, ratios, split_seed)

    def _maybe_loader(indices, offset_seed):
        if len(indices) == 0:
            return None
        return build_dataloader(
            mnist_root=mnist_root,
            split=use_mnist_split,
            batch_size=batch_size,
            num_samples=None,
            max_shift_px=max_shift_px,
            padding_mode=padding_mode,
            seed=offset_seed,
            num_workers=num_workers,
            indices=indices,
        )

    train_loader = _maybe_loader(train_idx, seed)
    val_loader = _maybe_loader(val_idx, seed + 1)
    test_loader = _maybe_loader(test_idx, seed + 2)
    return train_loader, val_loader, test_loader
