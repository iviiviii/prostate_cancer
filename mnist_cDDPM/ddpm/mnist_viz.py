#!/usr/bin/env python3
"""Visualization helpers for MNIST conditional DDPM."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt

from .mnist_utils import to_01_range


def _save_with_dpi(grid: torch.Tensor, path: Path, dpi: int) -> None:
    """Save a (C,H,W) grid tensor with specified dpi via matplotlib."""
    np_grid = grid.cpu().numpy()
    np_grid = np.transpose(np_grid, (1, 2, 0))  # C,H,W -> H,W,C
    plt.figure(figsize=(np_grid.shape[1] / dpi, np_grid.shape[0] / dpi), dpi=dpi)
    plt.imshow(np_grid, vmin=0, vmax=1, cmap="gray" if np_grid.shape[2] == 1 else None)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_triplet_grid(cond: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, path: Path, *, max_rows: int = 6, dpi: int = 600) -> None:
    """Save grid with 3 columns per sample (cond, pred, target), up to max_rows samples."""
    cond01 = to_01_range(cond.cpu())
    pred01 = to_01_range(pred.cpu())
    target01 = to_01_range(target.cpu())
    
    n = min(max_rows, cond01.shape[0])
    imgs = []
    for c, p, t in zip(cond01[:n], pred01[:n], target01[:n]):
        imgs.extend([c, p, t])  # order: cond, pred, target

    grid = make_grid(imgs, nrow=3, padding=2)
    _save_with_dpi(grid, path, dpi=dpi)


def save_pair_grid(a: torch.Tensor, b: torch.Tensor, path: Path, *, nrow: int = 2, dpi: int = 600) -> None:
    """Save grid with two-column comparison (a | b)."""
    a01 = to_01_range(a.cpu())
    b01 = to_01_range(b.cpu())
    imgs = []
    for x, y in zip(a01, b01):
        imgs.extend([x, y])
    grid = make_grid(imgs, nrow=nrow * 2 if nrow > 1 else 2, padding=2)
    _save_with_dpi(grid, path, dpi=dpi)


def save_single_grid(x: torch.Tensor, path: Path, *, nrow: int = 8, dpi: int = 600) -> None:
    """Save grid for a batch of images already in [-1,1]."""
    x01 = to_01_range(x.cpu())
    grid = make_grid(x01, nrow=nrow, padding=2)
    _save_with_dpi(grid, path, dpi=dpi)
