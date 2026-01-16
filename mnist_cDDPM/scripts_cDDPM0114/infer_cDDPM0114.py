#!/usr/bin/env python3
"""
Unified evaluation script for Conditional DDPM.
Handles both quick sampling (visualization) and full metric computation (MSE/SSIM).

python -m scripts_cDDPM0114.infer_cDDPM0114 \
  --ckpt-paths runs_cDDPM0114/train_full200000/ckpt_step040000.pt runs_cDDPM0114/train_full200000/ckpt_step100000.pt runs_cDDPM0114/train_full200000/ckpt_step140000.pt runs_cDDPM0114/train_full200000/model_last.pt \
  --output-dir runs_cDDPM0114/train_full200000/infer \
  --mode metrics \
  --use-mnist-split test \
  --save-ssim-map
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np 

from ddpm.mnist_model import ConditionalUNet
from ddpm.mnist_scheduler import NoiseScheduler
from ddpm.mnist_utils import build_threeway_dataloaders, to_01_range
from ddpm.mnist_viz import save_triplet_grid

def parse_args():
    p = argparse.ArgumentParser("Unified Evaluation for cDDPM")
    
    # Essential Inputs
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--ckpt-path", type=Path, help="Path to single .pt checkpoint")
    group.add_argument("--ckpt-paths", type=Path, nargs="+", help="Paths to multiple checkpoints (looped)")
    p.add_argument("--mode", type=str, default="sample", choices=["sample", "metrics"], 
                   help="'sample': gen images for visual check, 'metrics': calc MSE/SSIM on full set")
    
    # Data Settings (Should match training config mostly)
    p.add_argument("--mnist-root", type=Path, default=Path.home() / ".cache/mnist")
    p.add_argument("--num-samples", type=int, default=None, help="Optional cap; None uses full split length")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-shift-px", type=int, default=6)
    p.add_argument("--padding-mode", type=str, default="zeros", choices=["zeros", "border", "reflection"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--use-mnist-split", type=str, default="test", choices=["train", "test", "val"])
    
    # Output & Visualization
    p.add_argument("--output-dir", type=Path, default=None, help="If None, saves to ckpt_dir/eval_results (or subfolders for multiple ckpts)")
    p.add_argument("--sample-count", type=int, default=10, help="Number of cond/pred/target triplets to visualize")
    p.add_argument("--sample-max-batches", type=int, default=4, help="Max batches to draw from when collecting samples")
    
    # SSIM
    p.add_argument("--save-ssim-map", action="store_true", help="If set, save SSIM map images for worst/best samples (requires skimage full=True).",)
    p.add_argument("--ssim-map-limit", type=int, default=1, help="How many best/worst SSIM maps to save. default=1",)

    return p.parse_args()

def load_model_and_sched(ckpt_path: Path, device: torch.device):
    """Loads model and scheduler from checkpoint."""
    print(f"[Info] Loading checkpoint from {ckpt_path}...")

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
    return model, sched

def run_sampling(args, model, sched, loader, device, out_dir):
    """Quick visual check: collects up to sample-count images across a few batches and saves a grid."""
    print("[Mode] Sampling: Generating visualization grid...")

    cond_list, x0_list, pred_list = [], [], []

    it = iter(loader)
    for _ in range(max(1, args.sample_max_batches)):
        try:
            cond, x0 = next(it)
        except StopIteration:
            break

        cond = cond.to(device)
        x0 = x0.to(device)
        with torch.no_grad():
            pred = sched.sample(model, cond, device=device)

        cond_list.append(cond.cpu())
        x0_list.append(x0.cpu())
        pred_list.append(pred.cpu())

        if sum(t.shape[0] for t in cond_list) >= args.sample_count:
            break

    if len(cond_list) == 0:
        print("[Error] DataLoader is empty.")
        return

    cond_all = torch.cat(cond_list, dim=0)
    x0_all = torch.cat(x0_list, dim=0)
    pred_all = torch.cat(pred_list, dim=0)

    n = min(args.sample_count, cond_all.shape[0])
    save_path = out_dir / f"sample_{args.use_mnist_split}.png"
    
    save_triplet_grid(cond_all[:n], pred_all[:n], x0_all[:n], save_path, max_rows=n)

    print(f"[Done] Saved visualization to {save_path} (showing {n} triplets)")

def run_metrics(args, model, sched, loader, device, out_dir, *, ckpt_path: str):
    """Full evaluation: Computes MSE and SSIM over the entire dataset."""
    print("[Mode] Metrics: Calculating MSE and SSIM on full dataset...")
    
    # Try importing SSIM (scikit-image dependency)
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        print("[Error] scikit-image not found. Install via `pip install scikit-image`.")
        return

    mse_sum = 0.0
    denom = 0
    ssim_sum = 0.0
    ssim_count = 0
    
    min_ssim = float("inf")
    max_ssim = -float("inf")

    worst_triplet = None # (cond, pred, target) all in [-1,1]
    best_triplet = None  
    
    worst_map = None # np.ndarray (H,W)
    best_map = None 

    ssim_list = []

    for cond, x0 in tqdm(loader, desc="Evaluating"):
        cond = cond.to(device)
        x0 = x0.to(device)
        
        with torch.no_grad():
            pred = sched.sample(model, cond, device=device)
        
        # 1. MSE Calculation (pred, x0 are [-1,1] scale)
        mse_sum += F.mse_loss(pred, x0, reduction="sum").item()
        denom += x0.numel()
        
        # 2. SSIM (need [0,1] numpy on CPU)
        cond_cpu = cond.detach().cpu()
        x0_cpu = x0.detach().cpu()
        pred_cpu = pred.detach().cpu()

        p01 = to_01_range(pred).cpu().numpy()
        t01 = to_01_range(x0).cpu().numpy()
        
        # SSIM is computed per image
        # save_triplet_grid에서 to_01_range가 1번만 적용되어 회색 배경(더블 스케일링) 문제가 없음
        bs = pred_cpu.shape[0]
        for i in range(bs):
            val, s_map = ssim(t01[i].squeeze(), p01[i].squeeze(), data_range=1.0, full=True) # SSIM map 반환
            ssim_sum += val
            ssim_count += 1
            ssim_list.append(val)

            # worst
            if val < min_ssim:
                min_ssim = val
                worst_triplet = (
                    cond_cpu[i : i + 1],  # (1,1,28,28)
                    x0_cpu[i : i + 1],    # target in [-1,1]
                    pred_cpu[i : i + 1],  # pred in [-1,1]
                )
                worst_map = s_map

            # best
            if val > max_ssim:
                max_ssim = val
                best_triplet = (
                    cond_cpu[i : i + 1],
                    x0_cpu[i : i + 1],
                    pred_cpu[i : i + 1],
                )
                best_map = s_map

    final_mse = mse_sum / max(1, denom)
    final_ssim = ssim_sum / max(1, ssim_count)

    # Save Results
    res_path = out_dir / f"metrics_{args.use_mnist_split}.txt"
    with open(res_path, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Split: {args.use_mnist_split}\n")
        f.write(f"MSE: {final_mse:.6f}\n")
        f.write(f"SSIM: {final_ssim:.6f}\n")
        f.write(f"Num Samples: {ssim_count}\n")
        f.write(f"Min SSIM: {min_ssim:.6f}\n")
        f.write(f"Max SSIM: {max_ssim:.6f}\n")
    
    # worst/best 이미지 저장
    if worst_triplet is not None:
        worst_path = out_dir / f"worst_ssim_{args.use_mnist_split}.png"
        c, p, t = worst_triplet # worst_triplet = (cond, pred, target) in [-1,1]
        save_triplet_grid(c, p, t, worst_path, max_rows=1)
        print(f"[Info] Worst SSIM={min_ssim:.6f} saved to {worst_path}")
        if args.save_ssim_map and (worst_map is not None):
            worst_map_path = out_dir / f"worst_ssim_map_{args.use_mnist_split}.png"
            _save_ssim_map_png(worst_map, worst_map_path, title=f"Worst SSIM map ({min_ssim:.4f})")
            print(f"[Info] Worst SSIM map saved to {worst_map_path}")

    if best_triplet is not None:
        best_path = out_dir / f"best_ssim_{args.use_mnist_split}.png"
        c, p, t = best_triplet
        save_triplet_grid(c, p, t, best_path, max_rows=1)
        print(f"[Info] Best SSIM={max_ssim:.6f} saved to {best_path}")

        if args.save_ssim_map and (best_map is not None):
            best_map_path = out_dir / f"best_ssim_map_{args.use_mnist_split}.png"
            _save_ssim_map_png(best_map, best_map_path, title=f"Best SSIM map ({max_ssim:.4f})")
            print(f"[Info] Best SSIM map saved to {best_map_path}")
            

    # SSIM distribution plot
    if len(ssim_list) > 0:
        plt.figure()
        plt.hist(ssim_list, bins=50, range=(0, 1), alpha=0.8)
        plt.xlabel("SSIM")
        plt.ylabel("Count")
        plt.title(f"SSIM distribution ({args.use_mnist_split})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"ssim_hist_{args.use_mnist_split}.png", dpi=300)
        plt.close()
        print(f"[Info] SSIM histogram saved to {out_dir}/ssim_hist_{args.use_mnist_split}.png")
    
    print(f"[Done] Results saved to {res_path}")
    print(f"      MSE: {final_mse:.6f}")
    print(f"      SSIM: {final_ssim:.6f}")


def _save_ssim_map_png(ssim_map: np.ndarray, out_path: Path, title: str = "SSIM map") -> None:
    """
    ssim_map: (H,W) float array, typically in [0,1]
    """
    plt.figure()
    plt.imshow(ssim_map, vmin=0, vmax=1)
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = args.ckpt_paths if args.ckpt_paths is not None else [args.ckpt_path]

    ratios = (1.0, 0.0, 0.0)
    train_loader, val_loader, test_loader = build_threeway_dataloaders(
        mnist_root=args.mnist_root,
        batch_size=args.batch_size,
        ratios=ratios,
        split_seed=0,
        num_samples=args.num_samples,
        max_shift_px=args.max_shift_px,
        padding_mode=args.padding_mode,
        seed=42,
        num_workers=args.num_workers,
        use_mnist_split=args.use_mnist_split,
        no_split=True,  
    )
    
    if args.use_mnist_split == "train":
        loader = train_loader
    elif args.use_mnist_split == "val":
        loader = val_loader
    else:
        loader = test_loader

    if loader is None or len(loader) == 0:
        print("[Error] DataLoader is empty. Check split ratios.")
        sys.exit(1)

    for ckpt_path in ckpts:
        out_dir = args.output_dir
        if out_dir is None:
            out_dir = ckpt_path.parent / "eval_results"
        elif len(ckpts) > 1:
            out_dir = out_dir / ckpt_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        model, sched = load_model_and_sched(ckpt_path, device)

        if args.mode == "sample":
            run_sampling(args, model, sched, loader, device, out_dir)
        elif args.mode == "metrics":
            run_metrics(args, model, sched, loader, device, out_dir, ckpt_path=str(ckpt_path))
        else:
            print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
