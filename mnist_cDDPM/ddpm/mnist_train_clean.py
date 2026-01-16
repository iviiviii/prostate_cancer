#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from .mnist_model import ConditionalUNet
from .mnist_utils import set_seed, build_threeway_dataloaders
from .mnist_scheduler import NoiseScheduler
from .mnist_viz import save_triplet_grid, save_pair_grid


def get_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Train conditional DDPM on MNIST row-shift artifacts")

    # data
    p.add_argument("--mnist-root", type=Path, default=Path.home() / ".cache" / "mnist")
    p.add_argument("--num-samples", type=int, default=None, help="Optional cap; default None uses full MNIST train split.")

    p.add_argument("--max-shift-px", type=int, default=6)
    p.add_argument("--padding-mode", type=str, default="zeros", choices=["zeros", "border", "reflection"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)

    # training
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-steps", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--log-every", type=int, default=10000)
    p.add_argument("--sample-every", type=int, default=5000)
    p.add_argument("--num-sample-save", type=int, default=8)
    p.add_argument("--ckpt-every", type=int, default=20000, help="Save checkpoint every N steps.")

    # diffusion
    p.add_argument("--diffusion-steps", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=0.02)

    # model
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--time-dim", type=int, default=128)
    return p


def _get_device(cpu_flag: bool) -> torch.device:
    if cpu_flag or (not torch.cuda.is_available()):
        return torch.device("cpu")
    return torch.device("cuda")


def train(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = _get_device(args.cpu)
    set_seed(args.seed)

    cfg_path = args.output_dir / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    train_loader, val_loader, test_loader = build_threeway_dataloaders(
        mnist_root=args.mnist_root,
        batch_size=args.batch_size,
        ratios=None,
        split_seed=0,
        num_samples=args.num_samples,
        max_shift_px=args.max_shift_px,
        padding_mode=args.padding_mode,
        seed=args.seed,
        num_workers=args.num_workers,
        use_mnist_split="train",
        no_split=True,  
    )
    data_iter = iter(train_loader)

    model = ConditionalUNet(in_channels=2, base_channels=args.base_channels, time_dim=args.time_dim, out_channels=1).to(device)
    sched = NoiseScheduler(num_steps=args.diffusion_steps, beta_start=args.beta_start, beta_end=args.beta_end)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    csv_path = args.output_dir / "train_log.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("step,loss\n")

    def _save_ckpt(step: int, fname: str):
        ckpt = {
            "model": model.state_dict(),
            "args": vars(args),
            "diffusion": {
                "num_steps": args.diffusion_steps,
                "beta_start": args.beta_start,
                "beta_end": args.beta_end,
            },
            "step": step,
        }
        torch.save(ckpt, args.output_dir / fname)

    last_loss = None
    loss_hist = []
    step_hist = []
    record_every = 1000  
    for step in tqdm(range(1, args.train_steps + 1), desc="train", total=args.train_steps):
        try:
            cond, x0 = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            cond, x0 = next(data_iter)

        cond = cond.to(device)
        x0 = x0.to(device)

        B = cond.shape[0]
        t = torch.randint(0, sched.num_steps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = sched.q_sample(x0, t, noise)

        eps_pred = model(torch.cat([xt, cond], dim=1), t)
        loss = F.mse_loss(eps_pred, noise)
        last_loss = loss.item()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step % record_every == 0) or (step == 1) or (step == args.train_steps):
            step_hist.append(step)
            loss_hist.append(last_loss)
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(f"{step},{last_loss:.6f}\n")

        if (step % args.log_every == 0) or (step == 1):
            print(f"[step {step:06d}] loss={loss.item():.6f}")

        if (step % args.sample_every == 0) or (step == args.train_steps):

            model.eval() 
            with torch.no_grad():
                n = min(args.num_sample_save, B)
                cond_vis = cond[:n]
                x0_vis = x0[:n]
                pred = sched.sample(model, cond_vis, device=device) 
            save_triplet_grid(cond_vis, pred, x0_vis, args.output_dir / f"samples_step{step:06d}.png", max_rows=min(6, n))
            save_pair_grid(pred, x0_vis, args.output_dir / f"pred_vs_gt_step{step:06d}.png", nrow=min(4, n))
            model.train()

        if (step % args.ckpt_every == 0) or (step == args.train_steps):
            _save_ckpt(step, f"ckpt_step{step:06d}.pt")

    _save_ckpt(args.train_steps, "model_last.pt")
    print(f"[done] saved checkpoint to {args.output_dir / 'model_last.pt'}, final_loss={last_loss:.6f}")

    with open(csv_path, "a", encoding="utf-8") as f:
        f.write(f"final,{last_loss:.6f}\n")

    if len(loss_hist) > 0:
        plt.figure()
        plt.plot(step_hist, loss_hist)
        plt.xlabel("step")
        plt.ylabel("train loss (MSE)")
        plt.title("Training loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.output_dir / "loss_curve.png", dpi=300)
        plt.close()
        