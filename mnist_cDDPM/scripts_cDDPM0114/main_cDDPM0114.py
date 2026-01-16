#!/usr/bin/env python3
"""
Entry point for training conditional DDPM on MNIST row-shift artifacts.

python -m scripts_cDDPM0114.main_cDDPM0114 \
  --output-dir runs_cDDPM0114/train_full200000 \
  --train-steps 200000 --sample-every 5000\
  --batch-size 128 \
  --max-shift-px 6 --padding-mode zeros \
  --diffusion-steps 1000 --beta-start 1e-4 --beta-end 0.02 \
  --base-channels 64 --time-dim 128 
"""

from __future__ import annotations
from ddpm.mnist_train import get_arg_parser, train

def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()