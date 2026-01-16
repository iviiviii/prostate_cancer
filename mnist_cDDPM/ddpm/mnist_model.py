#!/usr/bin/env python3
"""Conditional U-Net for MNIST artifact removal DDPM."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        emb = timesteps.float()[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResidualBlock(nn.Module):
    """GN + SiLU residual block with time embedding injection.
    groups: GroupNorm에서 채널을 몇 개의 그룹으로 나눠 정규화할지 결정하는 값
    3채널처럼 8로 안 나눠지면 gcd(3,8)=1이라 1그룹(채널 전체를 한 그룹)으로 동작
    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups=max(1, math.gcd(in_channels, groups)), num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups=max(1, math.gcd(out_channels, groups)), num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.residual(x)


class Down(nn.Module):
    """Stride-2 conv downsample."""

    def __init__(self, channels: int):
        super().__init__()
        self.down = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Up(nn.Module):
    """Nearest-neighbor upsample + conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class ConditionalUNet(nn.Module):
    """Small U-Net with 2 downs/ups. Input: cat([x_t, cond], dim=1)."""

    def __init__(self, in_channels: int = 2, base_channels: int = 64, time_dim: int = 128, out_channels: int = 1):
        super().__init__()
        b = base_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.enc1 = ResidualBlock(in_channels, b, time_dim)
        self.down1 = Down(b)

        self.enc2 = ResidualBlock(b, b * 2, time_dim)
        self.down2 = Down(b * 2)

        self.mid = ResidualBlock(b * 2, b * 2, time_dim)

        self.up2 = Up(b * 2, b)
        self.dec2 = ResidualBlock(b * 3, b, time_dim)

        self.up1 = Up(b, b // 2)
        self.dec1 = ResidualBlock(b // 2 + b, b // 2, time_dim)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=max(1, math.gcd(b // 2, 8)), num_channels=b // 2),
            nn.SiLU(),
            nn.Conv2d(b // 2, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        h1 = self.enc1(x, t_emb)
        h2 = self.enc2(self.down1(h1), t_emb)
        h3 = self.mid(self.down2(h2), t_emb)

        u2 = self.up2(h3)
        u2 = self.dec2(torch.cat([u2, h2], dim=1), t_emb)

        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, h1], dim=1), t_emb)

        return self.out_conv(u1)
