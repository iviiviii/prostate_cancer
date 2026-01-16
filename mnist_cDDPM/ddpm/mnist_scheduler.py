#!/usr/bin/env python3
"""DDPM noise scheduler for MNIST conditional model."""

from __future__ import annotations

import torch


class NoiseScheduler:
    """DDPM forward/reverse process utilities."""

    def __init__(self, num_steps: int, beta_start: float, beta_end: float):
        self.num_steps = int(num_steps)
        betas = torch.linspace(beta_start, beta_end, self.num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_bar[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.alpha_bar_prev = alpha_bar_prev
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        self.posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        device = x0.device
        sqrt_ab = self.sqrt_alpha_bar.to(device).gather(0, t).view(-1, 1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar.to(device).gather(0, t).view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_omab * noise

    def predict_x0(self, xt: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        device = xt.device
        sqrt_ab = self.sqrt_alpha_bar.to(device).gather(0, t).view(-1, 1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar.to(device).gather(0, t).view(-1, 1, 1, 1)
        return (xt - sqrt_omab * eps) / torch.clamp(sqrt_ab, min=1e-8)

    def p_sample(
        self,
        model: torch.nn.Module,
        xt: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        device = xt.device
        betas = self.betas.to(device).gather(0, t).view(-1, 1, 1, 1)
        alphas = self.alphas.to(device).gather(0, t).view(-1, 1, 1, 1)
        alpha_bar = self.alpha_bar.to(device).gather(0, t).view(-1, 1, 1, 1)
        alpha_bar_prev = self.alpha_bar_prev.to(device).gather(0, t).view(-1, 1, 1, 1)
        sqrt_alpha = torch.sqrt(alphas)

        eps_pred = model(torch.cat([xt, cond], dim=1), t)
        x0_pred = self.predict_x0(xt, eps_pred, t)
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        coef1 = betas * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar)
        coef2 = (1.0 - alpha_bar_prev) * sqrt_alpha / (1.0 - alpha_bar)
        model_mean = coef1 * x0_pred + coef2 * xt

        var = self.posterior_variance.to(device).gather(0, t).view(-1, 1, 1, 1)
        if (t == 0).all():
            return model_mean
        noise = torch.randn_like(xt)
        return model_mean + torch.sqrt(torch.clamp(var, min=1e-20)) * noise

    @torch.no_grad()
    def sample(self, model: torch.nn.Module, cond: torch.Tensor, device: torch.device) -> torch.Tensor:
        model.eval()
        x = torch.randn_like(cond, device=device)
        for t_step in reversed(range(self.num_steps)):
            t = torch.full((cond.shape[0],), t_step, device=device, dtype=torch.long)
            x = self.p_sample(model, x, cond, t)
        return x
