"""
Conditional denoising diffusion probabilistic model (DDPM/DDIM) for cloud
removal — offered as an alternative architecture to the GAN, selectable via
`model.architecture: diffusion` in config.yaml.

The model learns to predict noise for a cloud-free target conditioned on
the cloudy image + cloud mask + SAR (concatenated as conditioning
channels), following the conditional-DDPM recipe used in works such as
Palette / SR3 adapted here to multispectral cloud removal. DDIM sampling is
used at inference for a fast, small number of reverse steps.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from src.models.attention import build_attention


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device).float() / half)
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class ResBlockTimeCond(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(torch.relu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(torch.relu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNetDenoiser(nn.Module):
    """U-Net epsilon-predictor conditioned on (cloudy, mask, sar) channels
    concatenated with the noisy target at each diffusion timestep."""

    def __init__(
        self,
        target_channels: int = 4,
        cond_channels: int = 7,  # optical(4) + mask(1) + sar(2)
        base_channels: int = 64,
        time_dim: int = 256,
        attention: str = "cbam",
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        in_ch = target_channels + cond_channels
        self.in_conv = nn.Conv2d(in_ch, base_channels, 3, padding=1)

        self.down1 = ResBlockTimeCond(base_channels, base_channels * 2, time_dim)
        self.pool1 = nn.AvgPool2d(2)
        self.down2 = ResBlockTimeCond(base_channels * 2, base_channels * 4, time_dim)
        self.pool2 = nn.AvgPool2d(2)

        self.mid = ResBlockTimeCond(base_channels * 4, base_channels * 4, time_dim)
        self.mid_attn = build_attention(attention, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.up_block2 = ResBlockTimeCond(base_channels * 4, base_channels * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.up_block1 = ResBlockTimeCond(base_channels * 2, base_channels, time_dim)

        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, target_channels, 3, padding=1)

    def forward(self, noisy_target: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x = torch.cat([noisy_target, cond], dim=1)
        x0 = self.in_conv(x)

        d1 = self.down1(x0, t_emb)
        p1 = self.pool1(d1)
        d2 = self.down2(p1, t_emb)
        p2 = self.pool2(d2)

        m = self.mid(p2, t_emb)
        m = self.mid_attn(m)

        u2 = self.up2(m)
        u2 = self.up_block2(torch.cat([u2, d2], dim=1), t_emb)
        u1 = self.up1(u2)
        u1 = self.up_block1(torch.cat([u1, d1], dim=1), t_emb)

        out = self.out_conv(torch.relu(self.out_norm(u1)))
        return out


class GaussianDiffusion:
    """Forward process q(x_t|x_0) + DDIM accelerated sampling for the
    conditional denoiser above."""

    def __init__(self, timesteps: int = 1000, schedule: str = "cosine", device: str = "cpu"):
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps) if schedule == "cosine" else torch.linspace(1e-4, 0.02, timesteps)
        self.betas = betas.to(device)
        self.alphas = (1.0 - self.betas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_1m_ac = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_ac * x0 + sqrt_1m_ac * noise, noise

    @torch.no_grad()
    def ddim_sample(
        self,
        model: ConditionalUNetDenoiser,
        cond: torch.Tensor,
        shape,
        sampling_steps: int = 50,
        eta: float = 0.0,
        device: str = "cpu",
    ) -> torch.Tensor:
        step_indices = torch.linspace(0, self.timesteps - 1, sampling_steps).long().flip(0)
        x = torch.randn(shape, device=device)

        for i, t_val in enumerate(step_indices):
            t_batch = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            eps_pred = model(x, cond, t_batch)

            alpha_cumprod_t = self.alphas_cumprod[t_val]
            alpha_cumprod_prev = self.alphas_cumprod[step_indices[i + 1]] if i + 1 < len(step_indices) else torch.tensor(1.0, device=device)

            x0_pred = (x - (1 - alpha_cumprod_t).sqrt() * eps_pred) / alpha_cumprod_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            dir_xt = (1 - alpha_cumprod_prev).sqrt() * eps_pred
            x = alpha_cumprod_prev.sqrt() * x0_pred + dir_xt

        return x
