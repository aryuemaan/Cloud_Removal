"""Attention building blocks: CBAM (channel + spatial attention) and a
lightweight self-attention block for capturing long-range spatial context —
important for reconstructing large, spatially-coherent cloud gaps where
local convolutional receptive fields alone are insufficient."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        attn = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class SelfAttention2d(nn.Module):
    """Non-local self-attention block for long-range spatial dependency
    modeling at low-resolution bottleneck stages (memory-efficient: applied
    only at the smallest feature map scale)."""

    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
        k = self.key(x).view(b, -1, h * w)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        v = self.value(x).view(b, -1, h * w)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)
        return x + self.gamma * out


def build_attention(kind: str, channels: int) -> nn.Module:
    if kind == "cbam":
        return CBAM(channels)
    if kind == "self_attention":
        return SelfAttention2d(channels)
    if kind == "none" or kind is None:
        return nn.Identity()
    raise ValueError(f"Unknown attention kind: {kind}")
