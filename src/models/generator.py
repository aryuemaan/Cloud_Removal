"""
Generator architecture: a multi-modal, attention-augmented encoder-decoder
(U-Net family) that fuses cloudy optical LISS-IV imagery with Sentinel-1 SAR
(cloud-penetrating radar, unaffected by cloud cover) and the binary/soft
cloud mask, to reconstruct a cloud-free optical scene.

Design rationale (industry-informed, echoes architectures such as
DSen2-CR / GLF-CR / McGAN):
  - Two parallel encoder stems (optical, SAR) with early + mid-level fusion
    via concatenation + learned gating, since SAR and optical have very
    different statistics and naive early concatenation underuses SAR signal.
  - The cloud mask is fed both as an input channel and used to build a
    gated skip-connection so the decoder relies more on SAR/context features
    in occluded regions and more on original optical features in clear
    regions (mask-aware skip gating).
  - CBAM attention at each decoder stage refines channel/spatial focus.
  - Optional lightweight temporal encoder fuses a short stack of prior
    cloud-free-ish reference frames (if available) as extra context.
  - Output is residual: the network predicts a *correction* added to the
    input, stabilizing training and preserving fine detail in clear regions.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.attention import build_attention


def conv_block(in_ch: int, out_ch: int, norm: bool = True) -> nn.Sequential:
    layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers += [nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = conv_block(in_ch, out_ch)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        feat = self.conv(x)
        down = self.pool(feat)
        return feat, down


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, attention: str):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = conv_block(out_ch + skip_ch, out_ch)
        self.attn = build_attention(attention, out_ch)

    def forward(self, x, skip, mask_gate: Optional[torch.Tensor] = None):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if mask_gate is not None:
            gate = nn.functional.interpolate(mask_gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            skip = skip * (1 - gate) + skip.mean(dim=(2, 3), keepdim=True) * gate * 0.5 + skip * gate * 0.5
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.attn(x)
        return x


class SAREncoderStem(nn.Module):
    """Lightweight encoder for SAR (VV/VH) input, fused at multiple scales."""

    def __init__(self, in_channels: int, base_channels: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels // 2, 3, padding=1),
            nn.InstanceNorm2d(base_channels // 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.stem(x)


class TemporalEncoder(nn.Module):
    """Fuses a short stack of temporal reference frames via a small 3D-ish
    conv (implemented as grouped 2D conv over stacked channels) producing a
    fixed-size context embedding concatenated with the bottleneck."""

    def __init__(self, in_channels_per_frame: int, n_frames: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels_per_frame * n_frames, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, stacked_frames: torch.Tensor) -> torch.Tensor:
        return self.net(stacked_frames)


class SARFusionAttentionGenerator(nn.Module):
    """
    Multi-modal generator: optical (cloudy) + cloud mask + SAR -> cloud-free
    optical reconstruction, residual-corrected.
    """

    def __init__(
        self,
        in_channels_optical: int = 4,
        in_channels_sar: int = 2,
        in_channels_mask: int = 1,
        base_channels: int = 64,
        depth: int = 5,
        attention: str = "cbam",
        use_temporal: bool = False,
        temporal_frames: int = 3,
    ):
        super().__init__()
        self.use_temporal = use_temporal
        self.depth = depth

        sar_stem_ch = base_channels // 2
        self.sar_stem = SAREncoderStem(in_channels_sar, base_channels)

        in_ch = in_channels_optical + in_channels_mask + sar_stem_ch
        self.input_proj = nn.Conv2d(in_ch, base_channels, 3, padding=1)

        # Encoder
        self.downs = nn.ModuleList()
        ch = base_channels
        chans = [ch]
        for i in range(depth):
            out_ch = min(ch * 2, base_channels * 16)
            self.downs.append(DownBlock(ch, out_ch))
            ch = out_ch
            chans.append(ch)

        bottleneck_in = ch
        if use_temporal:
            self.temporal_encoder = TemporalEncoder(
                in_channels_optical, temporal_frames, base_channels * 2
            )
            bottleneck_in += base_channels * 2

        self.bottleneck = nn.Sequential(
            conv_block(bottleneck_in, ch),
            build_attention(attention, ch),
        )

        # Decoder. `chans` = [base_channels, out_ch_down0, out_ch_down1, ...],
        # length depth+1. skips[k] (produced by down block k) has
        # chans[k+1] channels. For up-stage i (0-indexed from the
        # bottleneck), the matching skip is skips[depth-1-i], i.e. it has
        # chans[depth-i] channels, and the stage should output chans[depth-i-1]
        # channels to stay symmetric with the encoder.
        self.ups = nn.ModuleList()
        for i in range(depth):
            skip_ch = chans[depth - i]
            out_ch = chans[depth - i - 1]
            self.ups.append(UpBlock(ch, skip_ch, out_ch, attention))
            ch = out_ch

        self.output_proj = nn.Sequential(
            nn.Conv2d(ch, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, in_channels_optical, 3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        cloudy: torch.Tensor,
        cloud_mask: torch.Tensor,
        sar: torch.Tensor,
        temporal_stack: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sar_feat = self.sar_stem(sar)
        x = torch.cat([cloudy, cloud_mask, sar_feat], dim=1)
        x = self.input_proj(x)

        skips = []
        for down in self.downs:
            feat, x = down(x)
            skips.append(feat)

        if self.use_temporal:
            if temporal_stack is not None:
                temporal_feat = self.temporal_encoder(temporal_stack)
                temporal_feat = nn.functional.adaptive_avg_pool2d(temporal_feat, x.shape[-2:]) \
                    if temporal_feat.shape[-2:] != x.shape[-2:] else temporal_feat
            else:
                # use_temporal=True but no reference stack was supplied for
                # this forward pass (e.g. batch without temporal data) —
                # feed zeros so the fixed bottleneck input width is honored.
                b, _, h, w = x.shape
                temporal_channels = self.temporal_encoder.net[0].out_channels
                temporal_feat = torch.zeros(b, temporal_channels, h, w, device=x.device, dtype=x.dtype)
            x = torch.cat([x, temporal_feat], dim=1)

        x = self.bottleneck(x)

        for i, up in enumerate(self.ups):
            skip = skips[len(skips) - 1 - i]
            mask_gate = nn.functional.interpolate(cloud_mask, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = up(x, skip, mask_gate)

        residual = self.output_proj(x)
        # Residual reconstruction: blend correction only within (softened)
        # cloud-occluded areas, preserve original pixels elsewhere.
        blended = cloudy + residual * cloud_mask
        return torch.clamp(blended, -1, 1), residual
