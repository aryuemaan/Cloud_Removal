"""Factory functions that build model instances from the config object, so
training/inference/eval scripts stay architecture-agnostic."""
from __future__ import annotations

import torch.nn as nn

from src.models.diffusion_model import ConditionalUNetDenoiser, GaussianDiffusion
from src.models.discriminator import PatchGANDiscriminator
from src.models.generator import SARFusionAttentionGenerator


def build_generator(cfg) -> nn.Module:
    g_cfg = cfg.model.generator
    if cfg.model.architecture in ("sar_fusion_attention_gan", "plain_unet"):
        return SARFusionAttentionGenerator(
            in_channels_optical=g_cfg.in_channels_optical,
            in_channels_sar=g_cfg.in_channels_sar,
            in_channels_mask=g_cfg.in_channels_mask,
            base_channels=g_cfg.base_channels,
            depth=g_cfg.depth,
            attention=g_cfg.attention if cfg.model.architecture != "plain_unet" else "none",
            use_temporal=g_cfg.use_temporal,
            temporal_frames=g_cfg.temporal_frames,
        )
    elif cfg.model.architecture == "diffusion":
        cond_channels = g_cfg.in_channels_optical + g_cfg.in_channels_mask + g_cfg.in_channels_sar
        return ConditionalUNetDenoiser(
            target_channels=g_cfg.in_channels_optical,
            cond_channels=cond_channels,
            base_channels=g_cfg.base_channels,
            attention=g_cfg.attention,
        )
    raise ValueError(f"Unknown architecture: {cfg.model.architecture}")


def build_discriminator(cfg) -> nn.Module:
    d_cfg = cfg.model.discriminator
    return PatchGANDiscriminator(
        in_channels=d_cfg.in_channels,
        base_channels=d_cfg.base_channels,
        n_layers=d_cfg.n_layers,
    )


def build_diffusion_process(cfg, device: str) -> GaussianDiffusion:
    return GaussianDiffusion(
        timesteps=cfg.model.diffusion.timesteps,
        schedule=cfg.model.diffusion.beta_schedule,
        device=device,
    )
