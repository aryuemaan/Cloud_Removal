"""
Augmentation pipeline for multi-band (>3 channel) satellite imagery.

Albumentations' standard transforms assume 3-channel RGB in several
color-specific ops, so we restrict ourselves to geometric + generic
pixel-level transforms that are channel-count agnostic, and apply them
jointly across cloudy image, cloud-free target, SAR, and mask via
`additional_targets` so spatial alignment is preserved.
"""
from __future__ import annotations

from typing import Optional

import albumentations as A
import numpy as np

# NOTE: Albumentations' pixel-level color/noise transforms (GaussNoise,
# RandomBrightnessContrast's per-channel gamma paths, etc.) assume 1/3/4
# "standard image" channel layouts internally via OpenCV broadcasting and
# do not reliably generalize to arbitrary band counts (e.g. a 2-channel SAR
# `additional_target` alongside a 4-channel optical `image`). We therefore
# restrict the Albumentations pipeline to purely geometric transforms
# (which are channel-count agnostic) and apply brightness/noise
# perturbations manually, band-by-band, after the geometric augmentation.


def build_augmentation_pipeline(cfg) -> A.Compose:
    aug_cfg = cfg.data.augmentation
    transforms = [
        A.HorizontalFlip(p=aug_cfg.horizontal_flip),
        A.VerticalFlip(p=aug_cfg.vertical_flip),
        A.RandomRotate90(p=aug_cfg.rotate90),
    ]
    return A.Compose(
        transforms,
        additional_targets={
            "cloud_free": "image",
            "sar": "image",
            "mask": "mask",
        },
    )


def apply_radiometric_jitter(
    img: np.ndarray,
    brightness_limit: float = 0.15,
    contrast_limit: float = 0.15,
    p: float = 0.3,
) -> np.ndarray:
    """Channel-count-agnostic brightness/contrast jitter, applied per
    (C,H,W) array in-place-equivalent (returns a new array)."""
    if np.random.rand() >= p:
        return img
    brightness = 1.0 + np.random.uniform(-brightness_limit, brightness_limit)
    contrast = 1.0 + np.random.uniform(-contrast_limit, contrast_limit)
    mean = img.mean(axis=(1, 2), keepdims=True)
    out = (img - mean) * contrast + mean * brightness
    return np.clip(out, 0, 1).astype(img.dtype)


def apply_gaussian_noise(img: np.ndarray, sigma_range=(0.001, 0.02), p: float = 0.1) -> np.ndarray:
    """Channel-count-agnostic additive Gaussian noise."""
    if np.random.rand() >= p:
        return img
    sigma = np.random.uniform(*sigma_range)
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    return np.clip(img + noise, 0, 1).astype(img.dtype)


def apply_joint_augmentation(
    pipeline: A.Compose,
    cloudy: np.ndarray,
    cloud_mask: np.ndarray,
    cloud_free: Optional[np.ndarray] = None,
    sar: Optional[np.ndarray] = None,
) -> dict:
    """
    Inputs are (C, H, W). Albumentations expects (H, W, C), so we transpose,
    apply, and transpose back.
    """
    kwargs = {"image": np.transpose(cloudy, (1, 2, 0)), "mask": cloud_mask}
    if cloud_free is not None:
        kwargs["cloud_free"] = np.transpose(cloud_free, (1, 2, 0))
    if sar is not None:
        kwargs["sar"] = np.transpose(sar, (1, 2, 0))

    out = pipeline(**kwargs)

    cloudy_out = np.transpose(out["image"], (2, 0, 1))
    result = {
        "cloudy": apply_gaussian_noise(
            apply_radiometric_jitter(cloudy_out, p=0.3), p=0.1
        ),
        "cloud_mask": out["mask"],
    }
    if cloud_free is not None:
        result["cloud_free"] = np.transpose(out["cloud_free"], (2, 0, 1))
    if sar is not None:
        result["sar"] = np.transpose(out["sar"], (2, 0, 1))
    return result
