"""
Procedural synthetic cloud generator.

Genuine paired (cloudy, cloud-free) LISS-IV acquisitions of the exact same
ground location are scarce. To train a robust reconstruction model we
synthesize realistic cloud occlusions over genuine cloud-free scenes using
fractal (Perlin/simplex-like) noise fields thresholded into soft cloud
masks with varying opacity, size, and shape — approximating real cirrus/
cumulus footprints and their partial-transmittance boundary regions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class SyntheticCloudConfig:
    min_clouds: int = 1
    max_clouds: int = 4
    min_radius_frac: float = 0.08
    max_radius_frac: float = 0.35
    softness: float = 8.0          # gaussian blur sigma for soft cloud edges
    min_opacity: float = 0.6
    max_opacity: float = 1.0
    brightness_boost: float = 0.85  # clouds appear near-white/bright


class SyntheticCloudGenerator:
    def __init__(self, cfg: SyntheticCloudConfig = SyntheticCloudConfig()):
        self.cfg = cfg

    def _fractal_noise(self, h: int, w: int, octaves: int = 4) -> np.ndarray:
        noise = np.zeros((h, w), dtype=np.float32)
        amplitude = 1.0
        total_amp = 0.0
        for o in range(octaves):
            scale = 2 ** o
            small_h, small_w = max(2, h // (4 * scale)), max(2, w // (4 * scale))
            layer = np.random.randn(small_h, small_w).astype(np.float32)
            layer = gaussian_filter(layer, sigma=1.0)
            layer_resized = _resize_nearest(layer, h, w)
            noise += amplitude * layer_resized
            total_amp += amplitude
            amplitude *= 0.5
        noise /= total_amp
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        return noise

    def generate_mask(self, h: int, w: int) -> np.ndarray:
        """Return a soft opacity map in [0, 1], shape (H, W)."""
        n_clouds = np.random.randint(self.cfg.min_clouds, self.cfg.max_clouds + 1)
        opacity_map = np.zeros((h, w), dtype=np.float32)

        for _ in range(n_clouds):
            base_noise = self._fractal_noise(h, w)
            radius_frac = np.random.uniform(self.cfg.min_radius_frac, self.cfg.max_radius_frac)
            cy, cx = np.random.uniform(0.1, 0.9, size=2)
            yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / radius_frac
            blob = np.clip(1.0 - dist, 0, 1) ** 1.5
            shaped = blob * (0.5 + 0.5 * base_noise)
            opacity = np.random.uniform(self.cfg.min_opacity, self.cfg.max_opacity)
            opacity_map = np.maximum(opacity_map, shaped * opacity)

        opacity_map = gaussian_filter(opacity_map, sigma=self.cfg.softness)
        return np.clip(opacity_map, 0, 1)

    def apply(self, cloud_free: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Composite a synthetic cloud layer onto a clean (C,H,W) image.
        Returns (cloudy_image, binary_mask).
        """
        c, h, w = cloud_free.shape
        opacity = self.generate_mask(h, w)
        cloud_layer = np.ones_like(cloud_free) * self.cfg.brightness_boost
        cloudy = cloud_free * (1 - opacity) + cloud_layer * opacity
        binary_mask = (opacity > 0.15).astype(np.uint8)
        return np.clip(cloudy, 0, 1).astype(np.float32), binary_mask


def _resize_nearest(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Lightweight dependency-free nearest-neighbor resize for noise octaves."""
    src_h, src_w = arr.shape
    row_idx = (np.linspace(0, src_h - 1, h)).astype(np.int32)
    col_idx = (np.linspace(0, src_w - 1, w)).astype(np.int32)
    return arr[row_idx][:, col_idx]
