"""
Memory-efficient sliding-window inference over full-scene GeoTIFFs that
exceed GPU memory / model receptive design size, plus seamless blending of
overlapping patch predictions using Gaussian-weighted feathering to avoid
visible tile-seam artifacts in the mosaicked output — a standard requirement
for operational (as opposed to demo) satellite image processing.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch


def gaussian_weight_map(size: int, sigma_frac: float = 0.4) -> np.ndarray:
    sigma = size * sigma_frac
    ax = np.arange(size) - size / 2
    xx, yy = np.meshgrid(ax, ax)
    weight = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return weight / weight.max()


class SlidingWindowPredictor:
    def __init__(
        self,
        window_size: int = 256,
        stride: int = 192,
        blend_mode: str = "gaussian",
        device: str = "cpu",
    ):
        self.window_size = window_size
        self.stride = stride
        self.blend_mode = blend_mode
        self.device = device
        self._weight_cache = gaussian_weight_map(window_size) if blend_mode == "gaussian" else np.ones((window_size, window_size))

    def _windows(self, h: int, w: int):
        ys = list(range(0, max(h - self.window_size, 0) + 1, self.stride))
        xs = list(range(0, max(w - self.window_size, 0) + 1, self.stride))
        if not ys or ys[-1] != h - self.window_size:
            ys.append(max(h - self.window_size, 0))
        if not xs or xs[-1] != w - self.window_size:
            xs.append(max(w - self.window_size, 0))
        for y in ys:
            for x in xs:
                yield y, x

    def predict(
        self,
        predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        cloudy: np.ndarray,
        cloud_mask: np.ndarray,
        sar: Optional[np.ndarray] = None,
        batch_size: int = 8,
        tta: bool = False,
    ) -> np.ndarray:
        """
        cloudy: (C, H, W), cloud_mask: (1, H, W) or (H, W), sar: (C_sar, H, W)
        predict_fn: callable taking (cloudy_patch, mask_patch, sar_patch) each
                    shaped (B, C, ws, ws) and returning (B, C, ws, ws) prediction.
        Returns the full-scene reconstructed (C, H, W) array.
        """
        c, h, w = cloudy.shape
        if cloud_mask.ndim == 2:
            cloud_mask = cloud_mask[None, ...]
        if sar is None:
            sar = np.zeros((2, h, w), dtype=np.float32)

        accum = np.zeros((c, h, w), dtype=np.float32)
        weight_accum = np.zeros((1, h, w), dtype=np.float32)

        coords = list(self._windows(h, w))
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]
            cloudy_batch, mask_batch, sar_batch = [], [], []
            for (y, x) in batch_coords:
                cloudy_batch.append(cloudy[:, y:y + self.window_size, x:x + self.window_size])
                mask_batch.append(cloud_mask[:, y:y + self.window_size, x:x + self.window_size])
                sar_batch.append(sar[:, y:y + self.window_size, x:x + self.window_size])

            cloudy_np = np.stack(cloudy_batch)
            mask_np = np.stack(mask_batch)
            sar_np = np.stack(sar_batch)

            preds = predict_fn(cloudy_np, mask_np, sar_np)
            if tta:
                preds_flip = predict_fn(
                    np.ascontiguousarray(cloudy_np[..., ::-1]),
                    np.ascontiguousarray(mask_np[..., ::-1]),
                    np.ascontiguousarray(sar_np[..., ::-1]),
                )
                preds = (preds + preds_flip[..., ::-1]) / 2.0

            for j, (y, x) in enumerate(batch_coords):
                ph, pw = preds[j].shape[-2:]
                weight = self._weight_cache[:ph, :pw]
                accum[:, y:y + ph, x:x + pw] += preds[j] * weight
                weight_accum[:, y:y + ph, x:x + pw] += weight

        weight_accum = np.clip(weight_accum, 1e-6, None)
        return accum / weight_accum
