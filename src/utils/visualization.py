"""Visualization helpers: RGB composites, side-by-side comparison panels,
difference maps, and cloud-mask overlays for qualitative QA of reconstructed
LISS-IV scenes."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def to_display_rgb(arr: np.ndarray, band_indices: Sequence[int] = (2, 1, 0)) -> np.ndarray:
    """Convert a (C,H,W) normalized array into an (H,W,3) uint8 RGB composite.
    Default band_indices=(2,1,0) maps (NIR/Red/Green)->(R,G,B)-ish false color
    for LISS-IV ordering [G,R,NIR,SWIR]; adjust per sensor band order."""
    c = arr.shape[0]
    idx = [min(i, c - 1) for i in band_indices]
    rgb = np.stack([arr[i] for i in idx], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)


def save_comparison_panel(
    cloudy: np.ndarray,
    cloud_mask: np.ndarray,
    reconstructed: np.ndarray,
    ground_truth: Optional[np.ndarray],
    out_path: str | Path,
    band_indices: Sequence[int] = (2, 1, 0),
    title: str = "",
) -> None:
    """Save a multi-panel PNG: cloudy input | mask | reconstruction | (GT | diff)."""
    n_panels = 3 + (2 if ground_truth is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    axes[0].imshow(to_display_rgb(cloudy, band_indices))
    axes[0].set_title("Cloudy Input")
    axes[0].axis("off")

    axes[1].imshow(cloud_mask.squeeze(), cmap="gray")
    axes[1].set_title("Cloud Mask")
    axes[1].axis("off")

    axes[2].imshow(to_display_rgb(reconstructed, band_indices))
    axes[2].set_title("Reconstructed")
    axes[2].axis("off")

    if ground_truth is not None:
        axes[3].imshow(to_display_rgb(ground_truth, band_indices))
        axes[3].set_title("Ground Truth")
        axes[3].axis("off")

        diff = np.abs(reconstructed - ground_truth).mean(axis=0)
        im = axes[4].imshow(diff, cmap="inferno", vmin=0, vmax=diff.max() + 1e-6)
        axes[4].set_title("Abs Error")
        axes[4].axis("off")
        fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: dict, out_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key in history:
        if "loss" in key.lower():
            axes[0].plot(history[key], label=key)
        else:
            axes[1].plot(history[key], label=key)
    axes[0].set_title("Losses")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].set_title("Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
