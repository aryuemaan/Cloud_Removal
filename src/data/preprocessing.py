"""
End-to-end preprocessing pipeline that turns raw scenes (LISS-IV cloudy /
cloud-free, Sentinel-1 SAR, Sentinel-2 optical) into training-ready patches
saved as compressed .npz tensors.

Pipeline stages:
  1. Load raster scenes with rasterio.
  2. Radiometric normalization (percentile stretch to [0,1]).
  3. Co-registration of auxiliary sensors onto the LISS-IV reference grid.
  4. Cloud/shadow mask generation.
  5. Sliding-window patch extraction with configurable overlap and cloud
     fraction filtering (keeps patches that are informative for training —
     neither fully clear nor fully occluded, plus a fraction of clear-only
     patches for reconstruction-quality regularization).
  6. Train/val/test split at the *scene* level (not patch level) to prevent
     spatial leakage.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.data.cloud_mask import CloudMaskConfig, combined_occlusion_mask, cloud_fraction
from src.utils.geo_utils import percentile_normalize
from src.utils.logger import get_logger

logger = get_logger("preprocessing")


@dataclass
class PreprocessConfig:
    patch_size: int = 256
    overlap: int = 32
    cloud_fraction_min: float = 0.05
    cloud_fraction_max: float = 0.85
    clear_patch_keep_ratio: float = 0.15  # fraction of fully-clear patches to keep
    normalization_lower: float = 1.0
    normalization_upper: float = 99.0
    band_order: Tuple[str, ...] = ("G", "R", "NIR", "SWIR")
    seed: int = 42


def _sliding_windows(h: int, w: int, patch: int, overlap: int):
    stride = max(1, patch - overlap)
    for top in range(0, max(h - patch, 0) + 1, stride):
        for left in range(0, max(w - patch, 0) + 1, stride):
            yield top, left
    # ensure right/bottom edges are covered
    if (h - patch) % stride != 0:
        for left in range(0, max(w - patch, 0) + 1, stride):
            yield h - patch, left
    if (w - patch) % stride != 0:
        for top in range(0, max(h - patch, 0) + 1, stride):
            yield top, w - patch


def normalize_scene(scene: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    return percentile_normalize(
        scene, lower=cfg.normalization_lower, upper=cfg.normalization_upper
    )


def extract_patches_from_scene(
    cloudy: np.ndarray,
    cloud_free: Optional[np.ndarray],
    sar: Optional[np.ndarray],
    cfg: PreprocessConfig,
    mask_cfg: CloudMaskConfig = CloudMaskConfig(),
) -> List[dict]:
    """
    Extract informative patches from one co-registered scene tuple.
    Returns a list of dicts ready to be saved as .npz records.
    """
    _, h, w = cloudy.shape
    patches = []
    rng = random.Random(cfg.seed)

    for top, left in _sliding_windows(h, w, cfg.patch_size, cfg.overlap):
        cloudy_patch = cloudy[:, top:top + cfg.patch_size, left:left + cfg.patch_size]
        if cloudy_patch.shape[1] != cfg.patch_size or cloudy_patch.shape[2] != cfg.patch_size:
            continue  # skip incomplete edge patches

        mask = combined_occlusion_mask(cloudy_patch, cfg.band_order, mask_cfg)
        frac = cloud_fraction(mask)

        if frac < cfg.cloud_fraction_min:
            # mostly-clear patch: keep only a subsample for regularization
            if rng.random() > cfg.clear_patch_keep_ratio:
                continue
        elif frac > cfg.cloud_fraction_max:
            continue  # almost fully occluded — not useful as a training target

        record = {
            "cloudy": cloudy_patch.astype(np.float32),
            "cloud_mask": mask.astype(np.uint8),
            "top": top,
            "left": left,
            "cloud_frac": frac,
        }
        if cloud_free is not None:
            record["cloud_free"] = cloud_free[:, top:top + cfg.patch_size, left:left + cfg.patch_size].astype(np.float32)
        if sar is not None:
            record["sar"] = sar[:, top:top + cfg.patch_size, left:left + cfg.patch_size].astype(np.float32)
        patches.append(record)

    return patches


def save_patches(patches: List[dict], out_dir: str | Path, scene_id: str) -> List[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, p in enumerate(patches):
        fname = out_dir / f"{scene_id}_patch{i:05d}.npz"
        np.savez_compressed(fname, **p)
        paths.append(str(fname))
    return paths


def split_scenes(
    scene_ids: List[str], ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42
) -> dict:
    """Scene-level train/val/test split to avoid spatial data leakage."""
    rng = random.Random(seed)
    ids = scene_ids.copy()
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


def write_split_manifest(splits: dict, out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Wrote dataset split manifest to {out_path}")
