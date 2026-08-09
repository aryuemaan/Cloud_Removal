"""
Dataset preparation script: reads raw scenes (as produced either by
`generate_sample_data.py` for demo/testing, or real downloaded Bhoonidhi
LISS-IV + Sentinel-1 data placed in the same directory layout), extracts
training patches, and writes the train/val/test split manifest.

Usage:
    python scripts/prepare_dataset.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cloud_mask import CloudMaskConfig
from src.data.preprocessing import (
    PreprocessConfig,
    extract_patches_from_scene,
    save_patches,
    split_scenes,
    write_split_manifest,
)
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("prepare_dataset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    cloud_free_dir = Path(cfg.paths.cloud_free_dir)
    cloudy_dir = Path(cfg.paths.cloudy_dir)
    sar_dir = Path(cfg.paths.sar_dir)

    scene_files = sorted(cloud_free_dir.glob("*.npz"))
    if not scene_files:
        raise RuntimeError(
            f"No scenes found in {cloud_free_dir}. Run scripts/generate_sample_data.py "
            f"first, or place real Bhoonidhi LISS-IV scenes there."
        )

    preprocess_cfg = PreprocessConfig(
        patch_size=cfg.data.patch_size,
        overlap=cfg.data.patch_overlap,
        cloud_fraction_min=cfg.data.cloud_fraction_min,
        cloud_fraction_max=cfg.data.cloud_fraction_max,
        normalization_lower=cfg.data.normalization.lower_percentile,
        normalization_upper=cfg.data.normalization.upper_percentile,
        seed=cfg.project.seed,
    )
    mask_cfg = CloudMaskConfig()

    scene_ids = []
    total_patches = 0

    for scene_path in scene_files:
        scene_id = scene_path.stem
        scene_ids.append(scene_id)

        clean = np.load(scene_path)["image"]

        cloudy_path = cloudy_dir / f"{scene_id}.npz"
        if cloudy_path.exists():
            cloudy_data = np.load(cloudy_path)
            cloudy = cloudy_data["image"]
        else:
            cloudy = clean  # fallback: will get synthetic clouds at train time

        sar_path = sar_dir / f"{scene_id}.npz"
        sar = np.load(sar_path)["image"] if sar_path.exists() else None

        patches = extract_patches_from_scene(cloudy, clean, sar, preprocess_cfg, mask_cfg)
        save_patches(patches, cfg.paths.patches_dir, scene_id)
        total_patches += len(patches)
        logger.info(f"{scene_id}: extracted {len(patches)} patches")

    splits = split_scenes(scene_ids, ratios=tuple(cfg.data.train_val_test_split), seed=cfg.project.seed)
    write_split_manifest(splits, os.path.join(cfg.paths.processed_dir, "splits.json"))

    logger.info(f"Total patches extracted: {total_patches}")
    logger.info(f"Split sizes -> train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])} scenes")


if __name__ == "__main__":
    main()
