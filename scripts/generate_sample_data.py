"""
Generates a small synthetic sample dataset (fake GeoTIFF-like arrays saved
as .npz, mimicking LISS-IV cloudy/cloud-free scenes) so the ENTIRE pipeline
(preprocessing -> training -> inference -> evaluation) can be run end-to-end
out of the box, without needing to first download real Bhoonidhi/Sentinel
data. This is what CI, unit tests, and new-contributor onboarding use.

Usage:
    python scripts/generate_sample_data.py --num_scenes 6 --scene_size 512
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic_clouds import SyntheticCloudGenerator


def make_synthetic_scene(size: int, n_bands: int = 4, seed: int = 0) -> np.ndarray:
    """Generate a plausible-looking multi-band land-cover scene using
    layered Perlin-ish noise + structured patterns (fields, rivers, roads)
    so that downstream cloud-masking / metrics behave sensibly on it."""
    rng = np.random.RandomState(seed)
    scene = np.zeros((n_bands, size, size), dtype=np.float32)

    yy, xx = np.meshgrid(np.linspace(0, 6, size), np.linspace(0, 6, size), indexing="ij")
    base_terrain = 0.4 + 0.15 * np.sin(xx) * np.cos(yy) + 0.05 * rng.randn(size, size)

    river = np.exp(-((xx - 3 - 0.5 * np.sin(yy)) ** 2) / 0.05)
    fields = (np.sin(xx * 3) * np.cos(yy * 3) > 0.3).astype(np.float32) * 0.15

    for b in range(n_bands):
        band_variation = 0.03 * rng.randn(size, size)
        scene[b] = np.clip(base_terrain + 0.05 * b - 0.3 * river + fields + band_variation, 0, 1)

    return scene.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_scenes", type=int, default=6)
    parser.add_argument("--scene_size", type=int, default=512)
    parser.add_argument("--out_dir", type=str, default="data/raw")
    args = parser.parse_args()

    cloud_free_dir = os.path.join(args.out_dir, "cloud_free")
    cloudy_dir = os.path.join(args.out_dir, "cloudy")
    sar_dir = os.path.join(args.out_dir, "sentinel1_sar")
    for d in [cloud_free_dir, cloudy_dir, sar_dir]:
        os.makedirs(d, exist_ok=True)

    cloud_gen = SyntheticCloudGenerator()

    for i in range(args.num_scenes):
        scene_id = f"scene_{i:03d}"
        clean = make_synthetic_scene(args.scene_size, n_bands=4, seed=i)
        cloudy, mask = cloud_gen.apply(clean)

        # Fake SAR: correlated-but-different texture, unaffected by "cloud"
        sar = np.stack(
            [
                np.clip(clean[2] + 0.1 * np.random.randn(*clean.shape[1:]), 0, 1),  # VV proxy
                np.clip(clean[0] + 0.1 * np.random.randn(*clean.shape[1:]), 0, 1),  # VH proxy
            ]
        ).astype(np.float32)

        np.savez_compressed(os.path.join(cloud_free_dir, f"{scene_id}.npz"), image=clean)
        np.savez_compressed(os.path.join(cloudy_dir, f"{scene_id}.npz"), image=cloudy, mask=mask)
        np.savez_compressed(os.path.join(sar_dir, f"{scene_id}.npz"), image=sar)

        print(f"Generated {scene_id}: cloud fraction = {mask.mean():.3f}")

    print(f"\nDone. Generated {args.num_scenes} synthetic scenes under {args.out_dir}")


if __name__ == "__main__":
    main()
