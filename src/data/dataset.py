"""
PyTorch Dataset(s) for the LISS-IV cloud removal task.

`CloudRemovalDataset` loads pre-extracted .npz patches (see
`src.data.preprocessing`) containing:
    cloudy       (C_opt, H, W) float32, normalized [0,1]
    cloud_mask   (H, W) uint8, 1=occluded
    cloud_free   (C_opt, H, W) float32  [optional, only for supervised pairs]
    sar          (C_sar, H, W) float32  [optional auxiliary]

It also supports on-the-fly synthetic cloud simulation, which is critical
in practice: perfectly co-registered cloudy/cloud-free LISS-IV pairs of the
*same* scene are rare, so most training relies on synthetically clouding
real cloud-free scenes with realistic cloud shape/opacity, augmented with a
smaller set of genuine cloudy/cloud-free pairs where available.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augmentation import apply_joint_augmentation, build_augmentation_pipeline
from src.data.synthetic_clouds import SyntheticCloudGenerator


class CloudRemovalDataset(Dataset):
    def __init__(
        self,
        patch_dir: str,
        scene_ids: Optional[List[str]] = None,
        cfg=None,
        split: str = "train",
        use_synthetic_clouds: bool = True,
        synthetic_cloud_prob: float = 0.5,
    ):
        self.patch_dir = patch_dir
        self.cfg = cfg
        self.split = split
        self.use_synthetic_clouds = use_synthetic_clouds
        self.synthetic_cloud_prob = synthetic_cloud_prob

        all_files = sorted(glob.glob(os.path.join(patch_dir, "*.npz")))
        if scene_ids is not None:
            self.files = [f for f in all_files if any(sid in Path(f).name for sid in scene_ids)]
        else:
            self.files = all_files

        if not self.files:
            raise RuntimeError(f"No patch files found in {patch_dir} for split={split}")

        self.augment = split == "train" and cfg is not None
        self.aug_pipeline = build_augmentation_pipeline(cfg) if self.augment else None
        self.cloud_sim = SyntheticCloudGenerator()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data = np.load(self.files[idx])
        cloudy = data["cloudy"].astype(np.float32)
        cloud_mask = data["cloud_mask"].astype(np.uint8)
        cloud_free = data["cloud_free"].astype(np.float32) if "cloud_free" in data else None
        sar = data["sar"].astype(np.float32) if "sar" in data else None

        # Synthetic cloud injection: when we have a genuine cloud-free target,
        # we can optionally re-cloud it with a *synthetic* mask/opacity to
        # massively expand effective supervised training pairs.
        if (
            self.use_synthetic_clouds
            and cloud_free is not None
            and np.random.rand() < self.synthetic_cloud_prob
        ):
            cloudy, cloud_mask = self.cloud_sim.apply(cloud_free)

        if cloud_free is None:
            # No ground truth available (real unpaired cloudy scene) -> use
            # cloudy image itself as a weak target placeholder; the training
            # loop masks the loss to non-occluded pixels only in this case.
            cloud_free = cloudy.copy()

        if self.augment:
            out = apply_joint_augmentation(
                self.aug_pipeline, cloudy, cloud_mask, cloud_free, sar
            )
            cloudy, cloud_mask, cloud_free = out["cloudy"], out["cloud_mask"], out["cloud_free"]
            sar = out.get("sar", sar)

        sample = {
            "cloudy": torch.from_numpy(np.ascontiguousarray(cloudy)).float(),
            "cloud_mask": torch.from_numpy(np.ascontiguousarray(cloud_mask)).float().unsqueeze(0),
            "cloud_free": torch.from_numpy(np.ascontiguousarray(cloud_free)).float(),
            "has_gt": torch.tensor(1.0 if "cloud_free" in data else 0.0),
        }
        if sar is not None:
            sample["sar"] = torch.from_numpy(np.ascontiguousarray(sar)).float()
        else:
            c_sar = self.cfg.model.generator.in_channels_sar if self.cfg else 2
            sample["sar"] = torch.zeros((c_sar, *cloudy.shape[1:]), dtype=torch.float32)

        return sample


def build_dataloaders(cfg):
    """Convenience factory returning train/val/test DataLoaders from config."""
    import json

    from torch.utils.data import DataLoader

    manifest_path = os.path.join(cfg.paths.processed_dir, "splits.json")
    with open(manifest_path) as f:
        splits = json.load(f)

    loaders = {}
    for split in ["train", "val", "test"]:
        ds = CloudRemovalDataset(
            patch_dir=cfg.paths.patches_dir,
            scene_ids=splits[split],
            cfg=cfg,
            split=split,
            use_synthetic_clouds=(split == "train"),
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.training.batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.project.num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders
