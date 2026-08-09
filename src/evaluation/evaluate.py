"""
Batch evaluation over the held-out test split: computes quantitative metrics
(PSNR/SSIM/SAM/RMSE/ERGAS/MAE) and optionally saves qualitative comparison
panels, then writes a summary CSV/JSON report — the artifact typically
attached to a model release / experiment tracking entry.

Usage:
    python -m src.evaluation.evaluate --config config/config.yaml --checkpoint checkpoints/ckpt_best.pt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from src.data.dataset import CloudRemovalDataset
from src.evaluation.metrics import compute_all_metrics
from src.models.model_factory import build_generator
from src.utils.checkpoint import load_checkpoint, reconcile_model_config
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.visualization import save_comparison_panel

logger = get_logger("evaluate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out_dir", type=str, default="outputs/evaluation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")

    state = load_checkpoint(args.checkpoint, map_location=str(device))
    cfg = reconcile_model_config(cfg, state)

    with open(os.path.join(cfg.paths.processed_dir, "splits.json")) as f:
        splits = json.load(f)

    dataset = CloudRemovalDataset(
        patch_dir=cfg.paths.patches_dir,
        scene_ids=splits[args.split],
        cfg=cfg,
        split="test",
        use_synthetic_clouds=False,
    )

    model = build_generator(cfg).to(device)
    model.load_state_dict(state.get("ema_generator", state["generator"]))
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    all_metrics = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            cloudy = sample["cloudy"].unsqueeze(0).to(device)
            mask = sample["cloud_mask"].unsqueeze(0).to(device)
            sar = sample["sar"].unsqueeze(0).to(device)
            target = sample["cloud_free"].unsqueeze(0).to(device)

            pred, _ = model(cloudy, mask, sar)

            pred_np = ((pred[0] + 1) / 2).clamp(0, 1).cpu().numpy()
            target_np = ((target[0] + 1) / 2).clamp(0, 1).cpu().numpy()

            metrics = compute_all_metrics(pred_np, target_np, cfg.evaluation.metrics)
            metrics["sample_idx"] = idx
            all_metrics.append(metrics)

            if cfg.evaluation.save_visual_comparisons and idx < cfg.evaluation.num_visual_samples:
                cloudy_np = ((cloudy[0] + 1) / 2).clamp(0, 1).cpu().numpy()
                mask_np = mask[0].cpu().numpy()
                save_comparison_panel(
                    cloudy_np, mask_np, pred_np, target_np,
                    out_path=os.path.join(args.out_dir, f"comparison_{idx:04d}.png"),
                    title=f"Sample {idx} | PSNR={metrics.get('psnr', 0):.2f} dB",
                )

    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(args.out_dir, "metrics_per_sample.csv"), index=False)

    summary = df.drop(columns=["sample_idx"]).agg(["mean", "std", "min", "max"]).to_dict()
    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Evaluation complete. Summary:\n{json.dumps(summary, indent=2)}")
    logger.info(f"Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
