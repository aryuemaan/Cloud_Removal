"""
Inference entry point: run cloud removal on a full georeferenced LISS-IV
scene (GeoTIFF), optionally fused with a co-registered Sentinel-1 SAR scene,
and write an analysis-ready cloud-free GeoTIFF preserving the original
CRS/transform.

Usage:
    python -m src.inference.predict \
        --config config/config.yaml \
        --checkpoint checkpoints/ckpt_best.pt \
        --input path/to/cloudy_scene.tif \
        --sar path/to/sar_scene.tif \
        --output outputs/reconstructed_scene.tif
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.data.cloud_mask import CloudMaskConfig, combined_occlusion_mask
from src.inference.sliding_window import SlidingWindowPredictor
from src.models.model_factory import build_generator
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.geo_utils import percentile_normalize, read_raster, reproject_to_reference, write_raster
from src.utils.logger import get_logger

logger = get_logger("inference")


def load_model(cfg, checkpoint_path: str, device: torch.device, use_ema: bool = True):
    from src.utils.checkpoint import reconcile_model_config

    state = load_checkpoint(checkpoint_path, map_location=str(device))
    cfg = reconcile_model_config(cfg, state)
    model = build_generator(cfg).to(device)
    key = "ema_generator" if use_ema and "ema_generator" in state else "generator"
    model.load_state_dict(state[key])
    model.eval()
    logger.info(f"Loaded generator weights ({key}) from {checkpoint_path}")
    return model, cfg


def to_tensor_range(arr01: np.ndarray) -> np.ndarray:
    """[0,1] -> [-1,1], matching training-time normalization convention."""
    return arr01 * 2 - 1


def from_tensor_range(arr_pm1: np.ndarray) -> np.ndarray:
    return np.clip((arr_pm1 + 1) / 2, 0, 1)


def run_inference(
    cfg,
    checkpoint_path: str | None,
    input_path: str,
    output_path: str,
    sar_path: str | None = None,
    model=None,
    device: "torch.device | None" = None,
):
    """
    Run cloud-removal inference on one scene.

    If `model` is already loaded (e.g. a long-lived service that loaded it
    once at startup), pass it in directly along with `device` to avoid the
    cost — and, more importantly, the *doubled memory footprint* — of
    reloading the checkpoint from disk on every call. `checkpoint_path` is
    then optional and ignored. This is the path the FastAPI service uses;
    the CLI entrypoint below always loads fresh from `checkpoint_path`.
    """
    if device is None:
        device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")
    if model is None:
        if not checkpoint_path:
            raise ValueError("Either `model` or `checkpoint_path` must be provided.")
        model, cfg = load_model(cfg, checkpoint_path, device)

    cloudy_raw, profile = read_raster(input_path)
    cloudy_norm = percentile_normalize(
        cloudy_raw,
        lower=cfg.data.normalization.lower_percentile,
        upper=cfg.data.normalization.upper_percentile,
    )

    mask_cfg = CloudMaskConfig()
    band_order = tuple(cfg.data.get("band_order", ("G", "R", "NIR", "SWIR")))
    cloud_mask = combined_occlusion_mask(cloudy_norm, band_order, mask_cfg)
    logger.info(f"Detected cloud coverage: {cloud_mask.mean() * 100:.2f}% of scene")

    sar_norm = None
    if sar_path:
        reproject_to_reference(sar_path, profile, "outputs/_tmp_sar_coreg.tif")
        sar_raw, _ = read_raster("outputs/_tmp_sar_coreg.tif")
        sar_norm = percentile_normalize(sar_raw)

    predictor = SlidingWindowPredictor(
        window_size=cfg.inference.window_size,
        stride=cfg.inference.stride,
        blend_mode=cfg.inference.blend_mode,
        device=str(device),
    )

    def predict_fn(cloudy_batch, mask_batch, sar_batch):
        with torch.no_grad():
            c_t = torch.from_numpy(to_tensor_range(cloudy_batch)).float().to(device)
            m_t = torch.from_numpy(mask_batch).float().to(device)
            s_t = torch.from_numpy(to_tensor_range(sar_batch) if sar_batch.max() > 0 else sar_batch).float().to(device)
            out, _ = model(c_t, m_t, s_t)
            return from_tensor_range(out.cpu().numpy())

    reconstructed = predictor.predict(
        predict_fn,
        cloudy_norm,
        cloud_mask,
        sar=sar_norm,
        batch_size=cfg.inference.batch_size,
        tta=cfg.inference.tta,
    )

    # Rescale back to original radiometric range approximately using the
    # same percentile bounds computed at normalization time, for downstream
    # compatibility with standard remote-sensing tools expecting native DN.
    write_raster(output_path, (reconstructed * 10000).astype(np.uint16), profile)
    logger.info(f"Wrote reconstructed scene to {output_path}")
    return reconstructed, cloud_mask


def main():
    parser = argparse.ArgumentParser(description="Run cloud removal inference on a LISS-IV scene.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--sar", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_inference(cfg, args.checkpoint, args.input, args.output, args.sar)


if __name__ == "__main__":
    main()
