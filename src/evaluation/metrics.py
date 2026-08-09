"""
Quantitative evaluation metrics standard in remote-sensing image restoration
literature: PSNR, SSIM, SAM (Spectral Angle Mapper), RMSE, ERGAS, MAE.
Implemented on numpy arrays shaped (C, H, W) in [0, 1] for framework
independence (usable in both GAN and diffusion pipelines, and standalone
evaluation scripts).
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0, max_db: float = 80.0) -> float:
    """PSNR in dB, capped at `max_db`. Near-perfect reconstructions (MSE ~0,
    e.g. clear patches with no occlusion to reconstruct) would otherwise
    report as +inf, which is both uninformative and breaks downstream JSON/
    CSV aggregation (mean/std over inf is NaN/inf)."""
    val = sk_psnr(target, pred, data_range=data_range)
    if not np.isfinite(val):
        return max_db
    return float(min(val, max_db))


def compute_ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    # skimage expects channel-last for multichannel SSIM
    pred_hwc = np.transpose(pred, (1, 2, 0))
    target_hwc = np.transpose(target, (1, 2, 0))
    return float(
        sk_ssim(target_hwc, pred_hwc, data_range=data_range, channel_axis=-1)
    )


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def compute_sam(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Spectral Angle Mapper in degrees, averaged over all pixels. Lower is
    better; measures spectral-shape fidelity independent of brightness."""
    c, h, w = pred.shape
    pred_flat = pred.reshape(c, -1)
    target_flat = target.reshape(c, -1)

    dot = np.sum(pred_flat * target_flat, axis=0)
    pred_norm = np.linalg.norm(pred_flat, axis=0)
    target_norm = np.linalg.norm(target_flat, axis=0)
    denom = np.clip(pred_norm * target_norm, eps, None)
    cos_angle = np.clip(dot / denom, -1, 1)
    angles = np.arccos(cos_angle)
    return float(np.degrees(np.mean(angles)))


def compute_ergas(pred: np.ndarray, target: np.ndarray, resolution_ratio: float = 1.0) -> float:
    """Erreur Relative Globale Adimensionnelle de Synthese — a standard
    fusion-quality index in remote sensing literature, band-normalized
    global error metric."""
    c = pred.shape[0]
    total = 0.0
    for i in range(c):
        band_rmse = compute_rmse(pred[i], target[i])
        band_mean = np.mean(target[i]) + 1e-8
        total += (band_rmse / band_mean) ** 2
    return float(100.0 * resolution_ratio * np.sqrt(total / c))


METRIC_FUNCS = {
    "psnr": compute_psnr,
    "ssim": compute_ssim,
    "sam": compute_sam,
    "rmse": compute_rmse,
    "mae": compute_mae,
    "ergas": compute_ergas,
}


def compute_all_metrics(pred: np.ndarray, target: np.ndarray, metric_names: list[str]) -> dict:
    return {name: METRIC_FUNCS[name](pred, target) for name in metric_names if name in METRIC_FUNCS}
