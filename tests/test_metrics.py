import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    compute_ergas,
    compute_mae,
    compute_psnr,
    compute_rmse,
    compute_sam,
    compute_ssim,
)


def test_psnr_identical_images_is_capped_not_infinite():
    img = np.random.rand(4, 64, 64).astype(np.float32)
    psnr = compute_psnr(img, img)
    assert np.isfinite(psnr)
    assert psnr > 40  # near-identical -> capped high PSNR (see compute_psnr max_db)


def test_ssim_identical_images_equals_one():
    img = np.random.rand(4, 64, 64).astype(np.float32)
    ssim = compute_ssim(img, img)
    assert ssim == pytest.approx(1.0, abs=1e-5)


def test_sam_zero_for_identical_spectra():
    img = np.random.rand(4, 32, 32).astype(np.float32) + 0.1
    sam = compute_sam(img, img)
    assert sam == pytest.approx(0.0, abs=1e-1)  # float32 norm precision noise


def test_sam_nonzero_for_different_spectra():
    a = np.ones((4, 8, 8), dtype=np.float32)
    b = a.copy()
    b[0] *= 5.0  # distort one band's relative magnitude -> spectral angle changes
    sam = compute_sam(a, b)
    assert sam > 0


def test_rmse_and_mae_zero_for_identical():
    img = np.random.rand(4, 32, 32).astype(np.float32)
    assert compute_rmse(img, img) == pytest.approx(0.0, abs=1e-6)
    assert compute_mae(img, img) == pytest.approx(0.0, abs=1e-6)


def test_ergas_nonnegative():
    a = np.random.rand(4, 32, 32).astype(np.float32) + 0.1
    b = np.random.rand(4, 32, 32).astype(np.float32) + 0.1
    assert compute_ergas(a, b) >= 0


def test_compute_all_metrics_returns_requested_keys():
    a = np.random.rand(4, 32, 32).astype(np.float32) + 0.1
    b = a + 0.01
    result = compute_all_metrics(a, b, ["psnr", "ssim", "sam"])
    assert set(result.keys()) == {"psnr", "ssim", "sam"}
