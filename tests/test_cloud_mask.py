import numpy as np

from src.data.cloud_mask import CloudMaskConfig, cloud_fraction, combined_occlusion_mask, detect_clouds
from src.data.synthetic_clouds import SyntheticCloudGenerator


def _make_fake_scene_with_bright_patch(size=128):
    scene = np.random.rand(4, size, size).astype(np.float32) * 0.3 + 0.2
    # Force a bright, smooth, low-NDVI patch (cloud-like) in one corner
    scene[:, :40, :40] = 0.95
    return scene


def test_detect_clouds_finds_bright_smooth_region():
    scene = _make_fake_scene_with_bright_patch()
    mask = detect_clouds(scene, band_order=("G", "R", "NIR", "SWIR"), cfg=CloudMaskConfig())
    assert mask.shape == (128, 128)
    assert mask.dtype == np.uint8
    # Some detection should occur within the bright patch region
    assert mask[:40, :40].mean() > 0.3


def test_combined_occlusion_mask_shape_and_binary():
    scene = _make_fake_scene_with_bright_patch()
    mask = combined_occlusion_mask(scene)
    assert mask.shape == scene.shape[1:]
    assert set(np.unique(mask)).issubset({0, 1})


def test_cloud_fraction_bounds():
    mask = np.zeros((50, 50), dtype=np.uint8)
    assert cloud_fraction(mask) == 0.0
    mask[:] = 1
    assert cloud_fraction(mask) == 1.0


def test_synthetic_cloud_generator_produces_valid_output():
    gen = SyntheticCloudGenerator()
    clean = np.random.rand(4, 96, 96).astype(np.float32)
    cloudy, mask = gen.apply(clean)

    assert cloudy.shape == clean.shape
    assert mask.shape == clean.shape[1:]
    assert cloudy.min() >= 0.0 and cloudy.max() <= 1.0
    assert set(np.unique(mask)).issubset({0, 1})
    # Cloud application should meaningfully change some pixels
    assert not np.allclose(cloudy, clean)
