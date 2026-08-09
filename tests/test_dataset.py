import numpy as np

from src.data.preprocessing import PreprocessConfig, extract_patches_from_scene, normalize_scene, split_scenes


def test_normalize_scene_range():
    scene = (np.random.rand(4, 128, 128).astype(np.float32) * 5000)
    cfg = PreprocessConfig()
    normed = normalize_scene(scene, cfg)
    assert normed.min() >= 0.0
    assert normed.max() <= 1.0


def test_extract_patches_from_scene_basic():
    cloudy = np.random.rand(4, 300, 300).astype(np.float32)
    clean = np.random.rand(4, 300, 300).astype(np.float32)
    cfg = PreprocessConfig(patch_size=64, overlap=16, cloud_fraction_min=0.0, cloud_fraction_max=1.0, clear_patch_keep_ratio=1.0)
    patches = extract_patches_from_scene(cloudy, clean, None, cfg)
    assert len(patches) > 0
    for p in patches:
        assert p["cloudy"].shape == (4, 64, 64)
        assert p["cloud_free"].shape == (4, 64, 64)
        assert p["cloud_mask"].shape == (64, 64)


def test_split_scenes_disjoint_and_covers_all():
    ids = [f"scene_{i}" for i in range(20)]
    splits = split_scenes(ids, ratios=(0.7, 0.15, 0.15), seed=1)
    all_ids = splits["train"] + splits["val"] + splits["test"]
    assert sorted(all_ids) == sorted(ids)
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])
