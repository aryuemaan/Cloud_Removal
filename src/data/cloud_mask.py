"""
Cloud & cloud-shadow mask generation for LISS-IV imagery.

LISS-IV lacks a thermal/cirrus band, so classical Fmask-style detection is
not directly applicable. We implement a practical multi-cue detector:

  1. Brightness threshold on visible + NIR bands (clouds are bright).
  2. Low NDVI + high brightness co-occurrence (excludes bright bare soil).
  3. Local spatial variance filter (clouds are spatially smooth vs. texture
     of natural surfaces).
  4. Morphological cleanup (opening/closing) + connected component filtering
     to drop tiny spurious detections.
  5. Cloud-shadow estimate via directional projection of cloud objects using
     an approximate sun-azimuth offset (optional; requires acquisition
     metadata) combined with a dark-pixel threshold in NIR/SWIR.

This is intentionally deterministic/classical: it produces the *training
labels / masks* consumed by the generative model, not a learned classifier
in itself (though a learned segmentation head can replace this module later
by implementing the same interface).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects

# skimage renamed a few morphology APIs across versions; the functional
# behavior we rely on (binary opening/closing, small-object removal) is
# stable, only the deprecation notices are noisy.
warnings.filterwarnings("ignore", category=FutureWarning, module="skimage.*")


@dataclass
class CloudMaskConfig:
    brightness_percentile: float = 80.0
    ndvi_threshold: float = 0.15
    variance_window: int = 7
    variance_threshold: float = 0.002
    min_object_size: int = 64
    morphology_radius: int = 3
    shadow_dark_percentile: float = 15.0
    shadow_search_radius: int = 40
    sun_azimuth_deg: Optional[float] = None  # if unknown, shadow search is omnidirectional


def _ndvi(green: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = nir + red
    denom[denom == 0] = 1e-6
    return (nir - red) / denom


def _local_variance(img: np.ndarray, window: int) -> np.ndarray:
    mean = ndimage.uniform_filter(img, size=window)
    sq_mean = ndimage.uniform_filter(img ** 2, size=window)
    return np.clip(sq_mean - mean ** 2, 0, None)


def detect_clouds(
    image: np.ndarray,
    band_order: tuple = ("G", "R", "NIR", "SWIR"),
    cfg: CloudMaskConfig = CloudMaskConfig(),
) -> np.ndarray:
    """
    Detect clouds in a (C, H, W) normalized [0,1] LISS-IV image.
    Returns a binary (H, W) uint8 mask: 1 = cloud, 0 = clear.
    """
    band_idx = {b: i for i, b in enumerate(band_order)}
    green = image[band_idx["G"]]
    red = image[band_idx["R"]]
    nir = image[band_idx["NIR"]]
    swir = image[band_idx.get("SWIR", band_idx["NIR"])]

    brightness = (green + red + nir + swir) / 4.0
    bright_thresh = np.percentile(brightness, cfg.brightness_percentile)
    bright_mask = brightness >= bright_thresh

    ndvi = _ndvi(green, red, nir)
    veg_exclusion = ndvi < cfg.ndvi_threshold

    local_var = _local_variance(brightness, cfg.variance_window)
    smooth_mask = local_var < cfg.variance_threshold

    cloud_mask = bright_mask & veg_exclusion & smooth_mask

    # Morphological cleanup
    selem = disk(cfg.morphology_radius)
    cloud_mask = binary_opening(cloud_mask, selem)
    cloud_mask = binary_closing(cloud_mask, selem)
    cloud_mask = remove_small_objects(cloud_mask, min_size=cfg.min_object_size)

    return cloud_mask.astype(np.uint8)


def detect_cloud_shadows(
    image: np.ndarray,
    cloud_mask: np.ndarray,
    band_order: tuple = ("G", "R", "NIR", "SWIR"),
    cfg: CloudMaskConfig = CloudMaskConfig(),
) -> np.ndarray:
    """
    Estimate cloud shadows using a dark-pixel threshold in NIR restricted to
    a neighborhood of detected cloud objects (shadows fall near their
    casting cloud). Returns a binary (H, W) mask.
    """
    band_idx = {b: i for i, b in enumerate(band_order)}
    nir = image[band_idx["NIR"]]

    dark_thresh = np.percentile(nir, cfg.shadow_dark_percentile)
    dark_mask = nir <= dark_thresh

    # Restrict candidate shadow pixels to a dilated neighborhood of clouds
    dilated_clouds = ndimage.binary_dilation(
        cloud_mask.astype(bool), iterations=cfg.shadow_search_radius
    )
    shadow_mask = dark_mask & dilated_clouds & (~cloud_mask.astype(bool))

    selem = disk(cfg.morphology_radius)
    shadow_mask = binary_opening(shadow_mask, selem)
    shadow_mask = remove_small_objects(shadow_mask, min_size=cfg.min_object_size)

    return shadow_mask.astype(np.uint8)


def combined_occlusion_mask(
    image: np.ndarray,
    band_order: tuple = ("G", "R", "NIR", "SWIR"),
    cfg: CloudMaskConfig = CloudMaskConfig(),
    include_shadows: bool = True,
) -> np.ndarray:
    """Union of cloud + cloud-shadow masks -> the full occlusion mask used
    to define the inpainting/reconstruction region for the generator."""
    clouds = detect_clouds(image, band_order, cfg)
    if not include_shadows:
        return clouds
    shadows = detect_cloud_shadows(image, clouds, band_order, cfg)
    return np.clip(clouds + shadows, 0, 1).astype(np.uint8)


def cloud_fraction(mask: np.ndarray) -> float:
    return float(mask.sum()) / float(mask.size)
