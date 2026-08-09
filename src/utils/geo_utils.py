"""
Geospatial utility functions for reading/writing georeferenced rasters,
reprojection, co-registration, and windowed I/O — the backbone of any
operational (as opposed to toy) satellite imagery pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject, transform_bounds
    from rasterio.windows import Window
    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover - environment without GDAL available
    _HAS_RASTERIO = False


def _require_rasterio():
    if not _HAS_RASTERIO:
        raise ImportError(
            "rasterio/GDAL is required for this operation. "
            "Install via `pip install rasterio` (with GDAL system libs) or "
            "`conda install -c conda-forge rasterio`."
        )


def read_raster(path: str | Path, bands: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, dict]:
    """Read a raster into a (C, H, W) float32 array plus its profile/metadata."""
    _require_rasterio()
    with rasterio.open(path) as src:
        arr = src.read(bands) if bands else src.read()
        profile = src.profile.copy()
    return arr.astype(np.float32), profile


def write_raster(path: str | Path, array: np.ndarray, profile: dict) -> None:
    """Write a (C, H, W) array to disk using an existing rasterio profile."""
    _require_rasterio()
    profile = profile.copy()
    profile.update(count=array.shape[0], dtype=array.dtype)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)


def reproject_to_reference(
    src_path: str | Path,
    ref_profile: dict,
    dst_path: str | Path,
    resampling: str = "bilinear",
) -> None:
    """
    Co-register a source raster (e.g. Sentinel-1 SAR) onto the CRS/grid of a
    reference raster (e.g. LISS-IV scene). Essential for multi-sensor fusion.
    """
    _require_rasterio()
    resampling_enum = getattr(Resampling, resampling)
    with rasterio.open(src_path) as src:
        dst_transform = ref_profile["transform"]
        dst_crs = ref_profile["crs"]
        dst_height = ref_profile["height"]
        dst_width = ref_profile["width"]

        dst_kwargs = src.meta.copy()
        dst_kwargs.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
            }
        )
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **dst_kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling_enum,
                )


def compute_windows(width: int, height: int, patch_size: int, overlap: int):
    """Yield rasterio Windows tiling an image with the given overlap (for
    memory-efficient sliding-window inference over large GeoTIFFs)."""
    _require_rasterio()
    stride = patch_size - overlap
    for top in range(0, height, stride):
        for left in range(0, width, stride):
            w = min(patch_size, width - left)
            h = min(patch_size, height - top)
            if w <= 0 or h <= 0:
                continue
            yield Window(left, top, w, h)


def percentile_normalize(
    arr: np.ndarray, lower: float = 1, upper: float = 99, per_channel: bool = True
) -> np.ndarray:
    """Robust percentile-based normalization to [0, 1], per-channel."""
    arr = arr.astype(np.float32)
    out = np.empty_like(arr)
    if per_channel:
        for c in range(arr.shape[0]):
            lo = np.percentile(arr[c], lower)
            hi = np.percentile(arr[c], upper)
            hi = max(hi, lo + 1e-6)
            out[c] = np.clip((arr[c] - lo) / (hi - lo), 0, 1)
    else:
        lo = np.percentile(arr, lower)
        hi = np.percentile(arr, upper)
        hi = max(hi, lo + 1e-6)
        out = np.clip((arr - lo) / (hi - lo), 0, 1)
    return out
