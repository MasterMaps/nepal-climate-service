"""NaN-aware focal-mean gap fill for raster cubes.

Replicates the GEE gap-fill step (``focalMean(circle, radius, iterations)`` then
``unmask(smooth)``): missing pixels are filled with the mean of valid neighbours
within a circular kernel, applied ``iterations`` times so the fill reaches a few
pixels into larger holes. Original (valid) pixels are kept unchanged.

Operates per ``(y, x)`` slice over any leading dimensions (e.g. ``t``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.process import process


def _circular_kernel(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return ((xx * xx + yy * yy) <= radius * radius).astype("float64")


def _focal_mean_nan(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    from scipy.ndimage import convolve

    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0)
    weighted = convolve(filled, kernel, mode="nearest")
    counts = convolve(valid.astype("float64"), kernel, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, weighted / counts, np.nan)


@process(
    summary="Fill missing raster pixels with an iterated NaN-aware focal mean",
    description=(
        "For each (y, x) slice, fills NaN pixels with the mean of valid neighbours "
        "within a circular kernel of the given radius (in pixels), applied "
        "`iterations` times. Valid pixels are left unchanged. Mirrors GEE's "
        "focalMean(circle) + unmask gap-fill."
    ),
)
def focal_fill(data: xr.DataArray, radius: float = 3, iterations: float = 2) -> xr.DataArray:
    kernel = _circular_kernel(int(radius))
    n_iter = int(iterations)

    dims = list(data.dims)
    yx = [d for d in dims if d in ("y", "x")]
    if len(yx) != 2:
        raise ValueError(f"focal_fill expects y and x dimensions, got {dims}")
    lead = [d for d in dims if d not in ("y", "x")]
    da = data.transpose(*lead, "y", "x")

    arr = da.values.astype("float64")
    ny, nx = arr.shape[-2], arr.shape[-1]
    flat = arr.reshape(-1, ny, nx)
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        original = flat[i]
        smooth = original
        for _ in range(n_iter):
            smooth = _focal_mean_nan(smooth, kernel)
        out[i] = np.where(np.isfinite(original), original, smooth)

    result = da.copy(data=out.reshape(arr.shape).astype("float32"))
    return result
