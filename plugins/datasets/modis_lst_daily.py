"""Daily MODIS Terra LST + QC ingestion from Microsoft Planetary Computer.

Streams MODIS/061 ``MOD11A1`` (Terra) daily daytime Land Surface Temperature
into a GeoZarr store with two **raw** variables:

- ``lst_day``  — ``LST_Day_1km`` digital number (Kelvin = DN x 0.02)
- ``qc_day``   — ``QC_Day`` quality bitfield

The data is left raw on purpose: the ``modis_lst_monthly`` workflow replicates
the GEE processing (QC masking, Kelvin->Celsius, monthly median, focal gap-fill)
server-side with openEO processes. Tiles are mosaicked and reprojected from the
native sinusoidal grid to EPSG:4326 by ``odc.stac.load``. Public, no credentials.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.streaming.protocol import GridSpec

_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
_COLLECTION = "modis-11A1-061"
_RES_DEG = 0.009  # ~1 km in degrees
_BANDS = ("LST_Day_1km", "QC_Day")


def _load_day(day: str, bbox: list[float]) -> xr.Dataset | None:
    """Mosaic + reproject the Terra MOD11A1 tiles for one day over bbox, or None."""
    import odc.stac
    import planetary_computer as pc
    import pystac_client

    catalog = pystac_client.Client.open(_STAC_URL, modifier=pc.sign_inplace)
    search = catalog.search(collections=[_COLLECTION], bbox=bbox, datetime=f"{day}/{day}")
    # MODIS Terra granules are MOD11A1.*; the collection also holds Aqua (MYD11A1).
    items = [it for it in search.items() if it.id.startswith("MOD")]
    if not items:
        return None
    ds = odc.stac.load(
        items,
        bands=_BANDS,
        bbox=bbox,
        crs="EPSG:4326",
        resolution=_RES_DEG,
        resampling="nearest",
    )
    return ds.isel(time=0, drop=True) if "time" in ds.dims else ds


class ModisLstDailyPlugin:
    """Streaming plugin for daily MODIS Terra LST + QC over the instance extent."""

    max_concurrency = 1
    commit_batch_size = 1

    def __init__(self, **_: object) -> None:
        self._grid: tuple[int, int] | None = None

    async def probe(self, bbox: list[float], **_: Any) -> GridSpec:
        # Sample a recent clear-ish day to pin the output grid shape.
        sample = await asyncio.to_thread(self._first_available, ["2016-04-15", "2016-04-16", "2016-04-17"], bbox)
        self._grid = (int(sample.sizes["latitude"]), int(sample.sizes["longitude"]))
        sample.close()
        return GridSpec(shape=self._grid, crs=4326, dtype=np.dtype("float32"), nodata=float("nan"), time_dim="t")

    async def periods(self, start: str, end: str) -> list[str]:
        d0 = date.fromisoformat(str(start)[:10])
        d1 = date.fromisoformat(str(end)[:10])
        if d0 > d1:
            return []
        return [(d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]

    async def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        day = str(period_id)[:10]
        ds = await asyncio.to_thread(_load_day, day, bbox)
        if ds is None:
            ds = self._empty_like(bbox)
        ds = self._normalize(ds)
        ds = ds.expand_dims(t=[np.datetime64(day)])
        return ds.load()

    def _first_available(self, days: list[str], bbox: list[float]) -> xr.Dataset:
        for day in days:
            ds = _load_day(day, bbox)
            if ds is not None:
                return ds.compute()
        raise RuntimeError(f"No MODIS Terra LST tiles for any of {days} over {bbox}")

    def _normalize(self, ds: xr.Dataset) -> xr.Dataset:
        rename = {k: v for k, v in (("longitude", "x"), ("latitude", "y")) if k in ds.dims}
        rename.update({b: o for b, o in (("LST_Day_1km", "lst_day"), ("QC_Day", "qc_day")) if b in ds})
        ds = ds.rename(rename)
        # Ensure y ascending for the map viewer.
        if "y" in ds and float(ds["y"].values[0]) > float(ds["y"].values[-1]):
            ds = ds.isel(y=slice(None, None, -1))
        for v in ("lst_day", "qc_day"):
            if v in ds:
                ds[v] = ds[v].astype("float32")
        return ds

    def _empty_like(self, bbox: list[float]) -> xr.Dataset:
        ny, nx = self._grid or (1, 1)
        empty = np.full((ny, nx), np.nan, dtype="float32")
        xmin, ymin, xmax, ymax = map(float, bbox)
        return xr.Dataset(
            {"lst_day": (("y", "x"), empty), "qc_day": (("y", "x"), empty.copy())},
            coords={"y": np.linspace(ymin, ymax, ny), "x": np.linspace(xmin, xmax, nx)},
        )
