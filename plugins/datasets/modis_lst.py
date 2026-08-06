"""MODIS MOD11A1 monthly daytime Land Surface Temperature for Nepal.

Ingests pre-processed monthly LST GeoTIFFs supplied by B. Acharya (PhD). The
upstream processing (see modis_lst_2016/LST_gap_fill.txt) takes MODIS/061
MOD11A1 daily ``LST_Day_1km``, applies the MODIS quality flags, converts Kelvin
to degrees Celsius (``x 0.02 - 273.15``), takes the monthly median, gap-fills
small holes with a focal mean, and clips to Nepal. The files are therefore final
products: this plugin just reads one GeoTIFF per month and writes them to a
GeoZarr store with a ``t`` time dimension.

Files live next to this plugin: ``modis_lst_2016/LST_Day_2016_MM.tif``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rioxarray  # noqa: F401  # registers the .rio accessor / GeoTIFF reader
import xarray as xr

from open_climate_service.streaming import BaseDatasetPlugin

_DATA_DIR = Path(__file__).parent / "modis_lst_2016"
_YEAR = 2016


def _path_for(year: int, month: int) -> Path:
    return _DATA_DIR / f"LST_Day_{year}_{month:02d}.tif"


class ModisLstMonthlyPlugin(BaseDatasetPlugin):
    """Streaming plugin for the monthly MODIS LST (day) GeoTIFFs."""

    max_concurrency = 1
    commit_batch_size = 1

    def __init__(self, variable: str = "lst_day", **_: object) -> None:
        self.variable = variable

    async def periods(self, start: str, end: str) -> list[str]:
        start_ym = str(start)[:7]
        end_ym = str(end)[:7]
        periods: list[str] = []
        for month in range(1, 13):
            ym = f"{_YEAR}-{month:02d}"
            if start_ym <= ym <= end_ym and _path_for(_YEAR, month).exists():
                periods.append(f"{ym}-01")
        return periods

    async def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        month = int(str(period_id)[5:7])
        return self._read_month(_YEAR, month)

    def _read_month(self, year: int, month: int) -> xr.Dataset:
        path = _path_for(year, month)
        da = rioxarray.open_rasterio(path, masked=True).squeeze("band", drop=True).astype("float32")
        ds = da.to_dataset(name=self.variable)
        ds = ds.expand_dims(t=[np.datetime64(f"{year}-{month:02d}-01")])
        return ds.load()
