"""ERA5-Land monthly mean of daily maximum 2m temperature.

The CDS ``reanalysis-era5-land-monthly-means`` product only provides the monthly
mean of the instantaneous 2m temperature, not a daily-maximum statistic. This
plugin derives "monthly mean of daily maximum temperature": for each month it
fetches that month's daily-maximum 2m temperature from the CDS
``derived-era5-land-daily-statistics`` dataset (daily_statistic = daily_maximum)
and averages over the days. Output in degrees Celsius. Requires CDS credentials
(``~/.cdsapirc``).
"""

from __future__ import annotations

import asyncio
import calendar
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import xarray as xr
from ecmwf.datastores import Client as CdsClient

from open_climate_service.streaming import BaseDatasetPlugin, normalize_period


def _month_starts(start: str, end: str) -> list[str]:
    s = date.fromisoformat(str(start)[:7] + "-01")
    e = date.fromisoformat(str(end)[:7] + "-01")
    out: list[str] = []
    year, month = s.year, s.month
    while (year, month) <= (e.year, e.month):
        out.append(f"{year}-{month:02d}-01")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return out


class ERA5LandTmaxMonthlyPlugin(BaseDatasetPlugin):
    """Monthly mean of daily maximum 2m temperature, derived from CDS daily stats."""

    max_concurrency = 1
    commit_batch_size = 1

    def __init__(self, variable: str = "t2m_max", **_: Any) -> None:
        self.variable = variable

    async def periods(self, start: str, end: str) -> list[str]:
        return _month_starts(start, end)

    async def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        return await asyncio.to_thread(self._fetch_month_mean, period_id, tuple(map(float, bbox)))

    def _fetch_month_mean(self, period_id: str, bbox: tuple[float, float, float, float]) -> xr.Dataset:
        d = date.fromisoformat(str(period_id)[:10])
        year, month = d.year, d.month
        xmin, ymin, xmax, ymax = bbox
        _, last_day = calendar.monthrange(year, month)
        params = {
            "variable": ["2m_temperature"],
            "year": str(year),
            "month": str(month).zfill(2),
            "day": [str(x).zfill(2) for x in range(1, last_day + 1)],
            "daily_statistic": "daily_maximum",
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": [ymax, xmin, ymin, xmax],  # N, W, S, E
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "era5land_tmax.nc"
            CdsClient().submit("derived-era5-land-daily-statistics", params).download(str(target))
            ds = xr.open_dataset(target, engine="netcdf4").load()

        ds = ds[["t2m"]]
        t_dim = "valid_time" if "valid_time" in ds.dims else "time"
        # Mean over the month's daily maxima; Kelvin -> Celsius. Reducing the daily axis
        # away leaves a 2-D grid, so normalize_period stamps the month onto `t`.
        monthly = (ds["t2m"].mean(dim=t_dim) - 273.15).astype("float32")
        # No bbox clip: the CDS request already restricts to the bbox via `area`, and the
        # returned netCDF carries no CRS for rioxarray to clip against.
        return normalize_period(monthly, variable=self.variable, period=f"{year}-{month:02d}-01")
