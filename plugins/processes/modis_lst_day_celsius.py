"""Clean raw MODIS Terra LST into quality-masked degrees Celsius.

Replicates the GEE ``processLST`` step from Bipin Acharya's script:

- keep only good-quality pixels using the MOD11A1 ``QC_Day`` bitfield
  (mandatory QA ``bits 0-1 == 0`` and LST error ``bits 6-7 <= 1``),
- convert ``LST_Day_1km`` digital numbers to degrees Celsius (``x 0.02 - 273.15``),
- mask fill (DN == 0) and poor-quality pixels to NaN.

Input is the two-band ``modis_lst_daily`` cube (bands ``lst_day`` and ``qc_day``);
output is a single-variable ``(t, y, x)`` Celsius cube.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.process import process


def _band(data: Any, name: str) -> xr.DataArray:
    if "bands" in getattr(data, "dims", ()):
        return data.sel(bands=name).drop_vars("bands", errors="ignore")
    if isinstance(data, xr.Dataset):
        return data[name]
    return data  # single-band DataArray already


@process(
    summary="Quality-mask MODIS LST_Day and convert to degrees Celsius",
    description=(
        "Takes the raw two-band MODIS daily cube (lst_day = LST_Day_1km DN, "
        "qc_day = QC_Day bitfield) and returns good-quality daytime land surface "
        "temperature in degrees Celsius (LST x 0.02 - 273.15). Pixels failing the "
        "MOD11A1 mandatory-QA / LST-error flags, or fill values, become NaN."
    ),
)
def modis_lst_day_celsius(data: Any) -> xr.DataArray:
    lst = _band(data, "lst_day")
    qc = _band(data, "qc_day")

    qc_int = np.nan_to_num(qc.values, nan=255.0).astype("int64")
    mandatory_ok = (qc_int & 0b11) == 0          # bits 0-1: LST produced, good quality
    lst_error_ok = ((qc_int >> 6) & 0b11) <= 1   # bits 6-7: average LST error <= 2 K
    good = mandatory_ok & lst_error_ok

    dn = lst.values.astype("float32")
    celsius = dn * 0.02 - 273.15
    celsius = np.where(good & (dn > 0), celsius, np.nan).astype("float32")

    out = lst.copy(data=celsius)
    out = out.drop_vars("bands", errors="ignore")
    out.name = "lst_day"
    out.attrs = {"long_name": "MODIS daytime land surface temperature", "units": "degC"}
    return out
