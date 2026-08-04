#!/usr/bin/env python3
"""Animate a published GeoZarr dataset using the same colormap as the /map viewer.

Requirements
------------
- ffmpeg must be on PATH:  brew install ffmpeg
- mapflow must be installed:  uv sync --group scripts

Usage
-----
    uv run --group scripts scripts/animate.py chirps3_precipitation_daily
    uv run --group scripts scripts/animate.py era5land_temperature_monthly -o /tmp/temp.mp4
    uv run --group scripts scripts/animate.py cams_pm25_hourly --start 2025-06-01T00 --end 2025-06-01T18
    uv run --group scripts scripts/animate.py clms_ndvi_dekadal --start 2024-01-01 --fps 4
"""

from __future__ import annotations

import argparse
import sys

BASE_URL = "http://localhost:8001"


def _resolve_cmap(name: str) -> str:
    """Return a valid matplotlib colormap name via case-insensitive lookup.

    chroma-js (used by the map viewer) resolves names like "rdbu_r" that
    matplotlib registers as "RdBu_r". This lookup finds the canonical name.
    """
    import matplotlib

    if name in matplotlib.colormaps:
        return name
    name_lower = name.lower()
    for registered in matplotlib.colormaps:
        if registered.lower() == name_lower:
            return registered
    print(f"  warning: colormap '{name}' not found in matplotlib, using viridis", file=sys.stderr)
    return "viridis"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate a climate dataset with the same style as the /map viewer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset_id", help="Published dataset ID (e.g. chirps3_precipitation_daily)")
    parser.add_argument("--output", "-o", help="Output video path (default: <dataset_id>.mp4)")
    parser.add_argument("--start", help="Start of time slice (ISO 8601)")
    parser.add_argument("--end", help="End of time slice (ISO 8601)")
    parser.add_argument("--url", default=BASE_URL, help=f"Service URL (default: {BASE_URL})")
    parser.add_argument("--fps", type=int, help="Frames per second (default: 24)")
    parser.add_argument("--width", type=int, default=1200, help="Output video width in pixels (default: 1200)")
    parser.add_argument("--cmap", help="Override colormap (matplotlib name, e.g. RdBu_r)")
    parser.add_argument("--vmin", type=float, help="Override color range minimum")
    parser.add_argument("--vmax", type=float, help="Override color range maximum")
    args = parser.parse_args()

    output = args.output or f"{args.dataset_id}.mp4"

    import httpx
    from open_climate_service import ClimateService

    # ── 1. Fetch STAC collection (same source the /map viewer reads) ─────────
    url = args.url.rstrip("/")
    resp = httpx.get(f"{url}/stac/collections/{args.dataset_id}")
    if not resp.is_success:
        print(
            f"error: collection '{args.dataset_id}' not found ({resp.status_code}).\n"
            f"       Is the dataset published at {url}?",
            file=sys.stderr,
        )
        sys.exit(1)
    collection = resp.json()

    renders = collection.get("renders", {}).get("default", {})
    colormap_name = args.cmap or renders.get("colormap_name", "viridis")
    rescale = renders.get("rescale", [[0, 100]])
    vmin = args.vmin if args.vmin is not None else float(rescale[0][0])
    vmax = args.vmax if args.vmax is not None else float(rescale[0][1])
    variable = renders.get("open_climate_service:variable") or next(
        iter(collection.get("cube:variables", {}).keys()), "data"
    )
    units = renders.get("open_climate_service:units") or (
        collection.get("cube:variables", {}).get(variable, {}).get("unit", "")
    )
    native_crs = collection.get("proj:code", "EPSG:4326")
    epsg = int(native_crs.split(":")[-1]) if ":" in native_crs else 4326

    print(f"Dataset  : {collection.get('title', args.dataset_id)}")
    print(f"Variable : {variable}  |  Units: {units or '—'}  |  CRS: {native_crs}")
    print(f"Colormap : {colormap_name}  |  Range: [{vmin}, {vmax}]")

    # ── 2. Open Zarr store via ClimateService client ─────────────────────────
    service = ClimateService(url)
    try:
        ds = service.open_dataset(args.dataset_id)
    finally:
        service.close()

    da = ds[variable]

    # ── 3. Slice time (all periods by default) ───────────────────────────────
    time_dim = next((d for d in da.dims if d in ("t", "time", "times")), None)
    if time_dim is None:
        print(f"error: no time dimension found in {list(da.dims)}", file=sys.stderr)
        sys.exit(1)

    if args.start or args.end:
        da = da.sel({time_dim: slice(args.start, args.end)})

    n_frames = int(da.sizes[time_dim])
    if n_frames == 0:
        print("error: no frames in the selected time range", file=sys.stderr)
        sys.exit(1)
    print(f"Frames   : {n_frames}  →  {output}")

    if units:
        da.attrs.setdefault("units", units)

    # ── 4. Animate ───────────────────────────────────────────────────────────
    from mapflow import animate

    kwargs: dict = {
        "cmap": _resolve_cmap(colormap_name),
        "vmin": vmin,
        "vmax": vmax,
        "video_width": args.width,
    }
    if args.fps:
        kwargs["fps"] = args.fps

    animate(da=da, path=output, crs=epsg, pad_inches=0.2, **kwargs)
    print(f"Saved    : {output}")


if __name__ == "__main__":
    main()
