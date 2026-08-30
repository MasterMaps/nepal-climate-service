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
    uv run --group scripts scripts/animate.py chirps3_precipitation_daily --no-title --no-legend
"""

from __future__ import annotations

import argparse
import sys

BASE_URL = "http://localhost:8011"
DEFAULT_WIDTH = 1200


class _SyncPool:
    """Drop-in replacement for multiprocessing.Pool that runs imap in the calling
    process.  Used when the colorbar is hidden: mapflow spawns worker processes
    (macOS uses 'spawn', not 'fork'), so a plt.colorbar monkey-patch in the main
    process is invisible to workers.  By replacing Pool we keep all rendering in
    the main process where the patch is active."""

    def __init__(self, **_): ...

    def __enter__(self):
        return self

    def __exit__(self, *_): ...

    def imap(self, func, iterable):
        return (func(item) for item in iterable)


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


def _time_format(period_type: str | None) -> str:
    if period_type == "yearly":
        return "%Y"
    if period_type == "monthly":
        return "%Y-%m"
    if period_type == "hourly":
        return "%Y-%m-%dT%H:00"
    return "%Y-%m-%d"


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
    parser.add_argument(
        "--width", type=int, default=DEFAULT_WIDTH,
        help=f"Output video width in pixels (default: {DEFAULT_WIDTH})",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="Scale factor applied to --width (default: 1.0, e.g. 2.0 for 2× resolution)",
    )
    parser.add_argument("--cmap", help="Override colormap (matplotlib name, e.g. RdBu_r)")
    parser.add_argument("--vmin", type=float, help="Override color range minimum")
    parser.add_argument("--vmax", type=float, help="Override color range maximum")

    title_group = parser.add_mutually_exclusive_group()
    title_group.add_argument(
        "--title", dest="title", action="store_true", default=True,
        help="Show dataset name and timestamp as title (default: on)",
    )
    title_group.add_argument(
        "--no-title", dest="title", action="store_false",
        help="Hide title",
    )

    legend_group = parser.add_mutually_exclusive_group()
    legend_group.add_argument(
        "--legend", dest="legend", action="store_true", default=True,
        help="Show colorbar legend (default: on)",
    )
    legend_group.add_argument(
        "--no-legend", dest="legend", action="store_false",
        help="Hide colorbar legend",
    )

    args = parser.parse_args()

    output = args.output or f"{args.dataset_id}.mp4"
    video_width = int(args.width * args.scale)
    # No padding when both decorations are off → map fills the entire frame
    pad_inches = 0.0 if not args.title and not args.legend else 0.2

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
    period_type = next(
        (v.get("step") for v in collection.get("cube:dimensions", {}).values() if v.get("type") == "temporal"),
        None,
    )

    print(f"Dataset  : {collection.get('title', args.dataset_id)}")
    print(f"Variable : {variable}  |  Units: {units or '—'}  |  CRS: {native_crs}")
    print(f"Colormap : {colormap_name}  |  Range: [{vmin}, {vmax}]")
    print(f"Output   : {video_width}px  |  pad: {pad_inches}  |  title: {args.title}  |  legend: {args.legend}")

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

    # ── 4. Build title list and label ────────────────────────────────────────
    titles: list[str] | None = None
    if args.title:
        import pandas as pd
        fmt = _time_format(period_type)
        try:
            time_strs = pd.DatetimeIndex(da[time_dim].values).strftime(fmt).tolist()
        except Exception:
            time_strs = [str(t) for t in da[time_dim].values]
        dataset_title = collection.get("title", args.dataset_id)
        titles = [f"{dataset_title} – {t}" for t in time_strs]

    label: str | None = units if (args.legend and units) else None

    # ── 5. Animate via Animation directly (gives full control over title/label)
    import mapflow._classic as _mc
    import matplotlib.pyplot as plt
    from mapflow import Animation

    x_dim = next((d for d in da.dims if d in ("x", "lon", "longitude")), None)
    y_dim = next((d for d in da.dims if d in ("y", "lat", "latitude")), None)
    if x_dim is None or y_dim is None:
        non_time = [d for d in da.dims if d != time_dim]
        y_dim, x_dim = non_time[0], non_time[1]

    # Sort to ascending x/y order — same as mapflow's check_da; required for
    # origin="lower" imshow to render south-at-bottom / west-at-left correctly.
    da = da.sortby(x_dim).sortby(y_dim)
    da = da.transpose(time_dim, y_dim, x_dim)
    data = da.load().values

    anim = Animation(x=da[x_dim].values, y=da[y_dim].values, crs=epsg)

    kwargs: dict = {
        "cmap": _resolve_cmap(colormap_name),
        "vmin": vmin,
        "vmax": vmax,
        "video_width": video_width,
    }
    if args.fps:
        kwargs["fps"] = args.fps

    # Hiding the legend requires two patches:
    # 1. plt.colorbar → no-op so no colorbar is drawn.
    # 2. mapflow._classic.Pool → _SyncPool so frames render in the main process.
    #    mapflow uses multiprocessing.Pool; on macOS workers are spawned fresh and
    #    don't inherit the plt.colorbar patch from the parent process.
    _orig_colorbar = plt.colorbar
    _orig_pool = _mc.Pool
    if not args.legend:
        plt.colorbar = lambda **_: None
        _mc.Pool = _SyncPool
    try:
        anim(data=data, path=output, title=titles, label=label, pad_inches=pad_inches, **kwargs)
    finally:
        plt.colorbar = _orig_colorbar
        _mc.Pool = _orig_pool
    print(f"Saved    : {output}")


if __name__ == "__main__":
    main()
