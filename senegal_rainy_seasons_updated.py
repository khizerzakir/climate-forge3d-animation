#!/usr/bin/env python3
"""Senegal rainy-season climate anomalies: TerraClimate -> aligned grid -> bivariate frames.

Two renderers:
  --renderer forge3d      GPU terrain map using MapScene + BivariateRasterStyle
  --renderer matplotlib   GPU-free, rigorously georeferenced 2-D hillshade map

Default years 1976..2024 (49 seasons); climatology 1976..2005; JJAS.
Raw global data are streamed via Planetary Computer Zarr, not downloaded as files.

Examples:
  python senegal_rainy_seasons_updated.py --years 2015 --renderer matplotlib --no-video
  python senegal_rainy_seasons_updated.py --years 2015 --renderer forge3d --no-video
  python senegal_rainy_seasons_updated.py --start 1976 --end 2024 --renderer forge3d

CAVEAT: forge3d's exact pixel-to-world camera footprint must be validated before
geographic labels/outlines can be safely placed over its output. --forge-labels
opts in to an approximate north-up bounds assumption; leave OFF by default.
See README for implementation, data, and methodological notes.
"""
from __future__ import annotations

import argparse
import calendar
import json
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ----------------------------- Model/configuration -----------------------------
CRS = "EPSG:32628"  # UTM Zone 28N (metres). Matches Senegal's main territory.
BBOX = (-17.9, 12.2, -11.2, 16.8)  # west, south, east, north, lon/lat
BASE = (1976, 2005)
MONTHS = (6, 7, 8, 9)
T_NORMAL = 0.5    # degrees Celsius
P_NORMAL = 25.0   # percent
P_BASE_MIN = 20.0 # exclude pixels with baseline seasonal precip <= 20 mm
RES = 1000        # target display resolution, metres (not TerraClimate native resolution)
Z_EXAGGERATION = 12
W_PX = 1100

# palette[precip class][temperature class], low -> high on both axes
PALETTE = np.array([
    [[140, 180, 225, 255], [85, 85, 85, 255],    [125, 70, 15, 255]],
    [[65, 130, 195, 255],  [125, 125, 125, 255], [170, 110, 55, 255]],
    [[0, 85, 155, 255],    [185, 182, 178, 255], [215, 165, 110, 255]],
], dtype=np.uint8)

CITIES = {
    "Dakar": (-17.47, 14.69), "Saint-Louis": (-16.49, 16.02),
    "Thiès": (-16.93, 14.79), "Touba": (-15.88, 14.85),
    "Louga": (-16.22, 15.62), "Kaolack": (-16.08, 14.15),
    "Tambacounda": (-13.67, 13.77), "Kédougou": (-12.18, 12.56),
    "Kolda": (-14.95, 12.89), "Ziguinchor": (-16.27, 12.58),
}


def seasonal(ds: xr.Dataset, months: tuple[int, ...] = MONTHS) -> xr.Dataset:
    """Produce one seasonal temperature (day-weighted) and rainfall total per year.

    Expects tmax/tmin in degrees C and ppt in mm after CF decoding. Dataset must
    contain ONLY complete seasons, checked separately by validate_seasons().
    """
    ds = ds.sel(time=ds.time.dt.month.isin(months))
    days = ds.time.dt.days_in_month.astype("float32")
    t_mean_month = (ds["tmax"] + ds["tmin"]) / 2.0
    t_weighted = (t_mean_month * days).groupby("time.year").sum("time", skipna=False)
    season_days = days.groupby("time.year").sum("time")
    t = (t_weighted / season_days).rename("temp_C")
    p = ds["ppt"].groupby("time.year").sum("time", skipna=False).rename("rain_mm")
    t.attrs["units"] = "degC"
    p.attrs["units"] = "mm"
    return xr.Dataset({"t": t, "p": p})


def anomalies(seas: xr.Dataset, base: tuple[int, int] = BASE,
              rain_min: float = P_BASE_MIN) -> xr.Dataset:
    """Baseline average of complete seasons; percentage relative to mean seasonal mm.

    Temperature: seasonal temperature - baseline mean, in degrees C.
    Rain: 100 * (seasonal mm - baseline mm) / baseline mm, not normalized by
    the current-year rainfall. Rainfall percent is missing where baseline <= min.
    """
    baseline = seas.sel(year=slice(base[0], base[1]))
    wanted = set(range(base[0], base[1] + 1))
    got = set(int(y) for y in baseline.year.values)
    if not wanted.issubset(got):
        raise ValueError(f"Missing baseline year(s): {sorted(wanted - got)}")
    clim = baseline.mean("year", skipna=False)
    p0 = clim["p"].where(clim["p"] > rain_min)
    dt = (seas["t"] - clim["t"]).rename("temp_anomaly_C")
    dp = ((seas["p"] - clim["p"]) / p0 * 100.0).rename("rain_anomaly_pct")
    dt.attrs["units"] = "degC"
    dp.attrs["units"] = "%"
    return xr.Dataset({"dt": dt, "dp": dp})


def validate_seasons(ds: xr.Dataset, months: tuple[int, ...], needed: set[int]) -> list[int]:
    """Check each year has each desired monthly stamp exactly once."""
    dates = pd.DatetimeIndex(ds.time.values)
    available = []
    for y in sorted(needed):
        counts = [int(((dates.year == y) & (dates.month == m)).sum()) for m in months]
        if all(n == 1 for n in counts):
            available.append(y)
    return available


def bivariate_rgba(temp_grid: np.ndarray, rain_grid: np.ndarray,
                    valid_mask: np.ndarray) -> np.ndarray:
    """Native-grid exact 3x3 palette, with transparent missing/outside cells."""
    if temp_grid.shape != rain_grid.shape or temp_grid.shape != valid_mask.shape:
        raise ValueError("Temperature, rainfall, and Senegal mask must have same shape")
    tx = np.searchsorted([-T_NORMAL, T_NORMAL], temp_grid, side="right")
    py = np.searchsorted([-P_NORMAL, P_NORMAL], rain_grid, side="right")
    rgba = PALETTE[py, tx].copy()
    valid = valid_mask & np.isfinite(temp_grid) & np.isfinite(rain_grid)
    rgba[~valid] = (0, 0, 0, 0)
    return rgba


def bbox_terraclimate(ds: xr.Dataset) -> xr.Dataset:
    """Slice before reading any big variables; supports latitude order and 0..360 lon."""
    renames = {}
    if "longitude" in ds.dims:
        renames["longitude"] = "lon"
    if "latitude" in ds.dims:
        renames["latitude"] = "lat"
    if renames:
        ds = ds.rename(renames)
    if not {"lon", "lat"}.issubset(ds.coords) or ds.lon.ndim != 1 or ds.lat.ndim != 1:
        raise ValueError("Expected TerraClimate with 1-D lon and lat coordinates")
    w, s, e, n = BBOX
    # Convert only the small coordinate index now; avoid global variable shuffles.
    lon_wrapped = ((np.asarray(ds.lon.values) + 180) % 360) - 180
    lat = np.asarray(ds.lat.values)
    ii = np.where((lon_wrapped >= w - 0.1) & (lon_wrapped <= e + 0.1))[0]
    jj = np.where((lat >= s - 0.1) & (lat <= n + 0.1))[0]
    if not len(ii) or not len(jj):
        raise ValueError("No TerraClimate coordinates in Senegal bounding box")
    ds = ds.isel(lon=ii, lat=jj).assign_coords(lon=("lon", lon_wrapped[ii]))
    return ds.sortby("lon").sortby("lat")


def open_climate() -> xr.Dataset:
    """Lazy, spatially and temporally subset Zarr; no raw-file downloads."""
    try:
        import planetary_computer as pc
        import pystac_client
    except ImportError as exc:
        raise RuntimeError("Install: pip install pystac-client planetary-computer adlfs zarr dask") from exc
    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace)
    collection = cat.get_collection("terraclimate")
    asset = collection.assets["zarr-abfs"]
    # The STAC xarray:open_kwargs can contain both engine="zarr" and
    # storage_options. xr.open_zarr sets the engine itself, and storage_options
    # must be supplied exactly once. Planetary Computer collections differ in
    # whether the options live inside open_kwargs or a separate STAC field.
    kwargs = dict(asset.extra_fields.get("xarray:open_kwargs", {}) or {})
    kwargs.pop("engine", None)
    embedded_options = kwargs.pop("storage_options", {}) or {}
    separate_options = asset.extra_fields.get("xarray:storage_options", {}) or {}
    storage_options = {**embedded_options, **separate_options}
    kwargs.setdefault("consolidated", True)
    try:
        ds = xr.open_zarr(asset.href, storage_options=storage_options, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open Planetary Computer TerraClimate Zarr: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    ds = ds[["tmax", "tmin", "ppt"]]
    return bbox_terraclimate(ds)


def load_climate(years: list[int], months: tuple[int, ...]) -> xr.Dataset:
    """Read *only* baseline + target months, then aggregate before computing."""
    needed = set(range(BASE[0], BASE[1] + 1)) | set(years)
    ds = open_climate()
    ds = ds.sel(
    time=ds.time.dt.year.isin(sorted(needed))
    & ds.time.dt.month.isin(list(months))
    )
    valid = set(validate_seasons(ds, months, needed))
    missing_base = set(range(BASE[0], BASE[1] + 1)) - valid
    if missing_base:
        raise RuntimeError(f"Baseline incomplete or unavailable: {sorted(missing_base)}")
    available_targets = [y for y in years if y in valid]
    missing_targets = sorted(set(years) - valid)
    if missing_targets:
        warnings.warn(f"Skipping missing/incomplete target season(s): {missing_targets}")
    if not available_targets:
        raise RuntimeError("No requested target seasons are available")
    use = set(range(BASE[0], BASE[1] + 1)) | set(available_targets)
    ds = ds.sel(time=ds.time.dt.year.isin(sorted(use)))
    print(f"Calculating seasonal climatology from {BASE[0]}-{BASE[1]} and "
          f"{len(available_targets)} target season(s) (no raw file download).")
    # Compute the reduced seasonal grid only, not the full monthly source cube.
    seas = seasonal(ds, months).compute()
    for var, lo, hi in (("t", -30, 65), ("p", 0, 10000)):
        med = float(seas[var].median(skipna=True))
        if not np.isfinite(med) or not lo <= med <= hi:
            raise RuntimeError(f"Unexpected seasonal {var}={med:g}; check TerraClimate "
                               "scale_factor/units in the opened store")
    return anomalies(seas).sel(year=available_targets)


def senegal_polygon(cache_dir: Path):
    import geopandas as gpd
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "senegal_adm0.geojson"
    if not path.exists():
        import requests
        url = "https://www.geoboundaries.org/api/current/gbOpen/SEN/ADM0/"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        meta = response.json()
        geo = requests.get(meta["gjDownloadURL"], timeout=90)
        geo.raise_for_status()
        path.write_bytes(geo.content)
    return gpd.read_file(path).to_crs(CRS)


def load_dem() -> xr.DataArray:
    """Copernicus DEM: mosaic STAC scenes by first valid pixel (not by max height)."""
    try:
        import odc.stac
        import planetary_computer as pc
        import pystac_client
        import rioxarray  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install: pip install odc-stac pystac-client "
                           "planetary-computer rioxarray rasterio") from exc
    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace)
    items = cat.search(collections=["cop-dem-glo-90"], bbox=BBOX).item_collection()
    if not items:
        raise RuntimeError("No Copernicus DEM items for Senegal BBOX")
    mosaic = odc.stac.load(items, bands=["data"], crs=CRS, resolution=RES,
                           bbox=BBOX, resampling="bilinear", groupby="id",
                           chunks={"x": 512, "y": 512})["data"]
    # The imagery may be split across many STAC items/times. First valid wins,
    # unlike max(time), which can introduce artificial elevation ridges.
    dem = mosaic.isel(time=0, drop=True)
    for index in range(1, mosaic.sizes["time"]):
        dem = dem.combine_first(mosaic.isel(time=index, drop=True))
    dem = dem.astype("float32").compute()
    if not bool(np.isfinite(dem).any()):
        raise RuntimeError("DEM mosaic has no finite cells")
    return dem.rio.write_crs(CRS)


def to_grid(da: xr.DataArray, dem: xr.DataArray) -> np.ndarray:
    import rioxarray  # noqa: F401
    from rasterio.enums import Resampling
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs("EPSG:4326")
    warped = da.rio.reproject_match(dem, resampling=Resampling.bilinear)
    return np.asarray(warped.values, dtype="float32")


def render_forge3d(year: int, elevation: np.ndarray, mask: np.ndarray,
                   tg: np.ndarray, pg: np.ndarray, output: Path) -> None:
    try:
        import forge3d as f3d
    except ImportError as exc:
        raise RuntimeError("forge3d is not installed; pip install 'forge3d>=1.42' "
                           "or use --renderer matplotlib") from exc
    t = np.where(mask, tg, np.nan).astype("float32")
    p = np.where(mask, pg, np.nan).astype("float32")
    style = f3d.BivariateRasterStyle(
        [-T_NORMAL, T_NORMAL], [-P_NORMAL, P_NORMAL],
        tuple(tuple(tuple(int(v) for v in c) for c in row) for row in PALETTE),
        "Temperature anomaly", "Rainfall anomaly", x_units="°C", y_units="%")
    h, w = elevation.shape
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=elevation, style=f3d.RasterHeightSurfaceStyle(Z_EXAGGERATION, "m"),
            crs=CRS, metadata={"source_id": "cop-dem-glo-90", "width": w,
                               "height": h, "resolution": [RES, RES]},
            elevation_sampling_available=True),
        layers=(f3d.RasterOverlay(layer_id=f"anomalies-{year}", data=t,
                                  secondary_data=p, style=style, crs=CRS,
                                  metadata={"source_id": f"terraclimate-{year}"}),),
        lighting=f3d.LightingPreset(name="daylight",
                                    sun_direction=(0.8, 0.6, 0.0),
                                    settings={"hue_variation_strength": 0}),
        output=f3d.OutputSpec(width=W_PX, height=max(2, round(W_PX * h / w))),
    )
    report = scene.render(str(output))
    print(f"forge3d {year}: {getattr(report, 'status', 'rendered')}")


def render_matplotlib_map(elevation: np.ndarray, mask: np.ndarray,
                          tg: np.ndarray, pg: np.ndarray, output: Path) -> None:
    """Deterministic, north-up, georeferenced palette with hillshade modulation."""
    from matplotlib.colors import LightSource
    from PIL import Image
    rgba = bivariate_rgba(tg, pg, mask)
    shade = LightSource(azdeg=315, altdeg=42).hillshade(
        elevation, vert_exag=Z_EXAGGERATION, dx=RES, dy=RES)
    rgb = rgba[..., :3].astype("float32")
    factor = 0.72 + 0.34 * shade[..., None]
    rgba[..., :3] = np.clip(rgb * factor, 0, 255).astype("uint8")
    Image.fromarray(rgba, "RGBA").save(output)


def _mean(arr: np.ndarray, mask: np.ndarray) -> float:
    z = arr[mask & np.isfinite(arr)]
    return float(z.mean()) if z.size else float("nan")


def compose(year: int, map_png: Path, years: list[int],
            temp_series: list[float], rain_series: list[float],
            extent: tuple[float, float, float, float], poly,
            renderer: str, label_on_forge: bool, output: Path,
            months: tuple[int, ...] = MONTHS) -> None:
    """Use matplotlib for the plate/legend/timeline over the rendered raster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.patches import Rectangle
    from PIL import Image
    from pyproj import Transformer

    idx = years.index(year)
    img = np.asarray(Image.open(map_png).convert("RGBA"))
    fig = plt.figure(figsize=(9, 11.2), dpi=145, facecolor="#f7f7f4")
    fig.text(0.50, 0.977, f"{len(years)} Senegalese Rainy Season{'s' if len(years) != 1 else ''}",
             fontsize=21, weight="bold", ha="center", va="top", color="#20242a")
    month_label = "–".join([calendar.month_name[months[0]], calendar.month_name[months[-1]]])
    fig.text(0.50, 0.939, f"{month_label}: temperature × rainfall anomalies vs. "
             f"{BASE[0]}–{BASE[1]}", fontsize=10.5, ha="center", color="#49505a")
    fig.text(0.065, 0.888, str(year), fontsize=31, weight="bold", color="#222222")
    ts, ps = temp_series[idx], rain_series[idx]
    summary = f"Senegal grid-cell average: {ts:+.2f} °C  ·  {ps:+.1f} % rainfall"
    fig.text(0.95, 0.898, summary, fontsize=10, ha="right", color="#323840")
    ax = fig.add_axes([0.035, 0.310, 0.93, 0.555], facecolor="#ecf1f3")
    ax.imshow(img, origin="upper", extent=extent, interpolation="nearest", zorder=1)
    ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:]); ax.set_aspect("equal")
    # On a true projected Matplotlib map, overlays are metrically registered.
    # For a forge3d scene, the output camera may not cover the exact grid extent.
    show_geo = renderer == "matplotlib" or label_on_forge
    if show_geo:
        poly.boundary.plot(ax=ax, color="#404d53", linewidth=0.65, zorder=3)
        xy = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
        for name, (lon, lat) in CITIES.items():
            x, y = xy.transform(lon, lat)
            if extent[0] < x < extent[1] and extent[2] < y < extent[3]:
                ax.plot(x, y, "o", ms=3.4, color="white", mec="#242b31", mew=0.7, zorder=4)
                ax.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points",
                            fontsize=7.1, color="white", zorder=5,
                            path_effects=[pe.withStroke(linewidth=2, foreground="#212833")])
    else:
        fig.text(0.50, 0.314, "forge3d terrain: map labels withheld until camera "
                 "georegistration is verified", ha="center", fontsize=7, color="#6b6570")
    ax.set_axis_off()

    lg = fig.add_axes([0.078, 0.086, 0.165, 0.18])
    for r in range(3):
        for c in range(3):
            lg.add_patch(Rectangle((c, r), 1, 1,
                                   facecolor=PALETTE[r, c, :3] / 255,
                                   edgecolor="white", linewidth=0.9))
    lg.set(xlim=(0, 3), ylim=(0, 3)); lg.set_aspect("equal"); lg.axis("off")
    fig.text(0.087, 0.270, "Bivariate climate anomalies", fontsize=8.2,
             weight="bold", color="#272b30")
    fig.text(0.080, 0.075, "← Cooler        Warmer →", fontsize=7.7)
    fig.text(0.061, 0.171, "Drier     Wetter", fontsize=7.4, rotation=90,
             ha="center", va="center")
    fig.text(0.080, 0.053, f"Normal: ±{T_NORMAL:g}°C, ±{P_NORMAL:g}%", fontsize=7)
    fig.text(0.080, 0.035, f"Rain baseline > {P_BASE_MIN:g} mm", fontsize=6.8,
             color="#5d6670")

    for box, values, ttl, neg, pos in (
        ([0.380, 0.168, 0.57, 0.103], temp_series,
         "Mean temperature anomaly (°C)", "#3b79ae", "#ce7945"),
        ([0.380, 0.045, 0.57, 0.103], rain_series,
         "Mean rainfall anomaly (%)", "#a87b43", "#358f9a"),
    ):
        p = fig.add_axes(box, facecolor="none")
        colors = [neg if v < 0 else pos for v in values]
        p.bar(years, values, color=colors, width=0.78, zorder=2)
        p.axhline(0, color="#6d7177", lw=0.6, zorder=1)
        p.axvline(year, color="#212121", lw=1.6, zorder=3)
        p.set_xlim(min(years)-1, max(years)+1)
        p.set_title(ttl, loc="left", fontsize=8.5, weight="bold", pad=4)
        p.spines[["top", "right"]].set_visible(False)
        p.tick_params(axis="both", labelsize=7)
        p.set_xticks(sorted(set([years[0], years[-1]] + [y for y in years if y % 10 == 0])))
    fig.text(0.95, 0.011, "TerraClimate · Copernicus DEM GLO-90 · geoBoundaries  |  "
             f"baseline {BASE[0]}–{BASE[1]}", fontsize=6.6,
             ha="right", color="#565e62")
    fig.savefig(output, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_video(frames: Path, out: Path, fps: int) -> None:
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames / "frame_%04d.png"),
           "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out)]
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found. To make the video run:\n  " + " ".join(cmd))
        return
    subprocess.run(cmd, check=True)
    print(f"Video saved: {out}")


def parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--renderer", choices=["forge3d", "matplotlib"], default="forge3d")
    ap.add_argument("--years", type=int, nargs="+", help="Specific target years e.g. --years 2015 2020")
    ap.add_argument("--start", type=int, default=1976)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--months", type=int, nargs="+", default=list(MONTHS),
                    help="Months of each calendar-year season (default 6 7 8 9)")
    ap.add_argument("--output", type=Path, default=Path("senegal_out"))
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--forge-labels", action="store_true",
                    help="EXPERIMENTAL: overlay projected cities/ADM0 on forge3d image")
    return ap.parse_args()


def main() -> None:
    import rasterio.features
    args = parse_cli()
    months = tuple(sorted(set(args.months)))
    if not months or min(months) < 1 or max(months) > 12:
        raise ValueError("--months must be in 1..12")
    if args.start > args.end or args.fps < 1:
        raise ValueError("Invalid start/end or fps")
    years = sorted(set(args.years if args.years else range(args.start, args.end+1)))
    out = args.output
    rawdir, frames, cache = out / "renders", out / "frames", out / "cache"
    for d in (rawdir, frames, cache):
        d.mkdir(parents=True, exist_ok=True)
    # Prevent a shorter rerun from including stale frames in the encoded video.
    for old_frame in frames.glob("frame_*.png"):
        old_frame.unlink()
    print("Loading Senegal polygon...")
    poly = senegal_polygon(cache)
    print("Streaming TerraClimate subset and calculating seasonal anomalies...")
    anoms = load_climate(years, months)
    years = [int(y) for y in anoms.year.values]
    print("Preparing Copernicus DEM display grid...")
    dem = load_dem()
    inside = rasterio.features.rasterize(
        [(geom, 1) for geom in poly.geometry if geom is not None],
        out_shape=dem.shape, transform=dem.rio.transform(),
        fill=0, dtype="uint8").astype(bool)
    terrain = np.nan_to_num(dem.values, nan=0.0, posinf=0.0, neginf=0.0)
    terrain = np.clip(terrain, 0, None).astype("float32")
    left, bottom, right, top = dem.rio.bounds()
    extent = (left, right, bottom, top)
    temp_grids, rain_grids, temp_means, rain_means = [], [], [], []
    for y in years:
        tg = to_grid(anoms.dt.sel(year=y), dem)
        pg = to_grid(anoms.dp.sel(year=y), dem)
        temp_grids.append(tg); rain_grids.append(pg)
        temp_means.append(_mean(tg, inside)); rain_means.append(_mean(pg, inside))
    stats = pd.DataFrame({"year": years, "temp_anomaly_C": temp_means,
                          "rain_anomaly_pct": rain_means})
    stats.to_csv(out / "country_gridcell_means.csv", index=False)
    (out / "run_metadata.json").write_text(json.dumps({
        "baseline_years": list(BASE), "target_years": years, "months": list(months),
        "precipitation_baseline_min_mm": P_BASE_MIN,
        "temp_normal_C": T_NORMAL, "rain_normal_pct": P_NORMAL,
        "target_crs": CRS, "target_resolution_m": RES,
        "renderer": args.renderer,
        "caveats": ["rainfall percent is mean of valid pixel percentages, not percent of summed totals",
                    "projected 1-km grid is display resampling of ~4-km TerraClimate",
                    "forge3d camera extent is not georegistered in plate unless verified"],
    }, indent=2), encoding="utf-8")
    for i, y in enumerate(years):
        raw_file = rawdir / f"map_{y}.png"
        if args.renderer == "forge3d":
            render_forge3d(y, terrain, inside, temp_grids[i], rain_grids[i], raw_file)
        else:
            render_matplotlib_map(terrain, inside, temp_grids[i], rain_grids[i], raw_file)
        compose(y, raw_file, years, temp_means, rain_means, extent, poly,
                args.renderer, args.forge_labels, frames / f"frame_{i:04d}.png", months)
        print(f"Frame {i+1}/{len(years)}: {y}")
    if not args.no_video:
        make_video(frames, out / "senegal_rainy_seasons.mp4", args.fps)
    print(f"Finished. Outputs: {out.resolve()}")


if __name__ == "__main__":
    main()
