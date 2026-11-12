# Senegal rainy seasons: Planetary Computer or your own climate cube

This is an incremental update of your working `senegal_rainy_seasons_updated.py`. Existing `--renderer forge3d`, `--renderer matplotlib` and `--recompose-only` are retained. No scientific data are provided in the package.

## 1. Install / replace

Preserve `senegal_out/` containing your 46 rendered seasons. In your existing virtual environment:

```bash
python -m pip install -r requirements_senegal.txt
python -m pytest -q test_senegal_rainy_seasons.py
```

You do not need to reinstall forge3d if your current forge3d run succeeded.

## 2. Working TerraClimate workflow (1976–2021)

```bash
python senegal_rainy_seasons_updated.py --start 1976 --end 2021 --renderer forge3d --fps 4
```

The script now proactively invokes **only adlfs-specific weakref finalizers** before interpreter shutdown, rather than after fsspec's event loop has closed. It prints the number of cleaned-up Azure resources. This is a targeted, CPython-specific workaround using the private `weakref.finalize._registry` implementation; future adlfs/fsspec versions might remove the need or change the behavior. If the warning persists, share `pip show adlfs fsspec` and the new output. Remote access was not tested here.

If you only want to recombine existing image frames without opening TerraClimate again:

```bash
python senegal_rainy_seasons_updated.py --recompose-only --fps 4
```

## 3. Own climate dataset, one NetCDF or local Zarr

Input must be **monthly** with one timestamp per month, `time` and 1-D `lat/lon` (or `latitude/longitude`) geographic coordinates in EPSG:4326. It can use ascending or descending latitude and −180..180 or 0..360 longitude. Variables need to be CF-decoded and physically meaningful.

For a single NetCDF file with temperature maxima/minima in Celsius and precipitation in mm/month:

```bash
python senegal_rainy_seasons_updated.py \
    --source local --climate-path /data/my_climate_cube.nc \
    --tmax-var tmax --tmin-var tmin --ppt-var precip \
    --temp-units C --ppt-units mm/month \
    --start 1981 --end 2024 --baseline 1981 2010 \
    --renderer forge3d --fps 4 --output senegal_out_local
```

For a local Zarr cube, use `--climate-path /data/my_cube.zarr` with the same flags.

## 4. Own temperature and rainfall on different native grids

```bash
python senegal_rainy_seasons_updated.py \
    --source local \
    --temp-path '/data/era5_t2m/monthly_*.nc' --temp-var t2m --temp-units K \
    --rain-path '/data/chirps/monthly_*.nc' --ppt-var precip --ppt-units mm/month \
    --baseline 1981 2010 --start 1981 --end 2024 \
    --renderer forge3d --fps 4 --output senegal_out_era5_chirps
```

Temperature and rainfall are aggregated / compared to their own baselines **on their own native lat/lon grids**. There is no early joining or upsampling. Only after anomaly computation does `to_grid()` bilinearly reproject both anomalies to the common 1-km DEM **display grid**.

Unit options: `--temp-units C|K`, `--ppt-units mm/month|m/month|mm/day|m/day|kg/m2/s`. A rate conversion relies on each month's number of days. **Read source metadata first:** some ERA5 monthly products represent means of daily accumulations and must not be interpreted blindly as monthly totals. If units in your NetCDF are already CF-decoded/scaled, do not reapply scale factors.

## 5. Use your own DEM and/or Senegal boundary

```bash
python senegal_rainy_seasons_updated.py \
    --source local --climate-path /data/my_cube.zarr \
    --temp-var t2m --ppt-var precip --temp-units K \
    --baseline 1981 2010 --start 1981 --end 2024 \
    --dem-path /data/copernicus_senegal_dem.tif \
    --boundary-path /data/senegal_adm0.geojson \
    --renderer matplotlib --output senegal_out_local --fps 4
```

DEM is a *single-band*, **georeferenced** GeoTIFF; its CRS must be present. The boundary should be a GeoJSON/Shapefile that can be read by GeoPandas. The area of interest is still Senegal with the fixed source-code `BBOX`, and the map output CRS is EPSG:32628. To use another AOI, update `BBOX`, `CRS`, title, and city labels as well; do not simply substitute a different country polygon.

## 6. Fully local synthetic demonstration

```bash
python make_demo_local_data.py
python senegal_rainy_seasons_updated.py \
    --source local \
    --temp-path local_demo/temperature.nc --temp-var t2m --temp-units K \
    --rain-path local_demo/rainfall.nc --ppt-var rain --ppt-units mm/month \
    --dem-path local_demo/demo_dem.tif \
    --boundary-path senegal_out/cache/senegal_adm0.geojson \
    --years 2015 --renderer matplotlib --no-video \
    --output local_demo/output
```

**Demonstration data are artificial**; temperatures, rainfall and DEM values are NOT observational. Use your previously cached `senegal_adm0.geojson` for a completely offline test. If the cache is absent, omit `--boundary-path`; the script will fetch the boundary online.

## 7. Methodology / caveats

- Baseline is required **for all its years and all selected months**, including when you only render 2015. Missing target years are skipped with a warning. Change to `--baseline 1981 2010` if source coverage begins in 1981.
- Temperature: day-weighted average of monthly temperatures; anomaly in °C. Rain: sum of monthly totals; anomaly `100 * (yearly season − baseline mean season) / baseline mean season` in %, masking baseline precipitation ≤20 mm.
- Temperature/rainfall statistics in the chart are the mean over valid Senegal **display-grid cells** of per-cell anomalies. They are not inherently population weighted, area weighted, or a national rainfall-volume anomaly.
- All output grids are **1 km visualization grids**; reprojecting coarser native grids does not increase their physical resolution.
- Forge3d geographic camera footprint has not been independently calibrated. Country outlines and city labels are omitted on forge3d by default; `--forge-labels` opts in to an *approximate* overlay only.
- Local input must be monthly, 1-D geographic lat/lon and standard datetime; curvilinear grids and native projected climate rasters are not supported by this local loader. Preprocess those with Xarray/rioxarray before passing in.
- The `rasterio.warp.NotGeoreferencedWarning` seen with numpy-based reprojection is distinct from the Azure shutdown warning. The displayed map still needs visual alignment/extent checks, especially when switching renderer.

## 8. Tests

```bash
python -m pytest -q test_senegal_rainy_seasons.py
```

Offline tests cover anomaly maths, baseline selection, bivariate classes, single and separate-grid local NetCDF, a generated frame, recomposition, and targeted Azure finalizer execution with a simulated Azure resource. No Planetary Computer network or forge3d GPU render has been tested in this environment.
