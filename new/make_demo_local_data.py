#!/usr/bin/env python3
"""Build two small SYNTHETIC native-grid NetCDF datasets + a GeoTIFF DEM.

This demonstrates file formats only; NOT real Senegal climate observations.
Use the actual Senegal GeoJSON already cached by the main script for the map.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.transform import from_bounds

out = Path("local_demo")
out.mkdir(exist_ok=True)
years = list(range(1976, 2006)) + [2015]
times = pd.to_datetime([f"{y}-{m:02d}-15" for y in years for m in (6, 7, 8, 9)])
nt = len(times)

# Temperature in Kelvin on a 0.4-degree grid
lon_t = np.linspace(-17.7, -11.5, 16)
lat_t = np.linspace(12.3, 16.6, 12)
y_t, x_t = np.meshgrid(lat_t, lon_t, indexing="ij")
temp = np.repeat((299.0 + 2.0 * np.cos(x_t / 2.0))[None, ...], nt, axis=0).astype("float32")
temp[-4:] += 1.6
xr.Dataset({"t2m": (("time", "latitude", "longitude"), temp)},
           coords={"time": times, "latitude": lat_t, "longitude": lon_t}).to_netcdf(
               out / "temperature.nc")

# Rainfall in mm per month on a different 0.25-degree grid
lon_r = np.linspace(-17.7, -11.5, 24)
lat_r = np.linspace(12.3, 16.6, 18)
y_r, x_r = np.meshgrid(lat_r, lon_r, indexing="ij")
rain = np.repeat((100 + 30 * np.cos((y_r - 12.3) / 4))[None, ...], nt,
                 axis=0).astype("float32")
rain[-4:] *= 1.2
xr.Dataset({"rain": (("time", "lat", "lon"), rain)},
           coords={"time": times, "lat": lat_r, "lon": lon_r}).to_netcdf(
               out / "rainfall.nc")

# Artificial demonstration terrain, NOT a physical DEM.
h, w = 120, 180
z = np.tile(np.linspace(1, 300, w, dtype="float32"), (h, 1))
profile = {"driver": "GTiff", "dtype": "float32", "count": 1, "width": w,
           "height": h, "crs": "EPSG:4326",
           "transform": from_bounds(-18.0, 12.0, -11.0, 17.0, w, h)}
with rasterio.open(out / "demo_dem.tif", "w", **profile) as dst:
    dst.write(z, 1)
print("Created synthetic inputs under", out.resolve())
print("Use --boundary-path senegal_out/cache/senegal_adm0.geojson if present.")
