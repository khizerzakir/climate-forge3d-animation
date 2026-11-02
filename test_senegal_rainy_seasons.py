"""Offline numerical and raster/plate smoke tests for senegal_rainy_seasons_updated.py."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SCRIPT = Path(__file__).with_name("senegal_rainy_seasons_updated.py")
spec = importlib.util.spec_from_file_location("senegal", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_month_weighting_and_anomaly():
    years = list(range(1976, 2006)) + [2015]
    times = pd.to_datetime([f"{y}-{m:02d}-15" for y in years for m in [6, 7, 8, 9]])
    shape = (len(times), 2, 2)
    temp = np.full(shape, 20., dtype="float32")
    ppt = np.full(shape, 100., dtype="float32")
    # for 2015, monthly temp 2C higher, rainfall 20% higher
    temp[-4:] = 22.
    ppt[-4:] = 120.
    coords = {"time": times, "lat": [14., 15.], "lon": [-17., -16.]}
    ds = xr.Dataset({"tmax": (("time", "lat", "lon"), temp + 5),
                     "tmin": (("time", "lat", "lon"), temp - 5),
                     "ppt": (("time", "lat", "lon"), ppt)}, coords=coords)
    assert len(mod.validate_seasons(ds, mod.MONTHS, set(years))) == len(years)
    ss = mod.seasonal(ds)
    aa = mod.anomalies(ss)
    assert np.isclose(float(aa.dt.sel(year=2015).mean()), 2.)
    assert np.isclose(float(aa.dp.sel(year=2015).mean()), 20.)
    assert np.isclose(float(aa.dt.sel(year=2000).mean()), 0.)
    assert np.isclose(float(ss.p.sel(year=2015).mean()), 480.)
    # A month-weighting check: June 30d, July 31d, August 31d, Sept 30d
    ds2 = ds.sel(time=ds.time.dt.year == 2015).copy()
    ds2["tmax"][-4, ...] = 32.
    ds2["tmin"][-4, ...] = 22.  # June mean 27; other months mean 22
    expected = (27*30 + 22*(31+31+30)) / (30+31+31+30)
    assert np.isclose(float(mod.seasonal(ds2).t.mean()), expected)


def test_classification_and_mask():
    t = np.array([[-1, 0, 1], [-1, 0, 1]], dtype="float32")
    p = np.array([[-30, -30, -30], [30, 30, 30]], dtype="float32")
    mask = np.ones((2, 3), bool)
    mask[1, 2] = False
    rgba = mod.bivariate_rgba(t, p, mask)
    assert np.array_equal(rgba[0, 0], mod.PALETTE[0, 0])
    assert np.array_equal(rgba[0, 2], mod.PALETTE[0, 2])
    assert np.array_equal(rgba[1, 0], mod.PALETTE[2, 0])
    assert rgba[1, 2, -1] == 0


def test_missing_month():
    dates = pd.to_datetime(["2015-06-15", "2015-07-15", "2015-09-15"])
    ds = xr.Dataset(coords={"time": dates})
    assert mod.validate_seasons(ds, mod.MONTHS, {2015}) == []


def test_synthetic_frame(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box
    h, w = 75, 90
    terrain = np.tile(np.linspace(0, 400, w, dtype="float32"), (h, 1))
    tg = np.broadcast_to(np.linspace(-1.5, 1.5, w, dtype="float32"), (h, w)).copy()
    pg = np.broadcast_to(np.linspace(-50, 50, h, dtype="float32")[:, None], (h, w)).copy()
    mask = np.ones((h, w), dtype=bool)
    map_file = tmp_path / "map.png"
    frame_file = tmp_path / "frame.png"
    mod.render_matplotlib_map(terrain, mask, tg, pg, map_file)
    extent = (250000, 340000, 1300000, 1375000)
    poly = gpd.GeoDataFrame(geometry=[box(extent[0], extent[2], extent[1], extent[3])], crs=mod.CRS)
    mod.compose(2015, map_file, [2015], [1.0], [20.0], extent,
                poly, "matplotlib", False, frame_file)
    assert map_file.exists() and frame_file.exists()


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    test_month_weighting_and_anomaly()
    test_classification_and_mask()
    test_missing_month()
    with tempfile.TemporaryDirectory() as td:
        test_synthetic_frame(Path(td))
    print("PASS: 4 offline tests (incl. synthetic frame)")
