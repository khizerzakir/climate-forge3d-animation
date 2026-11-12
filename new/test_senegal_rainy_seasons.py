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


def test_open_climate_metadata_kwargs():
    """STAC metadata can embed storage_options and engine in open_kwargs."""
    import sys
    from types import SimpleNamespace
    from unittest.mock import patch

    asset = SimpleNamespace(
        href="az://cpdata/terraclimate.zarr",
        extra_fields={
            "xarray:open_kwargs": {
                "engine": "zarr",
                "consolidated": True,
                "chunks": {},
                "storage_options": {"account_name": "cpdataeuwest", "token": "old"},
            },
            "xarray:storage_options": {"token": "signed"},
        },
    )
    collection = SimpleNamespace(assets={"zarr-abfs": asset})
    catalog = SimpleNamespace(get_collection=lambda _: collection)
    pc = SimpleNamespace(sign_inplace=lambda obj: obj)
    pystac_client = SimpleNamespace(Client=SimpleNamespace(open=lambda *a, **kw: catalog))

    ds = xr.Dataset({v: (("time", "lat", "lon"), np.ones((1, 1, 1)))
                     for v in ["tmax", "tmin", "ppt"]},
                    coords={"time": [pd.Timestamp("2015-06-15")],
                            "lat": [14.], "lon": [-16.]})
    with patch.dict(sys.modules, {"planetary_computer": pc, "pystac_client": pystac_client}):
        with patch.object(mod.xr, "open_zarr", return_value=ds) as open_mock:
            result = mod.open_climate()
    call_args, call_kwargs = open_mock.call_args
    assert call_args == (asset.href,)
    assert "engine" not in call_kwargs
    assert call_kwargs["storage_options"] == {
        "account_name": "cpdataeuwest", "token": "signed"}
    assert list(result.data_vars) == ["tmax", "tmin", "ppt"]


def test_load_climate_selects_baseline_with_set_years():
    """Regression: Xarray .isin(set) can silently select no time steps."""
    from unittest.mock import patch

    years = list(range(1976, 2006)) + [2015]
    times = pd.to_datetime([f"{y}-{m:02d}-01" for y in years for m in mod.MONTHS])
    n = len(times)
    shape = (n, 1, 1)
    vals = np.full(shape, 25.0, dtype="float32")
    rain = np.full(shape, 100.0, dtype="float32")
    vals[-4:] += 2
    rain[-4:] *= 1.2
    mock_ds = xr.Dataset({
        "tmax": (("time", "lat", "lon"), vals + 5),
        "tmin": (("time", "lat", "lon"), vals - 5),
        "ppt": (("time", "lat", "lon"), rain),
    }, coords={"time": times, "lat": [14.0], "lon": [-16.0]})

    with patch.object(mod, "open_climate", return_value=mock_ds):
        result = mod.load_climate([2015], mod.MONTHS)
    assert result.sizes["year"] == 1
    assert float(result.dt.sel(year=2015).values.item()) == 2.0
    assert np.isclose(float(result.dp.sel(year=2015).values.item()), 20.0)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    test_month_weighting_and_anomaly()
    test_classification_and_mask()
    test_missing_month()
    test_open_climate_metadata_kwargs()
    test_load_climate_selects_baseline_with_set_years()
    with tempfile.TemporaryDirectory() as td:
        test_synthetic_frame(Path(td))
    print("PASS: 6 offline tests (incl. regression and synthetic frame)")


def test_sparse_tick_labels():
    assert mod.sparse_year_ticks(list(range(1976, 2022))) == [1976, 1980, 1990, 2000, 2010, 2020]
    assert mod.sparse_year_ticks([2015]) == [2015]
    assert mod.sparse_year_ticks([2019, 2021]) == [2019, 2021]


def test_recompose_existing(tmp_path, monkeypatch):
    import argparse
    import json
    import geopandas as gpd
    from shapely.geometry import box
    from PIL import Image

    renderdir = tmp_path / "renders"
    framedir = tmp_path / "frames"
    cachedir = tmp_path / "cache"
    for d in (renderdir, framedir, cachedir):
        d.mkdir()
    years = [1976, 1977]
    for y in years:
        Image.fromarray(np.full((24, 36, 4), [100, 170, 220, 255], dtype=np.uint8),
                        "RGBA").save(renderdir / f"map_{y}.png")
    pd.DataFrame({"year": years, "temp_anomaly_C": [-0.3, 0.8],
                  "rain_anomaly_pct": [-10.0, 20.0]}).to_csv(
                      tmp_path / "country_gridcell_means.csv", index=False)
    extent = [250000, 286000, 1300000, 1324000]
    (tmp_path / "run_metadata.json").write_text(json.dumps({
        "extent_m": extent, "renderer": "matplotlib", "months": [6, 7, 8, 9]}))
    poly = gpd.GeoDataFrame(geometry=[box(extent[0], extent[2], extent[1], extent[3])],
                            crs=mod.CRS)
    monkeypatch.setattr(mod, "senegal_polygon", lambda _: poly)
    monkeypatch.setattr(mod, "load_dem", lambda: (_ for _ in ()).throw(
        RuntimeError("No DEM request should be needed")))
    args = argparse.Namespace(start=1976, end=2024, years=None,
                              forge_labels=False, fps=4, no_video=True)
    mod.recompose_existing(args, tmp_path, renderdir, framedir, cachedir)
    assert (framedir / "frame_0000.png").exists()
    assert (framedir / "frame_0001.png").exists()


def test_local_climate_separate_grids(tmp_path):
    """Temperature and rainfall must stay on different native grids."""
    from argparse import Namespace
    yrs = list(range(1976, 2006)) + [2015]
    time = pd.to_datetime([f"{y}-{m:02d}-15" for y in yrs for m in mod.MONTHS])
    t = np.full((len(time), 3, 4), 26., dtype="float32")
    p = np.full((len(time), 4, 5), 90., dtype="float32")
    t[-4:] += 1.5
    p[-4:] *= 1.3
    ds_t = xr.Dataset({"t2m": (("time", "latitude", "longitude"), t + 273.15)},
                      coords={"time": time, "latitude": [13.0, 14.0, 15.0],
                              "longitude": [-17., -16., -15., -14.]})
    ds_p = xr.Dataset({"precip": (("time", "lat", "lon"), p)},
                      coords={"time": time, "lat": [12.5, 13.5, 14.5, 15.5],
                              "lon": [-17.5, -16.5, -15.5, -14.5, -13.5]})
    tpath = tmp_path / "temperature.nc"
    ppath = tmp_path / "rainfall.nc"
    ds_t.to_netcdf(tpath)
    ds_p.to_netcdf(ppath)
    args = Namespace(temp_path=str(tpath), rain_path=str(ppath), climate_path=None,
                     temp_var="t2m", tmax_var="tmax", tmin_var="tmin", ppt_var="precip",
                     temp_units="K", ppt_units="mm/month")
    climate = mod.local_anomalies([2015], mod.MONTHS, args)
    assert climate["dt"].shape == (1, 3, 4)
    assert climate["dp"].shape == (1, 4, 5)
    assert np.isclose(float(climate["dt"].mean()), 1.5, atol=1.e-4)
    assert np.isclose(float(climate["dp"].mean()), 30, atol=1.e-4)


def test_local_climate_combined_dataset(tmp_path):
    """Existing TerraClimate-like NetCDF layout remains supported."""
    from argparse import Namespace
    yrs = list(range(1976, 2006)) + [2015]
    time = pd.to_datetime([f"{y}-{m:02d}-15" for y in yrs for m in mod.MONTHS])
    t = np.ones((len(time), 2, 2), dtype="float32") * 25
    p = np.ones_like(t) * 100
    t[-4:] += 2
    p[-4:] += 10
    ds = xr.Dataset({"maximum": (("time", "lat", "lon"), t + 5),
                     "minimum": (("time", "lat", "lon"), t - 5),
                     "rain": (("time", "lat", "lon"), p)},
                    coords={"time": time, "lat": [13., 15.], "lon": [-17., -15.]})
    path = tmp_path / "climate.nc"
    ds.to_netcdf(path)
    args = Namespace(temp_path=None, rain_path=None, climate_path=str(path),
                     temp_var=None, tmax_var="maximum", tmin_var="minimum", ppt_var="rain",
                     temp_units="C", ppt_units="mm/month")
    out = mod.local_anomalies([2015], mod.MONTHS, args)
    assert np.isclose(float(out["dt"].mean()), 2)
    assert np.isclose(float(out["dp"].mean()), 10)


def test_azure_close_runs_finalizer_once():
    """Targeted adlfs callbacks close on the live loop and don't run at exit."""
    import weakref
    from fsspec.asyn import sync, get_loop
    obj_type = type("FakeAzureBlobFileSystem", (), {"__module__": "adlfs.spec"})
    obj = obj_type()
    called = []

    async def close_service_client(_):
        called.append("closed")

    close_service_client.__module__ = "adlfs.utils"
    fin = weakref.finalize(obj, sync, get_loop(), close_service_client, obj)
    try:
        assert fin.alive
        assert mod.close_azure_finalizers() >= 1
        assert called == ["closed"]
        assert not fin.alive
    finally:
        fin.detach()


def test_main_local_pipeline_without_remote_io(tmp_path, monkeypatch):
    """Exercise main() with two genuine local NetCDF inputs and fake DEM regridding."""
    from argparse import Namespace
    import geopandas as gpd
    from shapely.geometry import box
    from rasterio.transform import from_bounds

    yrs = list(range(1976, 2006)) + [2015]
    dates = pd.to_datetime([f"{y}-{m:02d}-15" for y in yrs for m in mod.MONTHS])
    temperature = np.full((len(dates), 2, 2), 26, dtype="float32")
    rain = np.full((len(dates), 3, 3), 100, dtype="float32")
    temperature[-4:] += 1
    rain[-4:] *= 1.25
    tp = tmp_path / "temp.nc"
    pp = tmp_path / "rain.nc"
    xr.Dataset({"t": (("time", "lat", "lon"), temperature)},
               coords={"time": dates, "lat": [13., 15.], "lon": [-17., -15.]}).to_netcdf(tp)
    xr.Dataset({"p": (("time", "lat", "lon"), rain)},
               coords={"time": dates, "lat": [12.5, 14., 15.5],
                       "lon": [-17.5, -15.5, -13.5]}).to_netcdf(pp)

    class FakeRio:
        def transform(self):
            return from_bounds(250000, 1300000, 300000, 1340000, 50, 40)

        def bounds(self):
            return 250000, 1300000, 300000, 1340000

    class FakeDEM:
        def __init__(self):
            self.values = np.full((40, 50), 100, dtype="float32")
            self.shape = self.values.shape
            self.rio = FakeRio()

    args = Namespace(
        source="local", climate_path=None, temp_path=str(tp), rain_path=str(pp),
        temp_var="t", tmax_var="tmax", tmin_var="tmin", ppt_var="p",
        temp_units="C", ppt_units="mm/month", dem_path="does_not_need_to_exist.tif",
        boundary_path=None, renderer="matplotlib", years=[2015],
        baseline=[1976, 2005], start=1976, end=2024,
        months=list(mod.MONTHS), fps=4, no_video=True,
        recompose_only=False, forge_labels=False, output=tmp_path / "result",
    )
    monkeypatch.setattr(mod, "parse_cli", lambda: args)
    poly = gpd.GeoDataFrame(geometry=[box(250100, 1300100, 299900, 1339900)], crs=mod.CRS)
    monkeypatch.setattr(mod, "senegal_polygon", lambda _: poly)
    monkeypatch.setattr(mod, "load_local_dem", lambda _: FakeDEM())
    monkeypatch.setattr(mod, "to_grid", lambda da, dem:
                        np.full(dem.shape, float(da.mean()), dtype="float32"))
    mod.main()
    assert (args.output / "frames" / "frame_0000.png").is_file()
    stats = pd.read_csv(args.output / "country_gridcell_means.csv")
    assert np.isclose(stats.iloc[0].temp_anomaly_C, 1)
    assert np.isclose(stats.iloc[0].rain_anomaly_pct, 25)
