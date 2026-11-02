# Senegal rainy seasons — updated climate-anomaly animation

## Contents

- `senegal_rainy_seasons_updated.py`: full pipeline (on-the-fly TerraClimate, COP-DEM, Senegal ADM0, climate statistics, rendered frames, optional MP4).
- `requirements_senegal.txt`: Python dependencies. `forge3d` remains optional if using Matplotlib.
- `test_senegal_rainy_seasons.py`: numerical, classification, missing-month and synthetic rendering smoke tests (do not require data downloads or GPU).

## Install

Python 3.11+ recommended, in a fresh environment:

```bash
python3.12 -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_senegal.txt
```

If Zarr reports a missing `numcodecs` module, install the explicitly required
codec dependency in the active environment:

```bash
python -m pip install --upgrade "zarr>=3" "numcodecs>=0.16"
```

For the **forge3d renderer** also run:

```bash
pip install "forge3d>=1.42"
```

You need a compatible physical GPU (and appropriate graphics drivers) for forge3d's native terrain renderer. Install `ffmpeg` separately if you want the MP4 (or use `--no-video`).

## Run

**Test numerical logic without Internet/GPU**:

```bash
python test_senegal_rainy_seasons.py
```

**Preview a single year on a normal computer** (no forge3d GPU needed):

```bash
python senegal_rainy_seasons_updated.py --years 2015 --renderer matplotlib --no-video
```

**Test the forge3d GPU map** (geographic city labels are deliberately disabled by default until the map's camera footprint is confirmed):

```bash
python senegal_rainy_seasons_updated.py --years 2015 --renderer forge3d --no-video
```

**Make the full animation**, 1976–2024 (49 annual JJAS seasons):

```bash
python senegal_rainy_seasons_updated.py --renderer matplotlib --start 1976 --end 2024 --fps 4
```

For **50 seasons, 1975–2024**, if all data years are available:

```bash
python senegal_rainy_seasons_updated.py --renderer matplotlib --start 1975 --end 2024
```

**June–August instead of JJAS**:

```bash
python senegal_rainy_seasons_updated.py --renderer matplotlib --years 2015 --months 6 7 8 --no-video
```

To see city labels on a forge3d render, add `--forge-labels`, **but only after checking that the image edges align exactly with the DEM's projected raster bounds**. Otherwise the labels may be offset. The Matplotlib renderer does not have this alignment uncertainty and is the safe starting point for exact geographic overlays.

## Outputs

`senegal_out/` will contain:

- `renders/map_YYYY.png`: rendered annual climate map.
- `frames/frame_0000.png`, etc.: completed animation plates.
- `senegal_rainy_seasons.mp4` (when ffmpeg is installed and video enabled).
- `country_gridcell_means.csv`: annual Senegal-masked means of pixel-level anomalies.
- `run_metadata.json`: configuration, baseline, and methods/caveats.
- `cache/senegal_adm0.geojson`: tiny reference polygon cached for reuse.

**No raw TerraClimate files are downloaded and kept.** The script uses lazily accessed Zarr, subsets the country and relevant season/months, then computes the reduced annual climate grids in memory. Internet access to Planetary Computer and geoBoundaries is required. Some remote-service calls can fail temporarily (including Zarr access); retry if a transient network error occurs.

## Method, accuracy, and caveats

1. **Baseline**: 1976–2005 (30 complete rainy seasons) even when rendering only one target year.
2. **Temperature**: TerraClimate monthly tmax and tmin are averaged, then a day-weighted seasonal mean is calculated. Temperature anomaly = seasonal temperature minus baseline seasonal mean in °C.
3. **Precipitation**: TerraClimate monthly ppt totals are summed over season. Rainfall anomaly = `(annual seasonal mm − baseline seasonal mm) / baseline seasonal mm × 100`. Pixels with baseline rainfall <= 20 mm are left out of percentage maps and rainfall means.
4. **Normal bins**: [-0.5, +0.5) °C and [-25, +25) % (with lower/upper boundary classification controlled by `np.searchsorted(..., side="right")`). Both axes use 3 classes, producing 9 colours. Palette matches the original supplied script rather than an exact measurement from a figure.
5. **Spatial alignment**: climate is reprojected from lon/lat to a 1-km UTM 28N DEM display grid, then masked by geoBoundaries ADM0. Rainfall and temperature are bilinearly resampled, appropriate for *visualizing continuous fields*, but **upsampling ~4-km TerraClimate does not create new climate detail**. The Copernicus DEM GLO-90 is a *digital surface model*, not bare-earth LiDAR DTM.
6. **National averages**: arithmetic means of valid, Senegal-masked 1-km projected grid cells. A mean of cell-level percent rainfall anomalies is **not equivalent** to the country-level percentage change in total precipitation. For precise national estimates, calculate area-weighted native-grid zonal aggregates with partial-cell coverage.
7. **Season semantics**: JJAS (June–September) for each calendar year. This is not necessarily the local agronomic onset/cessation of rainy season everywhere in Senegal. Modify `--months` if needed; the annual baseline uses the same set of months.
8. **forge3d camera**: style/overlay calls match the published public example, but matching terrain bounds to the rendered camera extent is not certified, so city labels are opt-in on the forge3d output. When accurate overlays are essential, use Matplotlib mode.
9. **Rendering/relief**: `Z_EXAGGERATION=12` is for display only. It must never be interpreted as a measured elevation change.
10. **Transparency/coverage**: Gambia is outside the Senegal polygon and is blank in the anomaly overlay. In forge3d mode the native terrain/background appearance outside the anomaly mask depends on the renderer and should be visually checked.

## Validation already performed

- Python compilation passed.
- Four offline tests passed: seasonal climate calculations, 3×3 classification and mask, missing month rejection, and synthetic map/frame composition.
- **Not validated here**: live Planetary Computer queries, actual DEM mosaic, native forge3d GPU render, and geographic camera alignment on a forge3d frame. These require network access and compatible hardware.

## References

- [forge3d thematic-raster guide](https://github.com/milos-agathon/forge3d/blob/main/docs/guides/thematic_raster_styles.md)
- [forge3d public example](https://github.com/milos-agathon/forge3d/blob/main/examples/mapscene_thematic_rasters.py)
- [Planetary Computer Zarr documentation](https://planetarycomputer.microsoft.com/docs/quickstarts/reading-zarr-data/)
- [TerraClimate variables and units](https://www.climatologylab.org/terraclimate-variables.html)
