"""
Stage 1: data acquisition for physics-informed precipitation downscaling.

Region: eastern India, lon 82-92E, lat 20-28N. Period: 2001-2022.
  - CHIRPS v2 daily, 0.05 deg  -> high-res target (GEE: UCSB-CHG/CHIRPS/DAILY)
  - ERA5 hourly total_precipitation, 0.25 deg -> coarse input for Stage 6 (CDS API)
  - SRTM elevation, mean-aggregated to 0.05 deg -> static covariate (GEE)

Prereqs:
  pip install -r requirements.txt
  earthengine authenticate           (once)
  .env in the project root (see .env.example) with CDSAPI_URL, CDSAPI_KEY, GEE_PROJECT
  Accept the ERA5 licence on the CDS dataset page (Download tab)

Usage:
  python3 download_data.py chirps
  python3 download_data.py era5 2019 2019
  python3 download_data.py era5 2020 2020
  python3 download_data.py era5 2021 2021
  python3 download_data.py era5 2022 2022
  python3 download_data.py dem

Outputs go to ./data/raw/ (keep local; do not commit).
"""
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

W, S, E, N = 82.0, 20.0, 92.0, 28.0
YEARS = range(2001, 2023)
RES = 0.05
NX, NY = int(round((E - W) / RES)), int(round((N - S) / RES))  # 200 x 160
RAW = Path("data/raw")


def _fetch_chirps(coll, region, transform, start, end, stem, out):
    """Download [start, end) as one GeoTIFF; on a size rejection, halve the range."""
    import ee
    import io
    import urllib.request
    import zipfile
    from datetime import date, timedelta

    f = out / f"{stem}.tif"
    if f.exists():
        return
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    ndays = (d1 - d0).days
    img = coll.filterDate(start, end).toBands().toFloat()
    data = None
    for attempt in range(6):
        try:
            url = img.getDownloadURL({
                "region": region, "crs": "EPSG:4326",
                "crs_transform": transform, "format": "GEO_TIFF",
            })
            data = urllib.request.urlopen(url, timeout=300).read()
            break
        except ee.EEException as e:
            if "request size" in str(e) and ndays > 1:
                mid = (d0 + timedelta(days=ndays // 2)).isoformat()
                print(f"  {stem}: too large, splitting at {mid}")
                _fetch_chirps(coll, region, transform, start, mid, stem + "_a", out)
                _fetch_chirps(coll, region, transform, mid, end, stem + "_b", out)
                return
            wait = 5 * 2 ** attempt
            print(f"  {stem}: {e}; retry in {wait}s")
            time.sleep(wait)
        except Exception as e:
            wait = 5 * 2 ** attempt
            print(f"  {stem}: {e}; retry in {wait}s")
            time.sleep(wait)
    if data is None:
        print(f"  giving up on {stem}, rerun later")
        return
    if data[:2] == b"PK":  # multi-band exports can come back zipped
        z = zipfile.ZipFile(io.BytesIO(data))
        data = z.read(z.namelist()[0])
    f.write_bytes(data)
    print("saved", f)


def chirps():
    import ee

    ee.Initialize(project=os.environ["GEE_PROJECT"])
    out = RAW / "chirps"
    out.mkdir(parents=True, exist_ok=True)
    region = ee.Geometry.Rectangle([W, S, E, N], proj="EPSG:4326", geodesic=False)
    transform = [RES, 0, W, 0, -RES, N]
    coll = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")

    for y in YEARS:
        for q, (m0, m1) in enumerate([(1, 4), (4, 7), (7, 10), (10, 13)], 1):
            start = f"{y}-{m0:02d}-01"
            end = f"{y + 1}-01-01" if m1 == 13 else f"{y}-{m1:02d}-01"
            _fetch_chirps(coll, region, transform, start, end, f"chirps_{y}_q{q}", out)


def era5(years=None):
    import cdsapi

    out = RAW / "era5"
    out.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client(url=os.environ["CDSAPI_URL"], key=os.environ["CDSAPI_KEY"])
    for y in (years or YEARS):
        f = out / f"era5_tp_{y}.nc"
        if f.exists():
            continue
        c.retrieve("reanalysis-era5-single-levels", {
            "product_type": "reanalysis",
            "variable": "total_precipitation",
            "year": str(y),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [N, W, S, E],  # N, W, S, E
            "grid": [0.25, 0.25],
            "format": "netcdf",
        }, str(f))
        print("saved", f)
    # NOTE (Stage 2): ERA5 tp at hour t is accumulated over (t-1h, t]. Shift time by
    # -1h before resampling to daily sums (metres -> mm) to match CHIRPS UTC days.
    # The final day of 2022 needs Jan 1 2023 00:00 from the next file; fetch it
    # separately or drop 2022-12-31 from the ERA5 comparison.


def dem():
    import time
    import urllib.request
    import ee
    import rasterio
    from rasterio.merge import merge

    ee.Initialize(project=os.environ["GEE_PROJECT"])
    tiles = RAW / "srtm_tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    srtm = ee.Image("USGS/SRTMGL1_003").select("elevation")
    agg = (srtm.setDefaultProjection("EPSG:4326", None, 30)
           .reduceResolution(ee.Reducer.mean(), maxPixels=65536)
           .reproject(crs="EPSG:4326", crsTransform=[RES, 0, W, 0, -RES, N]))

    for lon in range(int(W), int(E)):
        for lat in range(int(S), int(N)):
            f = tiles / f"srtm_{lon}_{lat}.tif"
            if f.exists():
                continue
            region = ee.Geometry.Rectangle([lon, lat, lon + 1, lat + 1],
                                           proj="EPSG:4326", geodesic=False)
            for attempt in range(5):
                try:
                    url = agg.getDownloadURL({
                        "region": region, "crs": "EPSG:4326",
                        "crs_transform": [RES, 0, lon, 0, -RES, lat + 1],
                        "format": "GEO_TIFF",
                    })
                    f.write_bytes(urllib.request.urlopen(url, timeout=300).read())
                    break
                except Exception as e:
                    print(f"  tile {lon},{lat}: {e}; retry")
                    time.sleep(5 * 2 ** attempt)
            else:
                print(f"  giving up on tile {lon},{lat}, rerun later")

    srcs = [rasterio.open(p) for p in sorted(tiles.glob("srtm_*.tif"))]
    mosaic, tf = merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=tf)
    with rasterio.open(RAW / "srtm_0p05.tif", "w", **meta) as dst:
        dst.write(mosaic)
    print("saved", RAW / "srtm_0p05.tif", mosaic.shape)

def era5_test():
    import cdsapi

    out = RAW / "era5"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "era5_tp_2019_2022.nc"
    if f.exists():
        return
    cdsapi.Client().retrieve("reanalysis-era5-single-levels", {
        "product_type": "reanalysis",
        "variable": "total_precipitation",
        "year": ["2019", "2020", "2021", "2022"],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [N, W, S, E],
        "grid": [0.25, 0.25],
        "format": "netcdf",
    }, str(f))
    print("saved", f)

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "era5" and len(sys.argv) == 4:
        era5(range(int(sys.argv[2]), int(sys.argv[3]) + 1))
    else:
        {"chirps": chirps, "era5": era5, "era5test": era5_test, "dem": dem}[cmd]()