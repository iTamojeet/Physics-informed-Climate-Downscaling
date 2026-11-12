"""Diagnose CHIRPS tiles: odd widths, -inf pixels, and chronological order.
Run from 01_data_acquisition/:  python diagnose_chirps.py
"""
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio

files = sorted(Path("data/raw/chirps").glob("*.tif"))

# 1) width mismatch: which files, and where is their left/top origin
wide, origins = [], Counter()
for f in files:
    with rasterio.open(f) as src:
        origins[(round(src.transform.c, 4), round(src.transform.f, 4), round(src.transform.a, 4))] += 1
        if src.width != 200:
            wide.append((f.name, src.width, src.height, src.count))
print("origins (west, north, pixel size) -> n files:", dict(origins))
print("files with width != 200:", len(wide))
for w in wide[:10]:
    print("  ", w)

# 2) -inf pixels: fixed spatial footprint (ocean mask) or random?
inf_count = None
n_days = 0
total_bands = 0
for f in files:
    with rasterio.open(f) as src:
        arr = src.read()[:, :, :200]  # crop any 201-wide tile for this check
        total_bands += arr.shape[0]
        m = ~np.isfinite(arr)
        c = m.sum(axis=0)
        inf_count = c if inf_count is None else inf_count + c
        n_days += arr.shape[0]
frac = inf_count / n_days
always = (frac == 1.0).mean()
never = (frac == 0.0).mean()
partial = 1 - always - never
print(f"pixels non-finite on ALL days: {always:.1%} | on NO days: {never:.1%} | on SOME days: {partial:.1%}")
print("(all-days share is the permanent mask, likely ocean; some-days would be a real problem)")

# 3) chronological sanity: domain-mean rain by day-of-year should peak Jul-Aug
series = []
for f in files:
    with rasterio.open(f) as src:
        arr = src.read()[:, :, :200].astype("float64")
        arr[~np.isfinite(arr)] = np.nan
        series.extend(np.nanmean(arr, axis=(1, 2)))
series = np.array(series)
print("total days:", len(series))
if len(series) == 8035:
    import datetime as dt
    days = [dt.date(2001, 1, 1) + dt.timedelta(i) for i in range(8035)]
    month = np.array([d.month for d in days])
    clim = [np.nanmean(series[month == m]) for m in range(1, 13)]
    print("monthly mean (mm/day), Jan..Dec:", [round(x, 2) for x in clim])
    print("peak month:", int(np.argmax(clim)) + 1, "(expect 7 or 8 if order is chronological)")
