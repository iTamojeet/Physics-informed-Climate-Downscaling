"""Check the CHIRPS download: day coverage, grid shape, NaNs, value range.
Run from 01_data_acquisition/:  python verify_chirps.py
"""
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio

files = sorted(Path("data/raw/chirps").glob("*.tif"))
print("tif files:", len(files))

dates, shapes, nan_total, vmin, vmax, no_desc = [], set(), 0, np.inf, -np.inf, 0
for f in files:
    with rasterio.open(f) as src:
        shapes.add((src.height, src.width))
        arr = src.read()
        nan_total += int(np.isnan(arr).sum())
        vmin, vmax = min(vmin, float(np.nanmin(arr))), max(vmax, float(np.nanmax(arr)))
        for d in src.descriptions:
            m = re.match(r"(\d{8})", d or "")
            if m:
                dates.append(m.group(1))
            else:
                no_desc += 1

print("grid shapes:", shapes, "(expected {(160, 200)})")
print("NaN pixels:", nan_total, "| value range (mm/day):", round(vmin, 2), "to", round(vmax, 2))

if dates:
    ds = sorted(dates)
    start, end = date(2001, 1, 1), date(2022, 12, 31)
    expected = {(start + timedelta(i)).strftime("%Y%m%d") for i in range((end - start).days + 1)}
    got = set(ds)
    print("bands:", len(dates), "| unique days:", len(got), "| expected days:", len(expected))
    print("missing days:", len(expected - got), "| duplicates:", len(dates) - len(got))
    if expected - got:
        print("first missing:", sorted(expected - got)[:10])
else:
    print("no date info in band descriptions; total bands unknown, expected 8035 (2001-2022)")
    print("bands without date:", no_desc)
