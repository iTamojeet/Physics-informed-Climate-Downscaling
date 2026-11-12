"""
Stage 2: merge CHIRPS tiles, build land mask, coarsen to 0.25 deg, index patches.

Run from 02_preprocessing/:  python preprocess.py
Reads  ../01_data_acquisition/data/raw/{chirps/*.tif, srtm_0p05.tif}
Writes ./data/processed/:
  chirps_daily.npy   (8035,160,200) float32, mm/day, ocean = 0
  land_mask.npy      (160,200) bool, True = valid land
  coarse_daily.npy   (8035,32,40) float32, mask-aware 0.25 deg block mean (mm/day)
  coarse_valid.npy   (32,40) float32, land fraction of each coarse block
  dem.npy            (160,200) float32, metres, ocean = 0
  patches.npy        (N,2) int, coarse-grid (row, col) origins of 8x8-block patches
  meta.json          split day-index ranges, norm stats, patch config
Day index i corresponds to 2001-01-01 + i days (files sorted by name are chronological,
checked by the monthly-cycle test in 01_data_acquisition/diagnose_chirps.py).
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import rasterio

RAW = Path("../01_data_acquisition/data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

H, W, F = 160, 200, 5          # fine grid size, upscale factor (0.05 -> 0.25 deg)
CH, CW = H // F, W // F        # coarse grid: 32 x 40
P, STRIDE, MIN_LAND = 8, 4, 0.8  # patch = 8x8 coarse blocks (40x40 fine px)
N_DAYS = 8035

# 1) merge tiles, cropping the extra east column some tiles carry (origin is identical)
files = sorted((RAW / "chirps").glob("*.tif"))
chunks = []
for f in files:
    with rasterio.open(f) as src:
        chunks.append(src.read()[:, :H, :W].astype("float32"))
data = np.concatenate(chunks, axis=0)
assert data.shape == (N_DAYS, H, W), data.shape

# 2) land mask = pixels that are never finite-invalid (verified permanent, not day-specific)
mask = np.isfinite(data).all(axis=0)
assert not (np.isfinite(data).any(axis=0) ^ mask).any(), "mask differs between days"
data[:, ~mask] = 0.0
assert (data[:, mask] >= 0).all(), "negative precipitation on land"
print(f"land fraction: {mask.mean():.3f} | max {data.max():.1f} mm/day")

# 3) mask-aware coarse field: mean over valid fine pixels in each 5x5 block
cnt = mask.reshape(CH, F, CW, F).sum(axis=(1, 3)).astype("float32")
coarse = data.reshape(N_DAYS, CH, F, CW, F).sum(axis=(2, 4)) / np.maximum(cnt, 1)
valid = cnt / (F * F)

# 4) DEM (ocean -> 0)
with rasterio.open(RAW / "srtm_0p05.tif") as src:
    dem = np.nan_to_num(src.read(1)[:H, :W].astype("float32"))
dem[~mask] = 0.0

# 5) patch origins on the coarse grid with enough land
patches = []
for r in range(0, CH - P + 1, STRIDE):
    for c in range(0, CW - P + 1, STRIDE):
        land = mask[r * F:(r + P) * F, c * F:(c + P) * F].mean()
        if land >= MIN_LAND:
            patches.append((r, c))
patches = np.array(patches, dtype=np.int32)
print("patches kept:", len(patches))

# 6) chronological split by year, norm stats from TRAIN land pixels only
days = [dt.date(2001, 1, 1) + dt.timedelta(i) for i in range(N_DAYS)]
year = np.array([d.year for d in days])
split = {
    "train": [0, int((year <= 2016).sum())],
    "val": [int((year <= 2016).sum()), int((year <= 2018).sum())],
    "test": [int((year <= 2018).sum()), N_DAYS],
}
tr = slice(*split["train"])
logp = np.log1p(data[tr][:, mask])
meta = {
    "split_day_index": split,
    "logp1_mean": float(logp.mean()), "logp1_std": float(logp.std()),
    "dem_mean": float(dem[mask].mean()), "dem_std": float(dem[mask].std()),
    "factor": F, "patch_coarse": P, "stride_coarse": STRIDE, "min_land": MIN_LAND,
    "start_date": "2001-01-01", "n_days": N_DAYS,
}

np.save(OUT / "chirps_daily.npy", data)
np.save(OUT / "land_mask.npy", mask)
np.save(OUT / "coarse_daily.npy", coarse.astype("float32"))
np.save(OUT / "coarse_valid.npy", valid)
np.save(OUT / "dem.npy", dem)
np.save(OUT / "patches.npy", patches)
(OUT / "meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
