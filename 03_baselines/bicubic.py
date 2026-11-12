"""
Stage 3: non-learned baselines on the test years (2019-2022), CHIRPS-coarsened input.
  nearest : block replication (exactly mass-conserving, no new detail)
  bicubic : standard interpolation, clamped at 0 (smooth, not exactly conserving)

Run from 03_baselines/:  python bicubic.py
Writes ./results/baselines.json
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path("../common").resolve()))
from metrics import evaluate  # noqa: E402

P = Path("../02_preprocessing/data/processed")
meta = json.loads((P / "meta.json").read_text())
a, b = meta["split_day_index"]["test"]

truth = np.load(P / "chirps_daily.npy", mmap_mode="r")[a:b].copy()
coarse = np.load(P / "coarse_daily.npy", mmap_mode="r")[a:b].copy()
mask = np.load(P / "land_mask.npy")

days = [dt.date(2001, 1, 1) + dt.timedelta(i) for i in range(a, b)]
jjas = np.array([d.month in (6, 7, 8, 9) for d in days])

x = torch.from_numpy(coarse)[:, None]
preds = {
    "nearest": Fn.interpolate(x, scale_factor=5, mode="nearest")[:, 0].numpy(),
    "bicubic": Fn.interpolate(x, scale_factor=5, mode="bicubic", align_corners=False)
    .clamp(min=0)[:, 0].numpy(),
}

results = {}
for name, pred in preds.items():
    results[name] = {
        "all_days": evaluate(pred, truth, coarse, mask),
        "jjas": evaluate(pred[jjas], truth[jjas], coarse[jjas], mask),
    }
    for k, v in results[name].items():
        print(name, k, {m: round(val, 3) for m, val in v.items()})

Path("results").mkdir(exist_ok=True)
Path("results/baselines.json").write_text(json.dumps(results, indent=2))
