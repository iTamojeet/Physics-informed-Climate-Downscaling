"""
Stage 5: evaluate every trained variant on the held-out test years (2019-2022).

Run from 05_evaluation/ after downloading the Drive checkpoints folder to ./checkpoints/
(layout: checkpoints/<variant>/{latest.pt,best.pt}):
    python evaluate.py
Full-domain inference (160x200) on every test day; metrics from common/metrics.py plus a
gradient-magnitude ratio. latest.pt (iteration 20000, EMA weights) is the primary result.

Writes ./results/test_results.json and ./figures/{maps,spectrum,tradeoff}.png
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints")
ap.add_argument("--data", default="../02_preprocessing/data/processed")
ap.add_argument("--split", default="test")
a = ap.parse_args()
sys.path.insert(0, str(Path("../common").resolve()))
from metrics import evaluate  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
D = Path(a.data)
meta = json.loads((D / "meta.json").read_text())
MU, SD, F5 = meta["logp1_mean"], meta["logp1_std"], meta["factor"]
d0, d1 = meta["split_day_index"][a.split]

truth = np.load(D / "chirps_daily.npy", mmap_mode="r")[d0:d1].copy()
coarse = np.load(D / "coarse_daily.npy", mmap_mode="r")[d0:d1].copy()
mask = np.load(D / "land_mask.npy")
mask_t = torch.from_numpy(mask).float().to(dev)
dem_n = (torch.from_numpy(np.load(D / "dem.npy")).to(dev) - meta["dem_mean"]) / meta["dem_std"] * mask_t
dem_c = Fn.avg_pool2d(dem_n[None, None], F5)[0, 0]
days = [dt.date(2001, 1, 1) + dt.timedelta(i) for i in range(d0, d1)]
jjas = np.array([d.month in (6, 7, 8, 9) for d in days])
cm_all = torch.from_numpy(coarse)[:, None].to(dev)
print("device:", dev, "| split:", a.split, "| days:", len(days))


def norm(x):
    return (torch.log1p(x) - MU) / SD


def denorm(z):
    return torch.expm1(z * SD + MU).clamp(min=0)


def bmean(x, m):
    cnt = Fn.avg_pool2d(m, F5) * F5 * F5
    return Fn.avg_pool2d(x * m, F5) * F5 * F5 / cnt.clamp(min=1), cnt > 0


class Res(nn.Module):  # identical to train.py so state_dict keys match
    def __init__(s, c):
        super().__init__()
        s.b = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.PReLU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(s, x):
        return x + s.b(x)


class G(nn.Module):
    def __init__(s, c=64, n=8, up="shuffle"):
        super().__init__()
        s.head = nn.Conv2d(2, c, 3, padding=1)
        s.body = nn.Sequential(*[Res(c) for _ in range(n)])
        s.up = (nn.Sequential(nn.Upsample(scale_factor=F5, mode="nearest"), nn.Conv2d(c, c, 3, padding=1), nn.PReLU())
                if up == "resize" else
                nn.Sequential(nn.Conv2d(c, c * F5 * F5, 3, padding=1), nn.PixelShuffle(F5), nn.PReLU()))
        s.tail = nn.Sequential(nn.Conv2d(c + 1, c, 3, padding=1), nn.PReLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(s, xc, dem_f):
        h = s.head(xc)
        h = h + s.body(h)
        return s.tail(torch.cat([s.up(h), dem_f], 1))


@torch.no_grad()
def predict(g, hard):
    out = []
    for i in range(0, len(cm_all), 16):
        cm = cm_all[i:i + 16]
        n = len(cm)
        ex = lambda t: t[None, None].expand(n, -1, -1, -1)  # noqa: E731
        mm = denorm(g(torch.cat([norm(cm), ex(dem_c)], 1), ex(dem_n))) * ex(mask_t)
        if hard:
            mm = mm + 1e-4 * ex(mask_t)
            bm, _ = bmean(mm, ex(mask_t))
            mm = mm * Fn.interpolate(cm / (bm + 1e-6), scale_factor=F5, mode="nearest")
        out.append(mm[:, 0].cpu().numpy())
    return np.concatenate(out)


def grad_ratio(pred, tr):
    mx, my = mask[:, 1:] & mask[:, :-1], mask[1:] & mask[:-1]
    g = lambda x: np.abs(np.diff(x, axis=2))[:, mx].mean() + np.abs(np.diff(x, axis=1))[:, my].mean()  # noqa: E731
    return float(g(np.where(mask, pred, 0)) / g(tr))


def radial_spectrum(x, step=5, nb=25):
    f = np.fft.fftshift(np.fft.fft2(x[::step].astype("float64")), axes=(-2, -1))
    p = (np.abs(f) ** 2).mean(axis=0)
    H, W = p.shape
    yy, xx = np.indices((H, W))
    r = np.hypot((yy - H // 2) / H, (xx - W // 2) / W)
    edges = np.linspace(0, 0.5, nb + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < nb)
    cnt = np.bincount(idx[ok], minlength=nb)
    ps = np.bincount(idx[ok], p.ravel()[ok], nb) / np.maximum(cnt, 1)
    return 0.5 * (edges[1:] + edges[:-1]), ps


def lattice_score(x, step=5):
    """Peak-to-background power at the 5-pixel lattice frequencies (0.2 and 0.4 cycles/px and their
    2D combinations). Sub-pixel-shuffle checkerboard artefacts raise this; real rain fields do not."""
    p = (np.abs(np.fft.fft2(x[::step].astype("float64"))) ** 2).mean(axis=0)
    H, W = p.shape
    fy, fx = H // F5, W // F5
    sc = []
    for n, m in [(0, fx), (fy, 0), (fy, fx), (0, 2 * fx), (2 * fy, 0), (2 * fy, 2 * fx), (fy, 2 * fx), (2 * fy, fx)]:
        win = p[max(n - 4, 0):n + 5, max(m - 4, 0):m + 5]
        sc.append(p[n, m] / ((win.sum() - p[n, m]) / (win.size - 1)))
    return float(np.mean(sc))


def edge_ratio(x):
    """Mean |jump| across 5x5 block borders / mean |jump| inside blocks (land pairs only).
    Multiplicative block rescaling can create steps at block borders; nearest-neighbour is the extreme."""
    x = np.where(mask, x[::5], 0)
    out = []
    for ax in (1, 2):
        d = np.abs(np.diff(x, axis=ax))
        valid = (mask[1:] & mask[:-1]) if ax == 1 else (mask[:, 1:] & mask[:, :-1])
        pos = np.arange(d.shape[ax]) % F5 == F5 - 1
        pos = pos[:, None] if ax == 1 else pos[None, :]
        out.append(d[:, valid & pos].mean() / d[:, valid & ~pos].mean())
    return float(np.mean(out))


# ---------------- predictions ----------------
x = torch.from_numpy(coarse)[:, None]
preds = {
    "nearest": Fn.interpolate(x, scale_factor=F5, mode="nearest")[:, 0].numpy(),
    "bicubic": Fn.interpolate(x, scale_factor=F5, mode="bicubic", align_corners=False).clamp(min=0)[:, 0].numpy(),
}
iters = {}
for vdir in sorted(p for p in Path(a.ckpt).iterdir() if p.is_dir()):
    for which in ("latest", "best"):
        f = vdir / f"{which}.pt"
        if not f.exists():
            continue
        ck = torch.load(f, map_location="cpu", weights_only=False)
        g = G(up=ck["args"].get("up", "shuffle")).to(dev)
        g.load_state_dict(ck["g"])
        g.eval()
        name = f"{vdir.name}/{which}"
        preds[name] = predict(g, ck["args"]["hard"]).astype("float32")
        iters[name] = ck["it"]
        print("predicted", name, "(iteration", ck["it"], ")")

# ---------------- metrics ----------------
results = {}
for name, pred in preds.items():
    results[name] = {"iteration": iters.get(name)}
    for tag, sel in (("all_days", slice(None)), ("jjas", jjas)):
        r = evaluate(pred[sel], truth[sel], coarse[sel], mask)
        r["grad_ratio"] = grad_ratio(pred[sel], truth[sel])
        r["lattice_ratio"] = lattice_score(np.where(mask, pred[sel], 0)) / lattice_score(truth[sel])
        r["edge_ratio"] = edge_ratio(pred[sel]) / edge_ratio(truth[sel])
        results[name][tag] = r
Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)
Path(f"results/{a.split}_results.json").write_text(json.dumps(results, indent=2))

cols = ["rmse", "mae", "conservation", "p99_bias_pct", "wet_freq_ratio", "subcoarse_power", "grad_ratio", "lattice_ratio", "edge_ratio"]
for tag in ("all_days", "jjas"):
    print(f"\n== {a.split} {tag} ==")
    print(f"{'model':28s}" + "".join(f"{c[:11]:>12s}" for c in cols))
    for name, r in results.items():
        print(f"{name:28s}" + "".join(f"{r[tag][c]:12.3f}" for c in cols))

# ---------------- figures ----------------
primary = {n: p for n, p in preds.items() if n in ("nearest", "bicubic") or n.endswith("/latest")}

# 1) maps for the wettest monsoon day in the split
dm = np.where(jjas, truth[:, mask].mean(axis=1), -1)
k = int(np.argmax(dm))
panels = [("CHIRPS truth", truth[k])] + [(n.replace("/latest", ""), p[k]) for n, p in primary.items()]
nr, nc = 2, (len(panels) + 1) // 2
fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc, 3.0 * nr))
vmax = float(np.percentile(truth[k][mask], 99.5))
for ax, (title, field) in zip(axs.ravel(), panels):
    im = ax.imshow(np.where(mask, field, np.nan), vmin=0, vmax=vmax, cmap="YlGnBu")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
for ax in axs.ravel()[len(panels):]:
    ax.axis("off")
fig.colorbar(im, ax=axs, shrink=0.7, label="mm/day")
fig.suptitle(f"Test day {days[k]} (latest.pt for learned models)", fontsize=10)
fig.savefig("figures/maps.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 2) radial power spectra
fig, ax = plt.subplots(figsize=(6, 4.2))
kk, ps = radial_spectrum(truth)
ax.loglog(kk[1:], ps[1:], "k", lw=2.2, label="CHIRPS truth")
for n, p in primary.items():
    kk, ps = radial_spectrum(np.where(mask, p, 0))
    ax.loglog(kk[1:], ps[1:], lw=1.2, label=n.replace("/latest", ""))
ax.axvline(0.1, color="grey", ls="--", lw=0.8)
ax.text(0.102, ax.get_ylim()[0] * 3, "0.25 deg\nNyquist", fontsize=7, color="grey")
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel("radial frequency (cycles / 0.05 deg pixel)")
ax.set_ylabel("power")
ax.legend(fontsize=7)
fig.savefig("figures/spectrum.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 3) trade-off: pixel error vs texture, annotated with conservation
fig, ax = plt.subplots(figsize=(6, 4.2))
for n in primary:
    r = results[n]["all_days"]
    ax.scatter(r["rmse"], r["subcoarse_power"], s=40)
    ax.annotate(f"{n.replace('/latest', '')}\ncons {r['conservation']:.2f}", (r["rmse"], r["subcoarse_power"]),
                fontsize=6, xytext=(3, 3), textcoords="offset points")
ax.axhline(1.0, color="grey", ls="--", lw=0.8)
ax.set_xlabel("RMSE (mm/day)")
ax.set_ylabel("sub-coarse power ratio (1 = truth)")
fig.savefig("figures/tradeoff.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nsaved results and figures")