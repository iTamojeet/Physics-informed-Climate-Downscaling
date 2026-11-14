"""
Stage 4: train one downscaling variant (5x, 0.25 -> 0.05 deg, CHIRPS-coarsened input).

Same generator (SRResNet + fine-grid DEM) for every variant; only the loss/constraint changes:
  mse            : --adv 0                    MSE only (SRResNet-MSE baseline)
  mse_hard       : --adv 0 --hard             MSE + conservation layer
  gan            : --adv 0.005                MSE + adversarial
  gan_soft_l<L>  : --adv 0.005 --lam L        MSE + adversarial + lam * L_physics
  gan_hard       : --adv 0.005 --hard         MSE + adversarial + conservation layer

Losses are in normalised log1p space (stable for zero-inflated rain). L_physics is the mean
|log1p(block-mean(pred)) - log1p(coarse input)| over land blocks, with block means taken in mm/day.
--hard rescales each 5x5 block multiplicatively so its (land) mean equals the coarse input exactly.

Colab:  python train.py --variant gan_hard --adv 0.005 --hard
Smoke test (Mac): python train.py --variant smoke --iters 50 --data ../02_preprocessing/data/processed \
                  --ckpt ./ckpt_smoke --code ../common     (PYTORCH_ENABLE_MPS_FALLBACK=1 if MPS complains)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True)
ap.add_argument("--adv", type=float, default=0.0)
ap.add_argument("--lam", type=float, default=0.0)
ap.add_argument("--hard", action="store_true")
ap.add_argument("--iters", type=int, default=20000)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", default="/content/data/processed")
ap.add_argument("--ckpt", default="/content/drive/MyDrive/GeoAI_Portfolio/Climate-Downscaling/checkpoints")
ap.add_argument("--code", default=".")
a = ap.parse_args()
sys.path.insert(0, str(Path(a.code).resolve()))
from metrics import evaluate  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
print("device:", dev, "| args:", vars(a))

# ---------------- data (everything lives on the device) ----------------
D = Path(a.data)
meta = json.loads((D / "meta.json").read_text())
MU, SD, F5, P = meta["logp1_mean"], meta["logp1_std"], meta["factor"], meta["patch_coarse"]
tr0, tr1 = meta["split_day_index"]["train"]
v0, v1 = meta["split_day_index"]["val"]
chirps = torch.from_numpy(np.load(D / "chirps_daily.npy")).to(dev)
coarse = torch.from_numpy(np.load(D / "coarse_daily.npy")).to(dev)
mask_np = np.load(D / "land_mask.npy")
mask = torch.from_numpy(mask_np).float().to(dev)
dem_n = (torch.from_numpy(np.load(D / "dem.npy")).to(dev) - meta["dem_mean"]) / meta["dem_std"] * mask
dem_c = Fn.avg_pool2d(dem_n[None, None], F5)[0, 0]
patches = np.load(D / "patches.npy")

pm = np.stack([coarse[tr0:tr1, r:r + P, c:c + P].mean(dim=(1, 2)).cpu().numpy() for r, c in patches], 1)
wet = np.argwhere(pm > 0.5)  # (day_rel, patch_idx) pairs with real rain; oversampled in training
print("train wet samples:", len(wet), "of", pm.size)


def norm(x):
    return (torch.log1p(x) - MU) / SD


def denorm(z):
    return torch.expm1(z * SD + MU).clamp(min=0)


def bmean(x, m):
    cnt = Fn.avg_pool2d(m, F5) * F5 * F5
    return Fn.avg_pool2d(x * m, F5) * F5 * F5 / cnt.clamp(min=1), cnt > 0


# ---------------- models ----------------
class Res(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.b = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.PReLU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(s, x):
        return x + s.b(x)


class G(nn.Module):
    def __init__(s, c=64, n=8):
        super().__init__()
        s.head = nn.Conv2d(2, c, 3, padding=1)
        s.body = nn.Sequential(*[Res(c) for _ in range(n)])
        s.up = nn.Sequential(nn.Conv2d(c, c * F5 * F5, 3, padding=1), nn.PixelShuffle(F5), nn.PReLU())
        s.tail = nn.Sequential(nn.Conv2d(c + 1, c, 3, padding=1), nn.PReLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(s, xc, dem_f):
        h = s.head(xc)
        h = h + s.body(h)
        return s.tail(torch.cat([s.up(h), dem_f], 1))


class Dis(nn.Module):
    def __init__(s, c=32):
        super().__init__()
        L, ch = [], 2
        for o, st in [(c, 1), (c, 2), (2 * c, 1), (2 * c, 2), (4 * c, 1), (4 * c, 2)]:
            L += [nn.Conv2d(ch, o, 3, stride=st, padding=1), nn.LeakyReLU(0.2)]
            ch = o
        s.f, s.out = nn.Sequential(*L), nn.Conv2d(ch, 1, 3, padding=1)

    def forward(s, fine, cond):
        return s.out(s.f(torch.cat([fine, cond], 1)))


def generate(g, cm, dc, df, m, hard):
    mm = denorm(g(torch.cat([norm(cm), dc], 1), df)) * m
    if hard:  # multiplicative conservation layer: block mean of output == coarse input
        mm = mm + 1e-4 * m
        bm, _ = bmean(mm, m)
        mm = mm * Fn.interpolate(cm / (bm + 1e-6), scale_factor=F5, mode="nearest")
    return mm


def losses(mm, tgt, cm, m):
    mse = (((norm(mm) - norm(tgt)) ** 2) * m).sum() / m.sum()
    bm, ok = bmean(mm, m)
    phys = ((torch.log1p(bm) - torch.log1p(cm)).abs() * ok).sum() / ok.sum()
    return mse, phys


def batch():
    n = a.bs
    pick = rng.random(n) < 0.8
    rnd = np.stack([rng.integers(tr1 - tr0, size=n), rng.integers(len(patches), size=n)], 1)
    idx = np.where(pick[:, None], wet[rng.integers(len(wet), size=n)], rnd)
    days, (r, c) = idx[:, 0] + tr0, patches[idx[:, 1]].T
    fs = lambda t, d, rr, cc: t[d, rr * F5:(rr + P) * F5, cc * F5:(cc + P) * F5]  # noqa: E731
    cs = lambda t, d, rr, cc: t[d, rr:rr + P, cc:cc + P]  # noqa: E731
    tgt = torch.stack([fs(chirps, d, rr, cc) for d, rr, cc in zip(days, r, c)])[:, None]
    cm = torch.stack([cs(coarse, d, rr, cc) for d, rr, cc in zip(days, r, c)])[:, None]
    m = torch.stack([fs(mask[None], 0, rr, cc) for rr, cc in zip(r, c)])[:, None]
    df = torch.stack([fs(dem_n[None], 0, rr, cc) for rr, cc in zip(r, c)])[:, None]
    dc = torch.stack([cs(dem_c[None], 0, rr, cc) for rr, cc in zip(r, c)])[:, None]
    return tgt, cm, m, df, dc


@torch.no_grad()
def validate(g):
    g.eval()
    days = np.arange(v0, v1, 4)
    out = []
    for i in range(0, len(days), 16):
        dd = torch.as_tensor(days[i:i + 16], device=dev)
        n = len(dd)
        ex = lambda t: t[None, None].expand(n, -1, -1, -1)  # noqa: E731
        out.append(generate(g, coarse[dd][:, None], ex(dem_c), ex(dem_n), ex(mask), a.hard)[:, 0].cpu().numpy())
    g.train()
    dd = torch.as_tensor(days, device=dev)
    return evaluate(np.concatenate(out), chirps[dd].cpu().numpy(), coarse[dd].cpu().numpy(), mask_np)


# ---------------- training ----------------
g = G().to(dev)
og = torch.optim.Adam(g.parameters(), 2e-4)
if a.adv > 0:
    d = Dis().to(dev)
    od = torch.optim.Adam(d.parameters(), 2e-4, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()
out_dir = Path(a.ckpt) / a.variant
out_dir.mkdir(parents=True, exist_ok=True)
best, hist, t0 = float("inf"), [], time.time()

for it in range(1, a.iters + 1):
    tgt, cm, m, df, dc = batch()
    mm = generate(g, cm, dc, df, m, a.hard)
    mse, phys = losses(mm, tgt, cm, m)
    loss = mse + a.lam * phys
    if a.adv > 0:
        zp, zt, cond = norm(mm), norm(tgt), Fn.interpolate(norm(cm), scale_factor=F5, mode="nearest")
        lo = bce(d(zt, cond), torch.ones_like(d(zt, cond))) + bce(d(zp.detach(), cond), torch.zeros_like(d(zt, cond)))
        od.zero_grad()
        lo.backward()
        od.step()
        logits = d(zp, cond)
        loss = loss + a.adv * bce(logits, torch.ones_like(logits))
    og.zero_grad()
    loss.backward()
    og.step()

    if it % 500 == 0:
        print(f"it {it} mse {mse.item():.4f} phys {phys.item():.4f} ({time.time() - t0:.0f}s)")
    if it % 2000 == 0 or it == a.iters:
        v = validate(g)
        hist.append({"it": it, **v})
        print("  val", {k: round(x, 3) for k, x in v.items()})
        ck = {"g": g.state_dict(), "it": it, "args": vars(a), "val": v, "hist": hist}
        torch.save(ck, out_dir / "latest.pt")
        if v["rmse"] < best:
            best = v["rmse"]
            torch.save(ck, out_dir / "best.pt")
print("done. best val rmse:", round(best, 3))
