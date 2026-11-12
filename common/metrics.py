"""Shared evaluation metrics. All fields are (T, H, W) in mm/day on the 0.05 deg grid.

evaluate() reports, over land pixels only:
  rmse, mae        : pixel errors (mm/day)
  psnr             : 20*log10(peak/rmse), fixed peak = 100 mm/day (comparable across models)
  conservation     : mean |block-mean(pred) - coarse input| over land blocks (mm/day).
                     0 means the 5x5 block mass is exactly preserved.
  p95_bias, p99_bias : wet-day quantile bias in percent (pred vs truth, wet = >1 mm/day)
  wet_freq_ratio   : wet-day frequency, pred / truth
  subcoarse_power  : spectral power at scales finer than the 0.25 deg input can resolve
                     (radial frequency > 0.1 cycles/px), pred / truth. 1 = realistic texture.
"""
import numpy as np

F = 5  # upscale factor


def block_mean(x, mask):
    """Mask-aware mean over FxF blocks. Returns (T, H/F, W/F) and a bool array of valid blocks."""
    T, H, W = x.shape
    s = (x * mask).reshape(T, H // F, F, W // F, F).sum(axis=(2, 4))
    c = mask.reshape(H // F, F, W // F, F).sum(axis=(1, 3))
    return s / np.maximum(c, 1), c > 0


def _subcoarse_power(x, step=5):
    f = np.fft.fftshift(np.fft.fft2(x[::step]), axes=(-2, -1))
    p = (np.abs(f) ** 2).mean(axis=0)
    H, W = p.shape
    yy, xx = np.indices((H, W))
    r = np.hypot((yy - H // 2) / H, (xx - W // 2) / W)
    return p[(r > 0.1) & (r <= 0.5)].sum()


def evaluate(pred, truth, coarse, mask, wet=1.0, peak=100.0):
    pred = np.where(mask, pred, 0.0).astype("float64")
    truth = truth.astype("float64")
    pv, tv = pred[:, mask], truth[:, mask]
    err = pv - tv
    rmse = float(np.sqrt((err ** 2).mean()))

    bm, ok = block_mean(pred, mask)
    cons = float(np.abs(bm - coarse)[:, ok].mean())

    def wq(v, q):
        w = v[v > wet]
        return float(np.percentile(w, q)) if w.size else float("nan")

    return {
        "rmse": rmse,
        "mae": float(np.abs(err).mean()),
        "psnr": float(20 * np.log10(peak / rmse)),
        "conservation": cons,
        "p95_bias_pct": 100 * (wq(pv, 95) / wq(tv, 95) - 1),
        "p99_bias_pct": 100 * (wq(pv, 99) / wq(tv, 99) - 1),
        "wet_freq_ratio": float((pv > wet).mean() / (tv > wet).mean()),
        "subcoarse_power": float(_subcoarse_power(pred) / _subcoarse_power(truth)),
    }
