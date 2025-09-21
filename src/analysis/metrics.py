# src/analysis/metrics.py
"""
Metrics used by export.py and figures.

- Spearman rank corr with bootstrap CIs
- Slope/intercept with bootstrap CIs
- Utility scalars: fraction |x|>τ, mean |x|
- Superadditivity summaries: mean mid‑grid boost, fraction of positive boost
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    if xr.size == 0:
        return float("nan")
    xr = (xr - xr.mean()) / (xr.std(ddof=0) + 1e-12)
    yr = (yr - yr.mean()) / (yr.std(ddof=0) + 1e-12)
    return float(np.clip(np.mean(xr * yr), -1.0, 1.0))

def spearman_ci(x, y, n_boot: int = 800, ci: float = 0.95):
    x = np.asarray(x); y = np.asarray(y)
    base = _spearman(x, y)
    n = len(x)
    alpha = 1.0 - ci
    qs = (100 * (alpha / 2), 100 * (1 - alpha / 2))
    boots = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        boots.append(_spearman(x[idx], y[idx]))
    lo, hi = np.percentile(boots, qs)
    return base, float(lo), float(hi)

def slope_intercept_ci(x, y, n_boot: int = 800, ci: float = 0.95):
    x = np.asarray(x); y = np.asarray(y)
    m, b = np.polyfit(x, y, 1)
    n = len(x)
    alpha = 1.0 - ci
    qs = (100 * (alpha / 2), 100 * (1 - alpha / 2))
    m_boot, b_boot = [], []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        mb, bb = np.polyfit(x[idx], y[idx], 1)
        m_boot.append(mb); b_boot.append(bb)
    m_lo, m_hi = np.percentile(m_boot, qs)
    b_lo, b_hi = np.percentile(b_boot, qs)
    return (float(m), float(m_lo), float(m_hi)), (float(b), float(b_lo), float(b_hi))

def fraction_abs_gt(vals: np.ndarray, tau: float = 0.1):
    vals = np.asarray(vals)
    if vals.size == 0:
        return float("nan")
    return float(np.mean(np.abs(vals) > tau))

def mean_abs(vals: np.ndarray):
    vals = np.asarray(vals)
    return float(np.mean(np.abs(vals))) if vals.size else float("nan")

def boost_mid_mean(rv: np.ndarray, rt: np.ndarray, boost: np.ndarray, rmin=0.3, rmax=0.7):
    mask = (np.minimum(rv, rt) >= rmin) & (np.minimum(rv, rt) <= rmax)
    sub = boost[mask]
    return float(np.mean(sub)) if sub.size else float("nan")

def frac_boost_pos(boost: np.ndarray):
    boost = np.asarray(boost)
    return float(np.mean(boost > 0.0)) if boost.size else float("nan")