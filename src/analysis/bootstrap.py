# src/analysis/bootstrap.py
"""
Bootstrap helpers for curves and heatmaps.

- binned_mean_ci: x∈[0,1] → mean(y) per bin + 95% bootstrap CI.
- grid_cell_ci:  2D x,y bins → mean(val), count, edges, and a boolean mask
                 marking cells whose bootstrap CI excludes 0 (significance).

These are used by plotting code (F1–F4).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def binned_mean_ci(df: pd.DataFrame, *, x: str, y: str,
                   nbins: int = 12, n_boot: int = 800,
                   ci: float = 0.95, min_count: int = 3) -> pd.DataFrame:
    """
    Bin by x∈[0,1], compute mean(y) per bin + bootstrap CI.
    Returns columns: x_mean, y_mean, lo, hi, n.
    """
    edges = np.linspace(0.0, 1.0, nbins + 1)
    tmp = df[[x, y]].dropna().copy()
    tmp["bin"] = pd.cut(tmp[x], edges, include_lowest=True)
    grp = tmp.groupby("bin", observed=True)

    alpha = 1.0 - ci
    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    rows = []
    for _, g in grp:
        if len(g) < min_count:
            continue
        arr = g[y].to_numpy()
        n = len(arr)
        boots = []
        for _ in range(n_boot):
            idx = np.random.randint(0, n, n)
            boots.append(arr[idx].mean())
        rows.append({
            "x_mean": float(g[x].mean()),
            "y_mean": float(arr.mean()),
            "lo": float(np.percentile(boots, lo_q)),
            "hi": float(np.percentile(boots, hi_q)),
            "n": int(n),
        })
    return pd.DataFrame(rows)

def grid_cell_ci(df: pd.DataFrame, *, x: str, y: str, val: str,
                 bins: int = 10, n_boot: int = 400,
                 min_count: int = 3, ci: float = 0.95):
    """
    For heatmaps: bin by x,y; return (mean, count, edges, significant_mask)
    where 'significant' means the bootstrap CI of mean(val) excludes 0.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    tmp = df[[x, y, val]].dropna().copy()
    tmp["xb"] = pd.cut(tmp[x], edges, include_lowest=True)
    tmp["yb"] = pd.cut(tmp[y], edges, include_lowest=True)

    mean = tmp.pivot_table(index="yb", columns="xb", values=val, aggfunc="mean",
                        observed=False).sort_index().sort_index(axis=1)
    count = tmp.pivot_table(index="yb", columns="xb", values=val, aggfunc="count",
                            observed=False).reindex_like(mean)

    sig = np.zeros_like(mean.values, dtype=bool)
    alpha = 1.0 - ci
    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)

    # Pre-index groups per cell to avoid repeated filtering
    for yb, xb in np.ndindex(mean.shape):
        c = count.values[yb, xb]
        if np.isnan(mean.values[yb, xb]) or c < min_count:
            continue
        # Build the category keys we need to match
        y_key = mean.index[yb]
        x_key = mean.columns[xb]
        gvals = tmp.loc[(tmp["yb"] == y_key) & (tmp["xb"] == x_key), val].to_numpy()
        n = len(gvals)
        boots = []
        for _ in range(n_boot):
            idx = np.random.randint(0, n, n)
            boots.append(gvals[idx].mean())
        lo = np.percentile(boots, lo_q)
        hi = np.percentile(boots, hi_q)
        if lo > 0.0 or hi < 0.0:
            sig[yb, xb] = True

    return mean, count, edges, sig