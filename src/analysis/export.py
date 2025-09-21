# src/analysis/export.py
"""
Export a compact, paper‑friendly metrics table from results.csv.

For each spoke we compute:
- Image‑fidelity: Spearman(S_fid, Rv), mean |S_fid − I_img|.
- Weighting: slope/intercept of Sv on Iv, Spearman(Iv, Sv),
             prevalence of |Δv|>0.1, and mean |Δv|.
- Superadditivity: mean mid‑grid boost and fraction of positive boost.
- Alignment: Spearman(R_joint, S_align), mean |S_align − I_align|.

Optionally facet by 'run_tag' (A/B/C), 'seed', etc.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from .metrics import (
    spearman_ci, slope_intercept_ci, fraction_abs_gt, mean_abs,
    boost_mid_mean, frac_boost_pos
)

def compute_all(results_csv: str, facet: str | None = None, out_dir: str | None = None) -> Path:
    """
    Compute a small, paper-friendly metrics table.
    Writes figures/metrics.csv and returns its path.
    """
    df = pd.read_csv(results_csv)
    out_dir = Path(out_dir) if out_dir else Path(results_csv).resolve().parents[0] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    groups = [("all", df)] if not facet else [(str(v), g) for v, g in df.groupby(facet, observed=True)]

    for tag, g in groups:
        # F1: image-fidelity
        f1 = g[g["spoke"] == "image-fidelity"]
        if not f1.empty:
            rho, lo, hi = spearman_ci(f1["Rv"], f1["S_fid"])
            rows += [{"facet": tag, "spoke": "image-fidelity", "metric": "spearman", "value": rho, "ci_low": lo, "ci_high": hi}]
            rows += [{"facet": tag, "spoke": "image-fidelity", "metric": "mean_abs_gap", "value": mean_abs(f1["S_fid"] - f1["I_img"]), "ci_low": "", "ci_high": ""}]

        # F2a: weighting calibration
        f2a = g[g["spoke"] == "weighting"]
        if not f2a.empty:
            (m, m_lo, m_hi), (b, b_lo, b_hi) = slope_intercept_ci(f2a["I_v"], f2a["S_v"])
            rows += [{"facet": tag, "spoke": "weighting", "metric": "slope", "value": m, "ci_low": m_lo, "ci_high": m_hi}]
            rows += [{"facet": tag, "spoke": "weighting", "metric": "intercept", "value": b, "ci_low": b_lo, "ci_high": b_hi}]
            rho, lo, hi = spearman_ci(f2a["I_v"], f2a["S_v"])
            rows += [{"facet": tag, "spoke": "weighting", "metric": "spearman", "value": rho, "ci_low": lo, "ci_high": hi}]
            rows += [{"facet": tag, "spoke": "weighting", "metric": "prevalence_|Δv|>0.1", "value": fraction_abs_gt(f2a["Delta_v"], 0.1), "ci_low": "", "ci_high": ""}]
            rows += [{"facet": tag, "spoke": "weighting", "metric": "mean_abs_Δv", "value": mean_abs(f2a["Delta_v"]), "ci_low": "", "ci_high": ""}]

        # F3: superadditivity
        f3 = g[g["spoke"] == "superadditivity"]
        if not f3.empty:
            rows += [{"facet": tag, "spoke": "superadditivity", "metric": "mean_boost_mid", "value": boost_mid_mean(f3["Rv"].to_numpy(), f3["Rt"].to_numpy(), f3["Boost"].to_numpy()), "ci_low": "", "ci_high": ""}]
            rows += [{"facet": tag, "spoke": "superadditivity", "metric": "frac_boost_pos", "value": frac_boost_pos(f3["Boost"]), "ci_low": "", "ci_high": ""}]

        # F4: representational alignment
        f4 = g[g["spoke"] == "repr-align"]
        if not f4.empty:
            rho, lo, hi = spearman_ci(f4["R_joint"], f4["S_align"])
            rows += [{"facet": tag, "spoke": "repr-align", "metric": "spearman", "value": rho, "ci_low": lo, "ci_high": hi}]
            rows += [{"facet": tag, "spoke": "repr-align", "metric": "mean_abs_gap", "value": mean_abs(f4["S_align"] - f4["I_align"]), "ci_low": "", "ci_high": ""}]

    out_path = out_dir / "metrics.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path