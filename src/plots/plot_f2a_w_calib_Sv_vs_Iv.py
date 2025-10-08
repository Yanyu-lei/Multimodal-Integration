# src/plots/plot_f2a_w_calib_Sv_vs_Iv.py
# =============================================================================
# F2a — Weighting calibration: S_v vs I_v
# =============================================================================
# Responsibilities
#   • Scatter S_v vs I_v with optional binned mean + 95% CI ribbon.
#   • Annotate a simple slope/intercept fit with bootstrap CIs.
#
# Notes
#   • Reads rows where spoke == "weighting".
#   • slope_intercept_ci and binned_mean_ci come from src/analysis. 
# =============================================================================
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ._style import apply_rc, apply_style, add_run_hash, rasterize_collections
from ..analysis.bootstrap import binned_mean_ci
from ..analysis.metrics   import slope_intercept_ci

def _draw(sub: pd.DataFrame, out_dir: Path, results_path: Path, *, ci: bool, nbins: int, filename: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_rc()
    fig, ax = plt.subplots(figsize=(5, 4))
    apply_style(ax, xlim=(0, 1), ylim=(0, 1))

    ax.scatter(sub["I_v"], sub["S_v"], s=6, alpha=0.25)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)  # ideal

    if ci:
        curve = binned_mean_ci(sub, x="I_v", y="S_v", nbins=nbins)
        if not curve.empty:
            ax.plot(curve["x_mean"], curve["y_mean"], linewidth=1)
            ax.fill_between(curve["x_mean"], curve["lo"], curve["hi"], alpha=0.20)
        (m, m_lo, m_hi), (b, b_lo, b_hi) = slope_intercept_ci(sub["I_v"], sub["S_v"])
        ax.text(0.02, 0.98, f"Sv ≈ {m:.2f}·Iv + {b:.2f}  (95% CI m∈[{m_lo:.2f},{m_hi:.2f}])",
                transform=ax.transAxes, ha="left", va="top", fontsize=8)

    else:
        # mean curve without CIs
        bins = np.linspace(0.0, 1.0, nbins + 1)
        sub["Iv_bin"] = pd.cut(sub["I_v"], bins, include_lowest=True)
        mean_curve = sub.groupby("Iv_bin", observed=False, as_index=False) \
                        .agg(Iv=("I_v", "mean"), Sv=("S_v", "mean")).dropna()
        ax.plot(mean_curve["Iv"], mean_curve["Sv"], linewidth=1)

    n_pairs = sub["pair_id"].nunique() if "pair_id" in sub.columns else len(sub)
    ax.text(0.98, 0.02, f"N≈{n_pairs}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.set_xlabel(r"$I_v=\frac{R_v}{R_v+R_t}$"); ax.set_ylabel(r"$S_v$")
    ax.set_title(r"Weighting calibration: $S_v$ vs $I_v$")

    out_path = out_dir / filename
    rasterize_collections(fig)
    add_run_hash(fig, results_csv=str(results_path))
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved {out_path}")

def save_figure(results_csv: str = "results.csv", out_dir: str | None = None,
                filename: str = "f2a_w_calib_Sv_vs_Iv.pdf", nbins: int = 12,
                ci: bool = False, facet_by: str | None = None, save_metrics: bool = False):
    rc = Path(results_csv).resolve()
    results_path = rc
    out_dir = Path(out_dir) if out_dir else (rc.parent / "figures")

    df = pd.read_csv(results_path)
    base = df[df["spoke"] == "weighting"].copy()
    if base.empty:
        print("No 'weighting' rows in results.csv"); return

    if facet_by and facet_by in base.columns:
        for val, sub in base.groupby(facet_by, observed=True):
            _draw(sub, out_dir / str(val), results_path, ci=ci, nbins=nbins, filename=filename)
    else:
        _draw(base, out_dir, results_path, ci=ci, nbins=nbins, filename=filename)

def main():
    save_figure()

if __name__ == "__main__":
    main()