# src/plots/plot_f4_align_vs_Rjoint.py
# =============================================================================
# F4 — Representational alignment: S_align vs R_joint
# =============================================================================
# Responsibilities
#   • Scatter S_align vs R_joint with optional binned mean + 95% CI ribbon.
#
# Notes
#   • Reads rows where spoke == "repr-align".
#   • R_joint = min(R_v, R_t); the ideal is y = x (plotted as dashed line). 
# =============================================================================

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ._style import apply_rc, apply_style, add_run_hash
from ..analysis.bootstrap import binned_mean_ci

def _draw(sub: pd.DataFrame, out_dir: Path, results_path: Path, *, ci: bool, nbins: int, filename: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_rc()
    fig, ax = plt.subplots(figsize=(5, 4))
    apply_style(ax, xlim=(0, 1), ylim=(0, 1))

    ax.scatter(sub["R_joint"], sub["S_align"], s=6, alpha=0.25)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)  # ideal

    if ci:
        curve = binned_mean_ci(sub, x="R_joint", y="S_align", nbins=nbins)
        if not curve.empty:
            ax.plot(curve["x_mean"], curve["y_mean"], linewidth=1)
            ax.fill_between(curve["x_mean"], curve["lo"], curve["hi"], alpha=0.20)
    else:
        bins = np.linspace(0.0, 1.0, nbins + 1)
        sub["Rj_bin"] = pd.cut(sub["R_joint"], bins, include_lowest=True)
        mean_curve = sub.groupby("Rj_bin", observed=False, as_index=False) \
                        .agg(Rj=("R_joint", "mean"), Sa=("S_align", "mean")).dropna()
        ax.plot(mean_curve["Rj"], mean_curve["Sa"], linewidth=1)

    n_pairs = sub["pair_id"].nunique() if "pair_id" in sub.columns else len(sub)
    ax.text(0.98, 0.02, f"N≈{n_pairs}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.set_xlabel(r"$R_{\mathrm{joint}}=\min(R_v,R_t)$"); ax.set_ylabel(r"$S_{\mathrm{align}}$")
    ax.set_title(r"Representational alignment: $S_{\mathrm{align}}$ vs $R_{\mathrm{joint}}$")

    out_path = out_dir / filename
    add_run_hash(fig, results_csv=str(results_path))
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out_path}")

def save_figure(results_csv: str = "results.csv", out_dir: str | None = None,
                filename: str = "f4_align_vs_Rjoint.pdf", nbins: int = 12,
                ci: bool = False, facet_by: str | None = None, save_metrics: bool = False):
    rc = Path(results_csv).resolve()
    results_path = rc
    out_dir = Path(out_dir) if out_dir else (rc.parent / "figures")

    df = pd.read_csv(results_path)
    base = df[df["spoke"] == "repr-align"].copy()
    if base.empty:
        print("No 'repr-align' rows in results.csv"); return

    if facet_by and facet_by in base.columns:
        for val, sub in base.groupby(facet_by, observed=True):
            _draw(sub, out_dir / str(val), results_path, ci=ci, nbins=nbins, filename=filename)
    else:
        _draw(base, out_dir, results_path, ci=ci, nbins=nbins, filename=filename)

def main():
    save_figure()

if __name__ == "__main__":
    main()