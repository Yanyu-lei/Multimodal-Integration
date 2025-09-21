# src/plots/plot_f1_fid_vs_rv_depth.py
# =============================================================================
# F1 — Image fidelity: S_fid vs R_v (with depth overlays)
# =============================================================================
# Responsibilities
#   • Scatter S_fid against R_v in [0,1], overlay per-vision-depth traces.
#   • Optional binned mean + 95% bootstrap CI ribbon for readability.
#
# Notes
#   • Reads rows where spoke == "image-fidelity".
#   • Depth comes from the evaluator’s patch-feature layers. 
# =============================================================================

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ._style import apply_rc, apply_style, add_run_hash
from ..analysis.bootstrap import binned_mean_ci  # bootstrap helper 

def _draw(sub: pd.DataFrame, out_dir: Path, results_path: Path, *, ci: bool, nbins: int, filename: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_rc()
    fig, ax = plt.subplots(figsize=(5, 4))
    apply_style(ax, xlim=(0, 1), ylim=(0, 1))

    depths = sorted([int(d) for d in sub["Depth"].dropna().unique()])
    for d in depths:
        sd = sub[sub["Depth"] == d]
        ax.scatter(sd["Rv"], sd["S_fid"], s=8, alpha=0.50, label=f"Depth {d}")

    # Ideal y = x
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    # Optional binned mean + CI
    if ci:
        curve = binned_mean_ci(sub, x="Rv", y="S_fid", nbins=nbins)
        if not curve.empty:
            ax.plot(curve["x_mean"], curve["y_mean"], linewidth=1)
            ax.fill_between(curve["x_mean"], curve["lo"], curve["hi"], alpha=0.20)

    n_pairs = sub["pair_id"].nunique() if "pair_id" in sub.columns else len(sub)
    ax.text(0.98, 0.02, f"N≈{n_pairs}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.set_xlabel(r"$R_v$"); ax.set_ylabel(r"$S_{\mathrm{fid}}$")
    ax.set_title(r"Image fidelity: $S_{\mathrm{fid}}$ vs $R_v$")
    if depths:
        ax.legend(frameon=False, fontsize=8)

    out_path = out_dir / filename
    add_run_hash(fig, results_csv=str(results_path))
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out_path}")

def save_figure(results_csv: str = "results.csv", out_dir: str | None = None,
                filename: str = "f1_fid_vs_rv_depth.pdf", nbins: int = 12,
                ci: bool = False, facet_by: str | None = None, save_metrics: bool = False):
    rc = Path(results_csv).resolve()
    results_path = rc
    out_dir = Path(out_dir) if out_dir else (rc.parent / "figures")
    df = pd.read_csv(results_path)
    base = df[df["spoke"] == "image-fidelity"].copy()
    if base.empty:
        print("No 'image-fidelity' rows in results.csv"); return

    if facet_by and facet_by in base.columns:
        for val, sub in base.groupby(facet_by, observed=True):
            _draw(sub, out_dir / str(val), results_path, ci=ci, nbins=nbins, filename=filename)
    else:
        _draw(base, out_dir, results_path, ci=ci, nbins=nbins, filename=filename)

def main():
    save_figure()

if __name__ == "__main__":
    main()