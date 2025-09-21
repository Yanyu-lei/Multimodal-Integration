# src/plots/plot_f3_sup_heat_boost.py
# =============================================================================
# F3 — Superadditivity heatmap: Boost = S_joint − I_joint across (R_v, R_t)
# =============================================================================
# Responsibilities
#   • 2D grid over R_v × R_t showing mean Boost per cell.
#   • Optional significance markers where the bootstrap CI excludes 0.
#
# Notes
#   • Reads rows where spoke == "superadditivity".
#   • See bootstrap.grid_cell_ci for binning/CIs; NaN cells greyed out. 
# =============================================================================

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ._style import apply_rc, apply_style, add_run_hash, cmap_with_nan
from ..analysis.bootstrap import grid_cell_ci

def _draw(sub: pd.DataFrame, out_dir: Path, results_path: Path, *,
          bins: int, vmin: float, vmax: float, min_count: int, filename: str, ci: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_rc()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    apply_style(ax, grid=False, xlim=(0, 1), ylim=(0, 1))
    ax.set_xticks(np.linspace(0, 1, 6)); ax.set_yticks(np.linspace(0, 1, 6))

    mean, count, edges, sig = grid_cell_ci(sub, x="Rv", y="Rt", val="Boost",
                                           bins=bins, n_boot=400, min_count=min_count)
    Z = mean.values.astype(float).copy()
    Z[count.values < min_count] = np.nan

    im = ax.imshow(Z, origin="lower", aspect="auto", extent=[0, 1, 0, 1],
                   vmin=vmin, vmax=vmax, cmap=cmap_with_nan("#e0e0e0"))

    if ci and sig.any():
        centers = (edges[:-1] + edges[1:]) / 2.0
        y_idx, x_idx = np.where(sig)
        ax.scatter(centers[x_idx], centers[y_idx], s=12, linewidths=0.8, facecolors="none")

    n_pairs = sub["pair_id"].nunique() if "pair_id" in sub.columns else len(sub)
    ax.text(0.98, 0.02, f"N≈{n_pairs}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.set_xlabel(r"$R_v$"); ax.set_ylabel(r"$R_t$")
    ax.set_title(r"Superadditivity: Boost $= S_{\mathrm{joint}} - I_{\mathrm{joint}}$")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85); cbar.set_label("Boost")

    out_path = out_dir / filename
    add_run_hash(fig, results_csv=str(results_path))
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out_path}")

def save_figure(results_csv: str = "results.csv", out_dir: str | None = None,
                filename: str = "f3_sup_heat_boost.pdf", bins: int = 10, vmin: float = -0.5, vmax: float = 0.5,
                min_count: int = 3, ci: bool = False, facet_by: str | None = None, save_metrics: bool = False):
    rc = Path(results_csv).resolve()
    results_path = rc
    out_dir = Path(out_dir) if out_dir else (rc.parent / "figures")

    df = pd.read_csv(results_path)
    base = df[df["spoke"] == "superadditivity"].copy()
    if base.empty:
        print("No 'superadditivity' rows in results.csv"); return

    if facet_by and facet_by in base.columns:
        for val, sub in base.groupby(facet_by, observed=True):
            _draw(sub, out_dir / str(val), results_path, bins=bins, vmin=vmin, vmax=vmax,
                  min_count=min_count, filename=filename, ci=ci)
    else:
        _draw(base, out_dir, results_path, bins=bins, vmin=vmin, vmax=vmax,
              min_count=min_count, filename=filename, ci=ci)

def main():
    save_figure()

if __name__ == "__main__":
    main()