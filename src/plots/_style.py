# src/plots/_style.py
# =============================================================================
# Plot style helpers (shared by all F1–F4 figures)
# =============================================================================
# Responsibilities
#   • Provide a consistent rc style across figures.
#   • Small helpers: tidy axes, add a SHA-256 footer of results.csv for provenance,
#     and a NaN-aware colormap for heatmaps.
#
# Notes
#   • These helpers are intentionally small so figure modules stay readable.
#   • No external styling packages; standard Matplotlib only. 
# =============================================================================

from pathlib import Path
import hashlib
import matplotlib.pyplot as plt

RC = {
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":   8,
}

def apply_rc() -> None:
    for k, v in RC.items():
        plt.rcParams[k] = v

def apply_style(ax, *, grid: bool = True, xlim=(0, 1), ylim=(0, 1)) -> None:
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.tick_params(direction="out", length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(True, alpha=0.20, linewidth=0.8)
        ax.set_axisbelow(True)

def add_run_hash(fig, *, results_csv: str = "results.csv", fontsize: int = 7) -> None:
    """
    Footer with sha256(results.csv) to make figures provenance-aware.
    """
    path = Path(results_csv)
    if not path.is_absolute():
        # default relative to the project root when called from modules
        path = Path(__file__).resolve().parents[2] / path
    try:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        fig.text(0.99, 0.01, f"sha256:{sha}", ha="right", va="bottom", fontsize=fontsize)
    except FileNotFoundError:
        pass

def cmap_with_nan(color: str = "#cccccc"):
    """
    Return a Viridis copy that renders NaNs in a light grey (for empty bins). 
    """
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color)
    return cmap