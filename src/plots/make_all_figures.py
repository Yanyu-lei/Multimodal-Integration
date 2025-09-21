# src/plots/make_all_figures.py
"""
Make all paper figures (F1–F4) from a single results.csv.

Outputs
-------
- <base>/overall/*.pdf (+ metrics.csv when --save-metrics)
- <base>/by_tag/<facet value>/*.pdf (+ metrics.csv)  [only if --facet is set]

Notes
-----
- 'base' is given by --out-dir (or inferred from --results if omitted).
- Faceting supports: run_tag (A/B/C), vision_mode, text_mode, seed
- Confidence intervals (95% bootstrap) are ON by default; use --no-ci to disable.
"""

from __future__ import annotations
import argparse
from pathlib import Path

from .plot_f1_fid_vs_rv_depth   import save_figure as f1
from .plot_f2a_w_calib_Sv_vs_Iv import save_figure as f2a
from .plot_f2b_w_heat_DeltaV    import save_figure as f2b
from .plot_f3_sup_heat_boost    import save_figure as f3
from .plot_f4_align_vs_Rjoint   import save_figure as f4
from ..analysis.export import compute_all  # writes a metrics.csv next to PDFs

def _save_set(results_csv: str, out_dir: Path, *, facet_by: str | None, ci: bool):
    """Call each figure writer once for this (out_dir, facet_by) tuple."""
    f1(results_csv=results_csv, out_dir=str(out_dir), facet_by=facet_by, ci=ci)
    f2a(results_csv=results_csv, out_dir=str(out_dir), facet_by=facet_by, ci=ci)
    f2b(results_csv=results_csv, out_dir=str(out_dir), facet_by=facet_by, ci=ci)
    f3(results_csv=results_csv, out_dir=str(out_dir), facet_by=facet_by, ci=ci)
    f4(results_csv=results_csv, out_dir=str(out_dir), facet_by=facet_by, ci=ci)

def main():
    ap = argparse.ArgumentParser(description="Build F1–F4 PDFs (and optional metrics.csv) from results.csv")
    ap.add_argument("--results", default="results.csv", help="Path to results.csv")

    # CI defaults to ON; reviewers get statistically annotated plots without extra flags.
    ci = ap.add_mutually_exclusive_group()
    ci.add_argument("--ci",    dest="ci", action="store_true",
                    help="Enable 95% bootstrap CIs (default).")
    ci.add_argument("--no-ci", dest="ci", action="store_false",
                    help="Disable CIs for faster plotting.")
    ap.set_defaults(ci=True)

    ap.add_argument(
        "--facet",
        default=None,
        choices=["run_tag", "vision_mode", "text_mode", "seed"],
        help="Split figures/metrics by this column (e.g., run_tag=A/B/C, or seed)",
    )
    ap.add_argument("--out-dir", default=None,
                    help="Base output dir. Default: <results_dir>/figures")
    ap.add_argument("--save-metrics", action="store_true",
                    help="Also write metrics.csv next to the PDFs")
    args = ap.parse_args()

    results_path = Path(args.results).resolve()
    base = Path(args.out_dir).resolve() if args.out_dir else results_path.parent / "figures"

    # 1) Overall (no faceting) → <base>/overall
    overall_dir = base / "overall"
    _save_set(str(results_path), overall_dir, facet_by=None, ci=args.ci)
    if args.save_metrics:
        out = compute_all(str(results_path), facet=None, out_dir=str(overall_dir))
        print(f"Saved metrics → {out}")

    # 2) Optional per‑facet (e.g., A/B/C) → <base>/by_tag/<value>
    if args.facet:
        by_tag_dir = base / "by_tag"
        _save_set(str(results_path), by_tag_dir, facet_by=args.facet, ci=args.ci)
        if args.save_metrics:
            out = compute_all(str(results_path), facet=args.facet, out_dir=str(by_tag_dir))
            print(f"Saved metrics → {out}")

if __name__ == "__main__":
    main()