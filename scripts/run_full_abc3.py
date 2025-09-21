#!/usr/bin/env python3
"""
Full ABC (multi‑seed) runner
============================

Purpose
-------
Run the three preset A/B/C conditions over *three* deterministic seeds and
append all rows to a single `results.csv`. Each seed uses a disjoint slice of
the manifest by advancing an offset, so the 3×(pairs) samples do not overlap.

A/B/C mapping (kept consistent with the paper and the shell wrapper):
  A → vision=gaussian, text=mask
  B → vision=blur,     text=shuffle
  C → vision=cutout,   text=replace

Typical usage
-------------
# Default: 3 seeds × 300 pairs/seed
python scripts/run_full_abc3.py

# Custom pairs/seed, custom seeds, explicit results path
python scripts/run_full_abc3.py \
  --pairs 400 \
  --seeds 101,102,103 \
  --results runs/$(date +%Y%m%d_%H%M%S)/results.csv

CLI arguments
-------------
--pairs           int   Number of (image,caption) pairs per seed per tag (default 300)
--results         str   CSV to append to (created if missing; header kept stable)
--seeds           str   Three comma‑separated integers (default "11051,22103,33259")
--manifest-seed   int   Seed for the manifest replay/shuffle (keeps offsets disjoint)

Outputs
-------
• Appends rows to the given CSV (or creates it).
• Prints the exact `python -m src.integration_evaluator ...` commands it runs.
• Returns a non‑zero exit status if any sub‑process fails.

Reproducibility notes
---------------------
Keeping `--manifest-seed` fixed ensures the per‑seed offsets slice disjoint
items from the same manifest ordering, so different machines get the same
triplets of datasets when they use the same seeds and pairs.

This script is invoked by `scripts/make_all.sh full [pairs]`.  See that wrapper
for the end‑to‑end “tests → run → figures” pipeline.  (No behavior changes here;
this file only gains documentation.) 
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Parse args, iterate three seeds × A/B/C, and append to one CSV."""
    ap = argparse.ArgumentParser(prog="run_full_abc3.py")
    ap.add_argument("--pairs", type=int, default=300,
                    help="pairs per seed (per A/B/C tag)")
    ap.add_argument("--results", type=str, default="results.csv",
                    help="CSV file to append results into")
    ap.add_argument(
        "--seeds",
        type=str,
        default="11051,22103,33259",
        help="comma-separated seeds for the three datasets (e.g., 101,102,103)",
    )
    ap.add_argument(
        "--manifest-seed",
        type=int,
        default=0,
        help="keeps the manifest order fixed so offsets are disjoint",
    )
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    combos = [
        ("A", "gaussian", "mask"),
        ("B", "blur",     "shuffle"),
        ("C", "cutout",   "replace"),
    ]

    for set_idx, seed in enumerate(seeds):
        offset = set_idx * args.pairs  # disjoint slices per seed
        for tag, vmode, tmode in combos:
            cmd = [
                sys.executable, "-m", "src.integration_evaluator",
                "--vision-mode",   vmode,
                "--text-mode",     tmode,
                "--seed",          str(seed),
                "--manifest-seed", str(args.manifest_seed),
                "--pairs",         str(args.pairs),
                "--offset",        str(offset),
                "--run-tag",       tag,
                "--save",          args.results,
            ]
            print("\n[full] " + " ".join(cmd))
            subprocess.run(cmd, check=True)

    print(f"\n✅ All runs complete. Appended to {Path(args.results).resolve()}")


if __name__ == "__main__":
    main()