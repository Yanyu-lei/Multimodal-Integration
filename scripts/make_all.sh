#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# Make-everything wrapper (tests → evaluation → figures+metrics)
# ------------------------------------------------------------------------------
# Usage:
#   scripts/make_all.sh quick [pairs]   # single-seed ABC sweep (default 120)
#   scripts/make_all.sh full  [pairs]   # three seeds × ABC (default 300)
#
# Artifacts:
#   runs/<timestamp>/{results.csv, run.log, pytest.log, figures/..., figures.log}
#   - figures/overall/…          # pooled (no faceting)
#   - figures/by_tag/{A,B,C}/…   # faceted by run_tag
# ==============================================================================

MODE="${1:-quick}"                # quick | full
PAIRS="${2:-120}"                 # number of (image,caption) pairs per condition

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Optional: activate local venv if present
if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

mkdir -p runs
RUN_DIR="runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo "[1/3] Running unit tests…"
pytest -q 2>&1 | tee "$RUN_DIR/pytest.log"

RESULTS="$RUN_DIR/results.csv"

echo "[2/3] Running evaluation…"
if [[ "$MODE" == "quick" ]]; then
  # Single-seed ABC sweep (appends A, then B, then C to one CSV)
  python -m src.integration_evaluator \
    --pairs "$PAIRS" \
    --seed 0 \
    --manifest-seed 0 \
    --offset 0 \
    --batch-all \
    --save "$RESULTS" 2>&1 | tee "$RUN_DIR/run.log"

elif [[ "$MODE" == "full" ]]; then
  # Three seeds × ABC; uses helper that offsets to get disjoint pairs
  # (seeds default to 11051,22103,33259 inside the helper).
  python scripts/run_full_abc3.py \
    --pairs "$PAIRS" \
    --results "$RESULTS" 2>&1 | tee "$RUN_DIR/run.log"
else
  echo "Usage: $0 [quick|full] [pairs]" >&2
  exit 2
fi

echo "[3/3] Making figures + metrics…"
python -m src.plots.make_all_figures \
  --results "$RESULTS" \
  --out-dir "$RUN_DIR/figures" \
  --facet run_tag \
  --save-metrics 2>&1 | tee "$RUN_DIR/figures.log"

echo ""
echo "Artifacts:"
echo "  results → $RESULTS"
echo "  figures → $RUN_DIR/figures/"
echo "  logs    → $RUN_DIR/"