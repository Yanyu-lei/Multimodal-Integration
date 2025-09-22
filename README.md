[![CI](https://github.com/Yanyu-lei/Multimodal-Integration/actions/workflows/ci.yml/badge.svg)](https://github.com/Yanyu-lei/Multimodal-Integration/actions/workflows/ci.yml)

# Multimodal-Integration

Evaluate **how a multimodal model (CLIP)** combines image + text by turning four neuroscience‑inspired behaviors into **measurable spokes**:

1. **Image fidelity (A)** — how much of the image signal remains as we corrupt it.  
2. **Modality weighting (B)** — do internal weights shift toward the more reliable cue?  
3. **Superadditivity (C, inverse effectiveness)** — is the joint response more/less than expected from the parts?  
4. **Representational alignment (D)** — do *matched* image–text pairs stay more similar than mismatched ones as cues weaken?

All spokes share a common reliability scale `R ∈ [0,1]`:  
- **Vision reliability `Rv`** is measured from **PSNR** of the disturbed image (PSNR/50, clipped).  
- **Text reliability `Rt`** is measured from **GPT‑2 cross‑entropy**, rescaled between clean and random.  
The “ideals” and the concrete measurements live in `src/*.py`.

---

## Repository map

- **Orchestrator / CLI** — `src/integration_evaluator.py`  
  Runs the full pipeline end‑to‑end, writes one tidy CSV, supports A/B/C presets, seeds, offsets, and a manifest builder. (CSV column set is fixed.)

- **Stimuli + reliability** — `src/stimuli.py`  
  Deterministic COCO streaming with a **manifest.json** for exact replay; image corruption (gaussian/blur/cutout) and **measured** `Rv` via PSNR; text corruption (mask/shuffle/replace) and `Rt` via CE mapping.

- **Spokes (math + measurement)**  
  - Image fidelity: `src/image_fidelity.py` (PSNR + identity ideal)  
  - Weighting: `src/weighting.py` (Iv,It from Rv,Rt)  
  - Superadditivity: `src/superadditivity.py` (I = Rv + Rt − Rv·Rt)  
  - Alignment: `src/repr_align.py` (I_align = min(Rv,Rt))

- **Model interface** — `src/model_hooks.py`  
  Loads CLIP ViT‑B/32, returns: weighting (pre‑norm projection norms), multi‑depth patch grids (image fidelity), pooled embeddings (joint/align).

- **Figures + metrics** — `src/plots/*.py`, orchestrated by `src/plots/make_all_figures.py`  
  Produces F1–F4 PDFs and `metrics.csv` (slope/intercept/Spearman/boost prevalence), with optional faceting by run tag or seed.

- **Runner scripts** — `scripts/make_all.sh`, `scripts/run_full_abc3.py`  
  One‑liner wrapper for tests → runs → figures; multi‑seed helper. Saved artifacts go under `runs/<timestamp>/`.

- **Tests** — `tests/*.py`  
  Cover stimuli invariants and monotonicity, PSNR/identity, weighting sum‑to‑one, alignment identities, hook outputs.

---

## Quick start

> Works on CPU; GPU just accelerates CLIP/GPT‑2.

```bash
# 1) Create and activate a virtual environment (reviewers make their own)
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate

# 2) Install PyTorch for your system (see pytorch.org for CUDA wheels)
pip install "torch>=2.2,<3" "torchvision>=0.17,<1"

# 3) Install the rest (and the package itself)
pip install -r requirements.txt
pip install -e .

# 4) One-liner (quick): tests → ABC (single seed) → F1–F4 + metrics
scripts/make_all.sh quick 120

# 5) Full ABC (3 seeds): runs → appends to one CSV → per-tag + overall figures
scripts/make_all.sh full 300
```

### Reproducibility at a glance
- **Determinism.** We fix seeds (11051, 22103, 33259) and replay the exact COCO bytes/order via `manifest.json`.  
- **Rebuild-only path.** All PDFs + `metrics.csv` can be regenerated from a single `results.csv` (no model rerun).  
- **Provenance.** Each PDF footer embeds `sha256(results.csv)` so anyone can verify it came from the same CSV.


## Rebuild only the figures from an existing CSV
You can also pass `results.csv.gz` directly; pandas handles gzip transparently. Tested on Python 3.10.

**Reproducible artifact:** [v0.1.0 — Reproducible Figures & CSV](https://github.com/Yanyu-lei/Multimodal-Integration/releases/tag/v0.1.0)

```bash
python -m src.plots.make_all_figures \
  --results runs/<timestamp>/results.csv \
  --out-dir runs/<timestamp>/figures \
  --facet run_tag \
  --save-metrics
```

**Note:** 95% bootstrap CIs are on by default; add `--no-ci` to disable for speed.
