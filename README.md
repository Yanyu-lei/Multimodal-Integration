# Multimodal-Integration • Test Harness (v0.1)

A lightweight, fully-reproducible framework for **evaluating how well
multimodal AI models integrate information** across five neuroscience-inspired
dimensions (spokes):

1. **Spatial congruence**
2. **Temporal congruence** (proxy via embedding stability)
3. **Modality weighting**
4. **Superadditivity**
5. **Representational alignment**

The end goal is a radar chart that visualises model generality.  We provide:

* Self-contained spoke modules (`src/spatial.py`, …)
* Stimulus builders (`src/stimuli.py`)
* Instrumented model hooks (`src/model_hooks.py`)  
  – capture **all five** spoke signals  
* **21 passing unit tests** covering every spoke + stimulus layer
* Editable Python package (`pip install -e .`) for rapid iteration

---

## Quick-start (CPU-only)

```bash
# 1. Clone the repo
git clone https://github.com/Yanyu-lei/Multimodal-Integration.git
cd Multimodal-Integration

# 2. Create & activate a virtual-env
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate

# 3. Install runtime deps
pip install -r requirements.txt
pip install -e .

# 4. Run all unit tests  (21 should pass)
pytest -q

# 5. Run the evaluator end-to-end
python -m src.integration_evaluator

# 6. (Coming soon) Plot the radar chart
