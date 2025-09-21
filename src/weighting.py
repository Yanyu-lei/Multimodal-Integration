# src/weighting.py
# =============================================================================
# Modality weighting (reliability mapping + ideal)
# =============================================================================
# Responsibilities
#   • Map PSNR(dB) to image reliability Rv and CE anchors to text reliability Rt.
#   • Define the classic reliability‑based weighting ideal:
#         I_v = Rv/(Rv+Rt), I_t = Rt/(Rv+Rt)
#   • Provide a small (Rv,Rt) grid helper used in tests/examples.
# =============================================================================

from __future__ import annotations
import numpy as np
from typing import Iterator, Tuple

EPS = 1e-8
PSNR_CAP_DB = 50.0  # 50 dB ↔ Rv ≈ 1.0


def text_reliability(ce_clean: float, ce_corr: float, ce_rand: float) -> float:
    """
    Cross‑entropy → Rt via clean/random anchors:
        Rt = (ce_rand − ce_corr) / (ce_rand − ce_clean), clipped to [0,1].
    """
    denom = max(ce_rand - ce_clean, EPS)
    return float(np.clip((ce_rand - ce_corr) / denom, 0.0, 1.0))


def vision_reliability(psnr_db: float, cap_db: float = PSNR_CAP_DB) -> float:
    """PSNR(dB) → Rv by dividing by the cap and clipping to [0,1]."""
    return float(np.clip(psnr_db / cap_db, 0.0, 1.0))


def compute_I(Rv: float, Rt: float) -> Tuple[float, float]:
    """
    Reliability‑based weighting ideal.
    Falls back to 0.5/0.5 if both reliabilities are zero.
    """
    total = Rv + Rt
    if total < EPS:
        return 0.5, 0.5
    return float(Rv / total), float(Rt / total)


def reliability_grid(
    psnr_levels: np.ndarray = np.linspace(0, PSNR_CAP_DB, 6),
    ce_levels: np.ndarray = np.linspace(0, 1, 6),
) -> Iterator[Tuple[float, float]]:
    """
    Yield a small grid of (Rv, Rt) pairs for demos/tests.

    We treat psnr_levels as the image gauge (mapped to Rv) and ce_levels as the
    text gauge (0→clean→Rt=1; 1→random→Rt=0).
    """
    for psnr in psnr_levels:
        Rv = vision_reliability(float(psnr))
        for ce_norm in ce_levels:
            Rt = 1.0 - float(ce_norm)
            yield Rv, Rt