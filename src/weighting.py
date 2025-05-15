# src/weighting.py

import numpy as np
from typing import Iterator, Tuple

def vision_reliability(psnr: float, max_psnr: float = 50.0) -> float:
    """
    Convert a PSNR value (in dB) into a reliability in [0,1].
    We assume linear mapping: R = min(psnr / max_psnr, 1.0).
    """
    return float(np.clip(psnr / max_psnr, 0.0, 1.0))

def text_reliability(entropy: float) -> float:
    """
    Convert a text‐shuffle “entropy” in [0,1] into reliability R_text = 1 - entropy.
    """
    return float(np.clip(1.0 - entropy, 0.0, 1.0))

def compute_I(Rv: float, Rt: float) -> Tuple[float, float]:
    """
    Bayesian ideal weights:
      I_v = Rv / (Rv + Rt)
      I_t = Rt / (Rv + Rt)
    If both Rv=Rt=0, split evenly.
    """
    total = Rv + Rt
    if total <= 0:
        return 0.5, 0.5
    return Rv / total, Rt / total

def psnr_to_sigma(psnr: float, max_pixel: float = 255.0) -> float:
    """
    Convert PSNR (dB) → Gaussian‐noise σ via
      σ = max_pixel / sqrt(10^(PSNR/10)).
    Higher PSNR → lower σ.
    """
    return float(max_pixel / np.sqrt(10 ** (psnr / 10)))

def noise_sweep(
    psnr_levels: np.ndarray = np.linspace(0, 50, 6),
    entropy_levels: np.ndarray = np.linspace(0, 1, 6)
) -> Iterator[Tuple[float, float]]:
    """
    Yield all combinations of (Rv, Rt) from the given PSNR & entropy grids.
    """
    for psnr in psnr_levels:
        Rv = vision_reliability(psnr)
        for ent in entropy_levels:
            Rt = text_reliability(ent)
            yield Rv, Rt
