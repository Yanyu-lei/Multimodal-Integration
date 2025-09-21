# src/image_fidelity.py
# =============================================================================
# Image fidelity utilities
# =============================================================================
# Responsibilities
#   • Provide a sweep of Gaussian-noise σ values for spoke A (image fidelity).
#   • Compute PSNR (dB) between clean and corrupted images.
#   • Define the ideal mapping I_img(Rv) used in image fidelity (identity).
#   • Convert PSNR(dB) ↔ Gaussian σ for the gaussian mechanism.
#
# Notes
#   • All mappings are bounded and tested by the repo’s unit tests.
#   • Downstream code consumes Rv∈[0,1]; we do not persist PSNR in CSV. :contentReference[oaicite:1]{index=1}
# =============================================================================

from __future__ import annotations
import numpy as np

MAX_PIXEL_VALUE = 255.0  # 8‑bit images


def noise_sigma_sweep(sigmas: list[float] | None = None, max_sigma: float = 255.0) -> tuple[list[float], float]:
    """
    Return (σ_list, max_sigma) used by the image‑fidelity loop.

    Default: 21 evenly spaced σ levels from 0 → max_sigma.
    For gaussian noise we normally set max_sigma=255; for blur/cutout we pass 50.
    """
    if sigmas is None:
        sigmas = list(np.linspace(0.0, max_sigma, num=21))
    return sigmas, max_sigma


def compute_psnr(orig: np.ndarray, pert: np.ndarray, max_val: float = MAX_PIXEL_VALUE) -> float:
    """
    Peak Signal‑to‑Noise Ratio (dB) between two uint8 image arrays.

    If arrays are identical, returns +inf.
    """
    mse = np.mean((orig.astype(np.float64) - pert.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(max_val) - 10.0 * np.log10(mse))


def compute_I(R: float) -> float:
    """
    Ideal for image fidelity: identity I_img(R) = R.
    """
    return float(R)


def psnr_to_sigma(psnr_db: float, max_val: float = MAX_PIXEL_VALUE) -> float:
    """
    Convert PSNR (dB) to Gaussian noise σ (pixel units).

    This is used only for the gaussian disturbance to back out the σ that
    would (approximately) yield a given PSNR level on natural images. :contentReference[oaicite:2]{index=2}
    """
    return float(max_val / (10 ** (psnr_db / 20.0)))