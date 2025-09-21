# tests/test_image_fidelity.py
"""
Tests for the Image‑Fidelity spoke.

What we verify
--------------
1) PSNR behaves correctly (identical images → ∞; more noise → lower PSNR).
2) The ideal for this spoke is identity: I_img(Rv) = Rv.
3) Helper that adds Gaussian noise preserves size/dtype.

These checks guard the mapping from the *gauge* (PSNR→Rv) and the ideal.
"""
import numpy as np
import pytest
from PIL import Image

from src.image_fidelity import compute_psnr, compute_I, noise_sigma_sweep
from src.stimuli import add_gaussian_noise


def test_psnr_and_I():
    # Simple 16×16 gray image
    img = Image.new("RGB", (16, 16), color=(128, 128, 128))
    orig = np.array(img)

    # Sweep σ = [0, 10], cap at 50 dB
    sigmas, max_sigma = noise_sigma_sweep(sigmas=[0, 10], max_sigma=50.0)

    # σ = 0 → PSNR = ∞ → Rv = 1.0
    arr0 = add_gaussian_noise(img, sigma=0)
    psnr0 = compute_psnr(orig, np.array(arr0))
    assert psnr0 == float("inf") or psnr0 > max_sigma
    R0 = min(psnr0 / max_sigma, 1.0)
    assert R0 == pytest.approx(1.0)
    assert compute_I(R0) == pytest.approx(R0)

    # σ = 10 → some noise → PSNR lower than clean
    arr10 = add_gaussian_noise(img, sigma=10)
    psnr10 = compute_psnr(orig, np.array(arr10))
    assert psnr10 < psnr0
    R10 = min(psnr10 / max_sigma, 1.0)
    assert 0.0 <= R10 < R0
    assert compute_I(R10) == pytest.approx(R10)


def test_add_gaussian_noise_preserves_size():
    img = Image.new("RGB", (8, 8), "white")
    arr_noise = add_gaussian_noise(img, sigma=5)
    assert isinstance(arr_noise, np.ndarray)
    assert arr_noise.shape == (8, 8, 3)