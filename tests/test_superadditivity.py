# tests/test_superadditivity.py
"""
Tests for the Superadditivity (inverse‑effectiveness) spoke.

We check:
• Ideal I_joint = Rv + Rt − Rv*Rt is within bounds and matches spot values.
• The “boost” over a simple base is largest mid‑grid (inverse effectiveness).
"""
import numpy as np
from src.superadditivity import compute_I


def test_bounds():
    rng = np.linspace(0, 1, 11)
    eps = 1e-9
    for Rv in rng:
        for Rt in rng:
            I = compute_I(Rv, Rt)
            assert max(Rv, Rt) - eps <= I <= 1.0 + eps


def test_weighted_sum_formula():
    assert np.isclose(compute_I(0.0, 0.0), 0.00)
    assert np.isclose(compute_I(0.5, 0.0), 0.50)
    assert np.isclose(compute_I(0.0, 0.5), 0.50)
    assert np.isclose(compute_I(0.3, 0.4), 0.58)   # 0.3 + 0.4 − 0.12
    assert np.isclose(compute_I(1.0, 0.6), 1.00)   # clamp at 1
    assert np.isclose(compute_I(0.9, 0.9), 0.99)   # 0.9 + 0.9 − 0.81


def test_inverse_effectiveness_peak():
    low = compute_I(0.2, 0.2) - 0.2
    mid = compute_I(0.5, 0.5) - 0.5
    high = compute_I(0.8, 0.8) - 0.8
    assert mid >= low and mid >= high
