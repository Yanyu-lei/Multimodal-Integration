# tests/test_modality_weighting.py
"""
Tests for the Modality‑Weighting spoke.

We check:
• Vision reliability mapping: PSNR/50 → [0, 1].
• Text reliability mapping: linear rescale from CE (clean ↔ random).
• Ideal normalisation: I_v + I_t = 1; both zero → 0.5/0.5.
• Default grid shape and bounds.
"""
import pytest
from src.weighting import (
    vision_reliability,
    text_reliability,
    compute_I,
    reliability_grid,
)


def test_vision_reliability_bounds():
    assert vision_reliability(0.0) == 0.0
    assert vision_reliability(25.0) == 0.5
    assert vision_reliability(50.0) == 1.0
    assert vision_reliability(60.0) == 1.0  # clip above cap


def test_text_reliability_bounds():
    ce_clean, ce_rand = 2.0, 6.0
    assert text_reliability(ce_clean, ce_clean, ce_rand) == 1.0      # clean
    ce_mid = (ce_clean + ce_rand) / 2
    assert text_reliability(ce_clean, ce_mid, ce_rand) == pytest.approx(0.5, 1e-6)
    assert text_reliability(ce_clean, ce_rand, ce_rand) == 0.0       # random


def test_compute_I_normalises():
    Iv, It = compute_I(0.8, 0.2)
    assert pytest.approx(Iv + It, abs=1e-6) == 1.0
    assert Iv > It
    Iv2, It2 = compute_I(0.0, 0.0)
    assert Iv2 == It2 == 0.5


def test_reliability_grid_shape_and_bounds():
    grid = list(reliability_grid())
    assert len(grid) == 6 * 6  # default 6×6
    for Rv, Rt in grid:
        assert 0.0 <= Rv <= 1.0 and 0.0 <= Rt <= 1.0