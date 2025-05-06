# tests/test_superadditivity.py

import pytest
from src.superadditivity import (
    compute_R,
    compute_I,
    SuperadditivityPerturbation,
    DEFAULT_K,
    EPS,
)

def test_superadd_reliability_min():
    cases = [
        (0.8, 0.5, 0.5),
        (0.3, 0.7, 0.3),
        (0.0, 1.0, 0.0),
    ]
    for R_v, R_t, expR in cases:
        assert compute_R(R_v, R_t) == pytest.approx(expR)

def test_compute_I_values():
    cases = [
        (0.8, 0.5),
        (0.3, 0.7),
        (0.2, 0.2),
    ]
    for R_v, R_t in cases:
        R = compute_R(R_v, R_t)
        expected_I = DEFAULT_K / (R + EPS)
        assert compute_I(R_v, R_t) == pytest.approx(expected_I)

def test_perturbation_outputs_R():
    sample = {"image": None, "text": None}
    cases = [
        (0.8, 0.5, 0.5),
        (0.3, 0.7, 0.3),
        (0.0, 1.0, 0.0),
    ]
    for R_v, R_t, expR in cases:
        pert = SuperadditivityPerturbation([(R_v, R_t)])
        _, out = pert(sample)
        assert out["superadditivity"] == pytest.approx(expR)
