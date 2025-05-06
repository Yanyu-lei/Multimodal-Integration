import pytest
from src.weighting import (
    VisionPSNRPerturbation,
    TextEntropyPerturbation,
    compute_I,
)

def test_vision_reliability_values():
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    sample = {"image": None, "text": "dummy"}
    for lvl in levels:
        pert = VisionPSNRPerturbation([lvl])
        _, R = pert(sample)
        assert R["vision"] == pytest.approx(lvl)
        assert R["text"]   == pytest.approx(1.0)

def test_text_reliability_values():
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    sample = {"image": None, "text": "dummy"}
    for lvl in levels:
        pert = TextEntropyPerturbation([lvl])
        _, R = pert(sample)
        expected = 1.0 - lvl
        assert R["text"]   == pytest.approx(expected)
        assert R["vision"] == pytest.approx(1.0)

def test_compute_I_values():
    cases = [
        (0.5, 0.5, 0.5),
        (0.8, 0.2, 0.8/(0.8+0.2)),
        (0.2, 0.8, 0.2/(0.2+0.8)),
        (0.0, 0.0, 0.5),
    ]
    for R_v, R_t, exp_I_v in cases:
        I_v, I_t = compute_I(R_v, R_t)
        assert I_v == pytest.approx(exp_I_v)
        assert I_t == pytest.approx(1.0 - exp_I_v)