# tests/test_representational_alignment.py
"""
Tests for the Representational‑Alignment spoke.

We confirm:
• Joint reliability uses the min rule.
• Ideal is identity: I_align = R_joint.
"""
import pytest
from src.repr_align import joint_reliability, compute_I


def test_joint_reliability_min_rule():
    assert joint_reliability(0.9, 0.3) == 0.3
    assert joint_reliability(0.1, 0.1) == 0.1
    assert joint_reliability(0.0, 0.0) == 0.0


def test_compute_I_identity():
    for R in [0.0, 0.25, 0.5, 0.75, 1.0]:
        assert compute_I(R) == pytest.approx(R, abs=1e-6)