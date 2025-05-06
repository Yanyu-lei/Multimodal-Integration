# tests/test_temporal_congruence.py

import pytest
import math
from src.temporal import noise_sweep, compute_I, TemporalPerturbation

def test_temporal_reliability_and_I():
    max_lag = 3
    sigma = 1.0
    pairs = noise_sweep(max_lag, sigma)
    # Check endpoints: lag=0 → R=1; lag=max_lag → R=exp(-(max_lag/sigma)^2)
    assert pairs[0] == (0, pytest.approx(1.0))
    expected_last = math.exp(- (max_lag / sigma) ** 2)
    assert pairs[-1] == (max_lag, pytest.approx(expected_last))
    # Check ideal I == R for every pair
    for lag, R in pairs:
        assert compute_I(R) == pytest.approx(R)

def test_temporal_perturbation_outputs_R():
    sample = {"audio": None, "video": None}
    sigma = 1.0
    max_lag = 4
    for lag, R in noise_sweep(max_lag, sigma):
        pert = TemporalPerturbation([lag], sigma=sigma)
        _, out = pert(sample)
        assert out["temporal"] == pytest.approx(R)
