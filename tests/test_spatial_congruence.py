import pytest
from src.spatial import noise_sweep, compute_I, SpatialPerturbation

def test_spatial_reliability_and_I():
    max_shift = 4
    pairs = noise_sweep(max_shift)
    assert pairs[0] == (0, pytest.approx(1.0))
    assert pairs[-1] == (max_shift, pytest.approx(0.0))
    for shift, R in pairs:
        assert compute_I(R) == pytest.approx(R)

def test_spatial_perturbation_outputs_R():
    sample = {"image": None, "text": None}
    for shift, R in noise_sweep(5):
        pert = SpatialPerturbation([shift])
        _, out = pert(sample)
        assert out["spatial"] == pytest.approx(R)
