import pytest
from src.repr_align import noise_sweep, compute_I, RepresentationalAlignmentPerturbation

def test_noise_sweep_values():
    pairs = noise_sweep()
    assert (True, pytest.approx(1.0)) in pairs
    assert (False, pytest.approx(0.0)) in pairs

def test_compute_I_values():
    # I should equal R for both cases
    for _, R in noise_sweep():
        assert compute_I(R) == pytest.approx(R)

def test_perturbation_outputs_R():
    sample = {"dummy": None}
    for is_match, R in noise_sweep():
        pert = RepresentationalAlignmentPerturbation([is_match])
        _, out = pert(sample)
        assert out["representational_alignment"] == pytest.approx(R)
