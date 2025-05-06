# src/repr_align.py

def noise_sweep():
    """
    Returns a list of (is_match, R) pairs:
    - R = 1.0 if the representations match
    - R = 0.0 if they don’t
    """
    return [(True, 1.0), (False, 0.0)]

def compute_I(R: float) -> float:
    """Ideal alignment equals the reliability."""
    return R

class RepresentationalAlignmentPerturbation:
    """
    Simulates whether two inputs are representationally aligned.
    When called, returns (sample, {'representational_alignment': R}).
    """
    def __init__(self, levels: list[bool]):
        self.is_match = levels[0]

    def __call__(self, sample: dict):
        R = 1.0 if self.is_match else 0.0
        return sample, {"representational_alignment": R}
