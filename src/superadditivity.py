# src/superadditivity.py

DEFAULT_K = 1.0
EPS = 1e-6

def compute_R(R_v: float, R_t: float) -> float:
    """
    Reliability = min of the two unimodal reliabilities.
    """
    return min(R_v, R_t)

def compute_I(R_v: float, R_t: float, k: float = DEFAULT_K, eps: float = EPS) -> float:
    """
    Ideal (inverse‐effectiveness): I = k / (R + eps).
    """
    R = compute_R(R_v, R_t)
    return k / (R + eps)

class SuperadditivityPerturbation:
    """
    Simulates combined cue reliability by directly specifying (R_v, R_t).
    Returns (sample, {'superadditivity': R}).
    """
    def __init__(self, levels: list[tuple[float, float]]):
        # levels is a list of one (R_v, R_t) pair
        self.R_v, self.R_t = levels[0]

    def __call__(self, sample: dict):
        R = compute_R(self.R_v, self.R_t)
        return sample, {"superadditivity": R}
