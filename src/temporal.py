# src/temporal.py

import math

def noise_sweep(max_lag: int = 5, sigma: float = 1.0):
    """
    Returns a list of (lag, R) pairs,
    where R = exp(-(lag / sigma)^2).
    """
    return [(t, math.exp(-(t / sigma) ** 2)) for t in range(max_lag + 1)]

def compute_I(R: float) -> float:
    """Ideal temporal congruence is exactly R."""
    return R

class TemporalPerturbation:
    """
    Simulates temporal lag noise.
    When called, returns (unchanged sample, {'temporal': R}).
    """
    def __init__(self, lags: list[int], sigma: float = 1.0):
        self.lag = lags[0]
        self.sigma = sigma

    def __call__(self, sample: dict):
        R = math.exp(- (self.lag / self.sigma) ** 2)
        # In a real harness you'd shift the audio/video sync here
        return sample, {"temporal": R}
