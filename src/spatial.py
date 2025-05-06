# src/spatial.py
def noise_sweep(max_shift: int = 5):
    return [(d, 1.0 - d/max_shift) for d in range(max_shift+1)]

def compute_I(R: float) -> float:
    return R

class SpatialPerturbation:
    def __init__(self, shifts: list[int]):
        self.shift = shifts[0]
    def __call__(self, sample: dict):
        # Here you’d shift sample['image'] by self.shift pixels
        R = 1.0 - self.shift/5
        return sample, {"spatial": R}
