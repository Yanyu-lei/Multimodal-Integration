# src/weighting.py

def vision_reliability(psnr: float) -> float:
    return psnr

def text_reliability(entropy: float) -> float:
    return 1.0 - entropy

def compute_I(R_v: float, R_t: float) -> tuple[float, float]:
    total = R_v + R_t
    if total == 0:
        return 0.5, 0.5
    return (R_v / total, 1.0 - R_v / total)

class VisionPSNRPerturbation:
    def __init__(self, psnr_levels: list[float]):
        self.psnr = psnr_levels[0]

    def __call__(self, sample: dict):
        return sample, {"vision": vision_reliability(self.psnr), "text": 1.0}

class TextEntropyPerturbation:
    def __init__(self, entropy_levels: list[float]):
        self.entropy = entropy_levels[0]

    def __call__(self, sample: dict):
        return sample, {"vision": 1.0, "text": text_reliability(self.entropy)}
