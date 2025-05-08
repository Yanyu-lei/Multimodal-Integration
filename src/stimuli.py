"""
Simple image / text perturbations reused by every spoke.
Keep PIL inputs & outputs so everything stays lightweight.
"""
from PIL import Image, ImageChops
import numpy as np
import random

# ---------- Spatial helpers ----------
def shift_image(img: Image.Image, dx: int, dy: int) -> Image.Image:
    """Return a copy of img shifted by (dx, dy) pixels."""
    return ImageChops.offset(img, dx, dy)

# ---------- Vision noise helpers ----------
def add_gaussian_noise(img: Image.Image, sigma: float = 25.0) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    arr += np.random.normal(0, sigma, arr.shape)
    return Image.fromarray(np.uint8(np.clip(arr, 0, 255)))

# ---------- Text helpers ----------
def shuffle_words(text: str) -> str:
    words = text.split()
    random.shuffle(words)
    return " ".join(words)
