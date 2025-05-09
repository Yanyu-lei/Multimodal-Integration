"""
src/stimuli.py

Simple perturbation helpers for all five spokes:
• Spatial: image shifts
• Temporal: synthetic video sequences
• Modality weighting: Gaussian noise + word shuffling
"""
from PIL import Image, ImageChops
import numpy as np
import random
from typing import List


# ---------- Spatial helpers ----------
def shift_image(img: Image.Image, dx: int, dy: int) -> Image.Image:
    """
    Return a copy of img shifted by (dx, dy) pixels.
    """
    return ImageChops.offset(img, dx, dy)


# ---------- Vision noise helpers ----------
def add_gaussian_noise(img: Image.Image, sigma: float = 25.0) -> Image.Image:
    """
    Add Gaussian noise (std=sigma) to a PIL Image and return a new Image.
    """
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr += noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------- Text helpers ----------
def shuffle_words(text: str) -> str:
    """
    Randomly shuffle the words in a text string.
    """
    words = text.split()
    random.shuffle(words)
    return " ".join(words)


# ---------- Video / temporal helpers ----------
def make_shifted_video(
    img: Image.Image,
    num_frames: int,
    max_shift: int,
    axis: str = "x"
) -> List[Image.Image]:
    """
    Create a synthetic video (list of PIL frames) by shifting `img`
    incrementally from 0 → max_shift over num_frames along `axis`.
    """
    frames: List[Image.Image] = []
    for t in range(num_frames):
        shift = int(round((t / (num_frames - 1)) * max_shift))
        dx, dy = (shift, 0) if axis == "x" else (0, shift)
        frames.append(shift_image(img, dx=dx, dy=dy))
    return frames
