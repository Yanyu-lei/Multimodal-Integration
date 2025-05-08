"""
Unit tests for src/stimuli.py

These are *sanity* checks, not exhaustive property tests.
"""

import numpy as np
from PIL import Image
import pytest

from src import stimuli


# ------------------------------------------------------------------ #
def test_shift_image_moves_pixel():
    """Red pixel at (10, 10) should appear at (14, 10) after dx=+4."""
    img = Image.new("RGB", (32, 32), "black")
    img.putpixel((10, 10), (255, 0, 0))          # set a red pixel

    shifted = stimuli.shift_image(img, dx=4, dy=0)

    assert shifted.getpixel((14, 10)) == (255, 0, 0)
    # original spot should be black
    assert shifted.getpixel((10, 10)) == (0, 0, 0)


# ------------------------------------------------------------------ #
def test_add_gaussian_noise_changes_some_pixels():
    """Noise should modify at least one pixel value but keep size/mode."""
    rng = np.random.default_rng(seed=0)  # deterministic
    img = Image.new("L", (8, 8), 128)    # gray image, 8‑bit, size 8×8

    noisy = stimuli.add_gaussian_noise(img, sigma=25.0)

    assert noisy.size == img.size
    assert noisy.mode == img.mode
    # at least one pixel differs
    assert np.any(np.array(noisy) != np.array(img))


# ------------------------------------------------------------------ #
def test_shuffle_words_preserves_multiset():
    """Shuffled text should have same words (frequency), order likely diff."""
    text = "the quick brown fox jumps over the lazy dog"
    shuffled = stimuli.shuffle_words(text)

    orig_words = sorted(text.split())
    new_words = sorted(shuffled.split())

    assert orig_words == new_words          # same bag of words
    if len(set(orig_words)) > 1:            # not all words identical
        assert shuffled != text             # usually different order
