"""
tests/test_stimuli.py

Unit tests for src/stimuli.py helpers.
"""
import numpy as np
from PIL import Image
import pytest

from src.stimuli import (
    shift_image,
    add_gaussian_noise,
    shuffle_words,
    make_shifted_video,
)


def test_shift_image_moves_pixel():
    img = Image.new("RGB", (32, 32), "black")
    img.putpixel((10, 10), (255, 0, 0))
    shifted = shift_image(img, dx=4, dy=0)

    # Pixel should move from (10,10) to (14,10)
    assert shifted.getpixel((14, 10)) == (255, 0, 0)
    # Original spot should now be black
    assert shifted.getpixel((10, 10)) == (0, 0, 0)


def test_add_gaussian_noise_changes_values_but_not_shape():
    img = Image.new("L", (8, 8), 128)  # gray
    noisy = add_gaussian_noise(img, sigma=25.0)

    # Mode and size preserved
    assert noisy.mode == img.mode
    assert noisy.size == img.size
    # At least one pixel changed
    assert np.any(np.array(noisy) != np.array(img))


def test_shuffle_words_preserves_and_mixes():
    text = "the quick brown fox"
    shuffled = shuffle_words(text)

    # Same multiset of words
    assert sorted(shuffled.split()) == sorted(text.split())
    # Should still have four words
    assert len(shuffled.split()) == 4


def test_make_shifted_video_length_and_shift():
    img = Image.new("RGB", (16, 16), "black")
    img.putpixel((0, 0), (255, 0, 0))

    frames = make_shifted_video(img, num_frames=5, max_shift=8, axis="x")

    # 1. Five frames
    assert isinstance(frames, list)
    assert len(frames) == 5

    # 2. First frame unshifted
    assert frames[0].getpixel((0, 0)) == (255, 0, 0)

    # 3. Last frame shifted by max_shift=8
    assert frames[-1].getpixel((8, 0)) == (255, 0, 0)

    # 4. Middle frame ~half shift
    assert frames[2].getpixel((4, 0)) == (255, 0, 0)
