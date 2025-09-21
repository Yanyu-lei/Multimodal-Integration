# tests/test_stimuli.py
"""
Tests for stimulus helpers (image/text) and reliability mapping.

We check:
• Gaussian noise preserves shape/bounds.
• `prepare_image` returns an Rv in [0,1] and is monotonic with 'sigma'.
• Masking increases CE; reliability mapping is monotonic.
• (Optional) COCO streaming smoke test (skipped in CI).
"""
import numpy as np
import pytest
from PIL import Image
from src.stimuli import (
    load_coco_streaming,
    add_gaussian_noise,
    prepare_image,
    prepare_text,
    mask_ids,
    ce_loss,
    text_reliability,
)


def test_add_gaussian_noise_shape_bounds():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    noisy = add_gaussian_noise(arr, sigma=5.0)
    assert noisy.shape == arr.shape
    assert noisy.dtype == np.uint8
    assert noisy.min() >= 0 and noisy.max() <= 255


def test_prepare_image_returns_rv_in_0_1():
    img = Image.new("RGB", (8, 8), "white")
    rv0 = prepare_image(img, sigma=0.0)[2]
    rv1 = prepare_image(img, sigma=20.0)[2]
    assert 0.0 <= rv0 <= 1.0 and 0.0 <= rv1 <= 1.0
    assert rv0 >= rv1  # more corruption → lower Rv


def test_mask_ids_and_ce_loss_monotonic():
    ids_clean, ce_clean, ce_rand = prepare_text("a small red car on the street")
    ids_lo = mask_ids(ids_clean, 0.1)
    ids_hi = mask_ids(ids_clean, 0.9)
    ce_lo = ce_loss(ids_lo)
    ce_hi = ce_loss(ids_hi)
    assert ce_lo <= ce_hi
    rt_lo = text_reliability(ce_clean, ce_lo, ce_rand)
    rt_hi = text_reliability(ce_clean, ce_hi, ce_rand)
    assert 0.0 <= rt_hi <= rt_lo <= 1.0


@pytest.mark.skip(reason="dataset may not be present in CI")
def test_load_coco_streaming_returns_pairs():
    pairs = list(load_coco_streaming(sample_size=1))
    assert len(pairs) == 1
    img, cap = pairs[0]
    assert hasattr(img, "size") or isinstance(img, str)
    assert isinstance(cap, str) and cap.strip()