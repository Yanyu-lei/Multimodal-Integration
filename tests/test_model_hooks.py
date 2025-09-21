# tests/test_model_hooks.py
"""
Unit tests for CLIP hooks (`src/model_hooks.py`).

We verify that:
• The forward pass returns the expected keys.
• Weighting is (1×2) and sums to ~1.
• Pooled embeddings exist for superadditivity.
• CLS embedding changes with the image (sanity check).
"""
import torch
from PIL import Image
import pytest
from src.model_hooks import ModelHooks


@pytest.fixture(scope="module")
def hook():
    return ModelHooks(device="cpu")


def test_static_keys_and_shapes(hook):
    img = Image.new("RGB", (224, 224), "white")
    out = hook.forward(image=img, text="hello")
    expected = {
        "spatial",
        "representational_alignment",
        "weighting",
        "img_emb",
        "txt_emb",
        "joint_emb",
        "cls_emb",
        "image_fidelity",
        "image_fidelity_layers",
    }
    assert expected.issubset(out.keys())
    assert isinstance(out["spatial"], torch.Tensor)
    assert isinstance(out["cls_emb"], torch.Tensor)


def test_weighting_sums_to_one(hook):
    img = Image.new("RGB", (224, 224), "white")
    w = hook.forward(image=img, text="test")["weighting"]
    assert w.shape == (1, 2)
    assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-3)


def test_superadditivity_embs_present(hook):
    img = Image.new("RGB", (224, 224), "white")
    out = hook.forward(image=img, text="foo")
    for key in ["img_emb", "txt_emb", "joint_emb"]:
        assert key in out and isinstance(out[key], torch.Tensor)


def test_cls_emb_changes_with_image(hook):
    img1 = Image.new("RGB", (224, 224), "white")
    img2 = Image.new("RGB", (224, 224), "black")
    e1 = hook.forward(image=img1, text="x")["cls_emb"]
    e2 = hook.forward(image=img2, text="x")["cls_emb"]
    assert not torch.allclose(e1, e2)  # different images → different CLS