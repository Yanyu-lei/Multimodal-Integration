"""
tests/test_model_hooks.py

Unit tests for src/model_hooks.py using CLIP-only hooks.
"""
import torch
from PIL import Image
import pytest

from src.model_hooks import ModelHooks

@pytest.fixture(scope="module")
def hook():
    return ModelHooks(device="cpu")


def test_static_keys_and_shapes(hook):
    """
    forward(image, text) should populate all five spokes + raw embeds:
    spatial, representational_alignment, weighting,
    img_emb, txt_emb, joint_emb, cls_emb
    """
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
    }
    assert expected.issubset(out.keys())
    assert isinstance(out["spatial"], torch.Tensor)
    assert isinstance(out["cls_emb"], torch.Tensor)


def test_weighting_sums_to_one(hook):
    """
    weighting is shape (1,2) and sums ~1.
    """
    img = Image.new("RGB", (224, 224), "white")
    w = hook.forward(image=img, text="test")["weighting"]
    assert w.shape == (1, 2)
    assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-3)


def test_superadditivity_embs_present(hook):
    """
    raw image, text, and joint embeddings exist and are tensors.
    """
    img = Image.new("RGB", (224, 224), "white")
    out = hook.forward(image=img, text="foo")
    for key in ["img_emb", "txt_emb", "joint_emb"]:
        assert key in out and isinstance(out[key], torch.Tensor)


def test_cls_emb_changes_with_image(hook):
    """
    cls_emb should reflect the image: two different images give different embeddings.
    """
    img1 = Image.new("RGB", (224, 224), "white")
    img2 = Image.new("RGB", (224, 224), "black")
    e1 = hook.forward(image=img1, text="x")["cls_emb"]
    e2 = hook.forward(image=img2, text="x")["cls_emb"]
    # not exactly equal
    assert not torch.allclose(e1, e2)
