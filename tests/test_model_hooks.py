from src.model_hooks import ModelHooks
from PIL import Image

def test_spatial_hook():
    h = ModelHooks(device="cpu")          # CPU is fine for a quick test
    img = Image.new("RGB", (224, 224), "white")
    out = h.forward(img, "a white square")
    assert "spatial" in out
    assert out["spatial"].shape[-1] == 49  # 7×7 ViT patches
