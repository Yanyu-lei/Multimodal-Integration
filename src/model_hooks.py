"""
model_hooks.py
==============

Tiny wrapper around CLIP that plants *forward hooks* inside the network so we
can read out internal signals (S) for the evaluator.

Currently captures two metrics:

1. spatial
   Average CLS‑to‑patch attention weights from the **last** ViT self‑attention
   layer.  Shape returned: (batch, num_patches) where num_patches = 49 for
   ViT‑B/32 @ 224×224.

2. representational_alignment
   Cosine similarity between the final global image embedding and text
   embedding.

Extend this file with more hooks as other spokes need them.
"""
from typing import Dict, Any

import torch
from transformers import CLIPModel, CLIPProcessor


class ModelHooks:
    def __init__(self, device: str = "cuda"):
        self.device = device

        # 1. Load CLIP
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            attn_implementation="eager"  # avoid future warning
        ).to(device).eval()

        # 2. Paired tokenizer / image pre‑processor
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        # 3. Container the hooks will write into each forward pass
        self._records: Dict[str, Any] = {}

        # 4. Actually register the hooks
        self._register_hooks()

    # ------------------------------------------------------------------ #
    def _register_hooks(self) -> None:
        """Attach PyTorch forward hooks that capture the desired signals."""

        # ---- Spatial congruence: attention weights ------------------- #
        def save_spatial(_, __, output):
            """
            output is a tuple: (attn_output, attn_weights).
            We keep the mean CLS‑to‑patch weight over all heads.
            """
            attn_weights = output[1]                          # (B, heads, Q, K)
            if attn_weights is None:                          # should not happen
                return
            cls_to_patch = attn_weights[:, :, 0, 1:].mean(dim=1)  # (B, num_patches)
            self._records["spatial"] = cls_to_patch.detach().cpu()

        last_attn = self.model.vision_model.encoder.layers[-1].self_attn
        last_attn.register_forward_hook(save_spatial)

        # ---- Representational alignment: cosine of embeddings -------- #
        def save_alignment(_, __, output):
            """
            output is a transformers.CLIPOutput object.
            """
            img_emb = output.image_embeds                      # (B, D)
            txt_emb = output.text_embeds                       # (B, D)
            if img_emb is None or txt_emb is None:
                return
            cos = torch.nn.functional.cosine_similarity(img_emb, txt_emb, dim=-1)
            self._records["representational_alignment"] = cos.detach().cpu()

        # registering on the *top‑level* model gives us final embeddings
        self.model.register_forward_hook(save_alignment)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(self, image, text) -> Dict[str, Any]:
        """
        Run a forward pass and return the metrics captured by hooks.
        """
        inputs = self.processor(
            images=[image],
            text=[text],
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Important: ask CLIP to return attentions so the spatial hook gets data
        _ = self.model(**inputs, output_attentions=True)

        # Copy out the results, then clear for the next call
        out = self._records.copy()
        self._records.clear()
        return out