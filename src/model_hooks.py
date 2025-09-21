# src/model_hooks.py
# =============================================================================
# CLIP interface with lightweight forward hooks
# =============================================================================
# Responsibilities
#   • Load CLIP ViT‑B/32 and expose one `forward()` that returns all signals
#     used by the four spokes in one pass.
#   • Capture *pre‑norm* projection vectors via forward hooks to compute
#     relative raw‑norm weights (weighting spoke).
#   • Expose multi‑depth patch features for image fidelity (three depths).
#
# Returns from forward(image, text)
#   {
#     "img_emb":   (B,D) pooled image embeddings,
#     "txt_emb":   (B,D) pooled text embeddings,
#     "joint_emb": (B,D) image+text vector,
#     "cls_emb":   (B,D) alias of image_embeds (kept for back‑compat),
#     "image_fidelity_layers": list[(B,seq-1,D)] at depths [4,8,last] (CLS removed),
#     "image_fidelity":        final layer grid (alias of last in list),
#     "weighting": (B,2) = [Iv/(Iv+It), It/(Iv+It)] from pre‑norm projections,
#     "representational_alignment": softmax over texts for the first text,
#     "spatial":   final layer grid (kept for back‑compat)
#   }
#
# The keys/shape contracts above are exercised by tests. :contentReference[oaicite:4]{index=4}
# =============================================================================

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Union

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ModelHooks:
    def __init__(self, device: str = "cpu"):
        self.device = device

        # 1) Load CLIP and freeze
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

        # 2) Processor: prefer fast tokenizer when available
        try:
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        except TypeError:  # older Transformers
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # 3) Storage for hook outputs
        self._records: Dict[str, torch.Tensor] = {}

        # 4) Register lightweight forward hooks on projection layers
        def _save_raw_img(_, __, out):  # out shape: (B,D)
            self._records["raw_img"] = out.detach().cpu()

        def _save_raw_txt(_, __, out):  # out shape: (B,D)
            self._records["raw_txt"] = out.detach().cpu()

        self.model.visual_projection.register_forward_hook(_save_raw_img)
        self.model.text_projection.register_forward_hook(_save_raw_txt)

    @torch.no_grad()
    def forward(self, image: Union[Image.Image, List[Image.Image]], text: Union[str, List[str]]) -> Dict[str, Any]:
        # Normalize inputs to lists
        imgs = image if isinstance(image, list) else [image]
        txts = text if isinstance(text, list) else [text]

        # Build batches using the processor’s parts directly (robust to API changes)
        img_inputs = self.processor.image_processor(images=imgs, return_tensors="pt")
        txt_inputs = self.processor.tokenizer(text=txts, return_tensors="pt", padding=True, truncation=True)

        # To device
        img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
        txt_inputs = {k: v.to(self.device) for k, v in txt_inputs.items()}

        # Run CLIP (hooks populate _records)
        outputs = self.model(
            pixel_values=img_inputs["pixel_values"],
            input_ids=txt_inputs["input_ids"],
            attention_mask=txt_inputs.get("attention_mask"),
            output_hidden_states=True,
            return_dict=True,
        )

        # Raw pre‑norm projection vectors captured by hooks
        raw_img = self._records.pop("raw_img")  # (B,D)
        raw_txt = self._records.pop("raw_txt")  # (B,D)
        iv = raw_img.norm(dim=-1, keepdim=True)  # (B,1)
        it = raw_txt.norm(dim=-1, keepdim=True)  # (B,1)
        total = iv + it + 1e-8
        weighting = torch.cat([iv / total, it / total], dim=-1).cpu()  # (B,2)

        # Pooled embeddings (already L2‑normalised by CLIP)
        img_emb = outputs.image_embeds.cpu()
        txt_emb = outputs.text_embeds.cpu()
        joint_emb = (outputs.image_embeds + outputs.text_embeds).cpu()
        cls_emb = outputs.image_embeds.cpu()  # kept for compatibility with earlier code/tests

        # Multi‑depth patch features for image fidelity (drop CLS)
        hidden_states = outputs.vision_model_output.hidden_states
        layer_indices = [4, 8, len(hidden_states) - 1]
        image_fidelity_layers: List[torch.Tensor] = [hidden_states[i][:, 1:, :].cpu() for i in layer_indices]
        image_fidelity = image_fidelity_layers[-1]
        spatial = image_fidelity  # alias

        # Light proxy for representational alignment: softmax over texts for the first text
        sim = outputs.logits_per_image
        repr_align = torch.softmax(sim, dim=-1)[:, 0].cpu()

        return {
            "img_emb": img_emb,
            "txt_emb": txt_emb,
            "joint_emb": joint_emb,
            "cls_emb": cls_emb,
            "image_fidelity": image_fidelity,
            "image_fidelity_layers": image_fidelity_layers,
            "weighting": weighting,
            "representational_alignment": repr_align,
            "spatial": spatial,
        }

    @torch.no_grad()
    def raw_norms(self, image: Union[Image.Image, List[Image.Image]], text: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience: return (||raw_img||, ||raw_txt||) as (B,1) tensors.
        """
        imgs = image if isinstance(image, list) else [image]
        txts = text if isinstance(text, list) else [text]

        img_inputs = self.processor.image_processor(images=imgs, return_tensors="pt")
        txt_inputs = self.processor.tokenizer(text=txts, return_tensors="pt", padding=True, truncation=True)
        img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
        txt_inputs = {k: v.to(self.device) for k, v in txt_inputs.items()}

        _ = self.model(
            pixel_values=img_inputs["pixel_values"],
            input_ids=txt_inputs["input_ids"],
            attention_mask=txt_inputs.get("attention_mask"),
            output_hidden_states=False,
            return_dict=True,
        )

        raw_img = self._records.pop("raw_img")
        raw_txt = self._records.pop("raw_txt")
        return raw_img.norm(dim=-1, keepdim=True), raw_txt.norm(dim=-1, keepdim=True)