"""
src/model_hooks.py

Tiny wrapper around CLIP that plants forward-hooks for all five spokes:
1. spatial congruence
2. temporal congruence (via cls_emb proxy)
3. modality weighting
4. superadditivity (raw embeddings)
5. representational alignment
"""
from typing import Any, Dict
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image


class ModelHooks:
    def __init__(self, device: str = "cuda"):
        self.device = device
        # 1. Load CLIP
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") \
                             .to(device).eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # 2. Container for hook outputs
        self._records: Dict[str, Any] = {}
        # 3. Register all hooks
        self._register_hooks()

    def _register_hooks(self):
        # --- Spatial congruence: CLS→patch attention --- #
        def save_spatial(module, inp, out):
            # out = (attn_output, attn_weights)
            attn_weights = out[1]  # (B, heads, Q, K)
            cls_to_patch = attn_weights[:, :, 0, 1:].mean(dim=1)  # (B, num_patches)
            self._records["spatial"] = cls_to_patch.detach().cpu()

        last_attn = self.model.vision_model.encoder.layers[-1].self_attn
        last_attn.register_forward_hook(save_spatial)

        # --- Representational alignment: cosine(img, text) --- #
        def save_alignment(module, inp, out):
            # out: CLIPOutput with image_embeds & text_embeds
            img_e, txt_e = out.image_embeds, out.text_embeds
            cos = torch.nn.functional.cosine_similarity(img_e, txt_e, dim=-1)
            self._records["representational_alignment"] = cos.detach().cpu()

        self.model.register_forward_hook(save_alignment)

        # --- Modality weighting: ||img|| vs ||txt|| ---------------- #
        def save_weighting(module, inp, out):
            img_e, txt_e = out.image_embeds, out.text_embeds
            w_img = img_e.norm(dim=-1, keepdim=True)
            w_txt = txt_e.norm(dim=-1, keepdim=True)
            total = w_img + w_txt + 1e-8
            weights = torch.cat([w_img/total, w_txt/total], dim=1)
            self._records["weighting"] = weights.detach().cpu()

        self.model.register_forward_hook(save_weighting)

        # --- Superadditivity: capture raw embeddings ------------- #
        def save_embs(module, inp, out):
            self._records["img_emb"]   = out.image_embeds.detach().cpu()
            self._records["txt_emb"]   = out.text_embeds.detach().cpu()
            self._records["joint_emb"] = (out.image_embeds + out.text_embeds).detach().cpu()

        self.model.register_forward_hook(save_embs)

        # --- Temporal proxy: capture CLS embedding --------------- #
        def save_cls(module, inp, out):
            self._records["cls_emb"] = out.image_embeds.detach().cpu()

        self.model.register_forward_hook(save_cls)

    @torch.no_grad()
    def forward(self, image: Image.Image, text: str) -> Dict[str, Any]:
        """
        Run a single CLIP forward pass on image+text.
        Hooks populate:
          spatial, representational_alignment,
          weighting, img_emb, txt_emb, joint_emb, cls_emb
        """
        inputs = self.processor(
            images=[image],
            text=[text],
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Ask for attentions so spatial hook fires
        _ = self.model(**inputs, output_attentions=True)

        out = self._records.copy()
        self._records.clear()
        return out
