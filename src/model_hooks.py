"""
src/model_hooks.py

Direct CLIP embedding extractor for all five spokes, without forward hooks.
Provides:
  - img_emb: (B, D) image embeddings
  - txt_emb: (B, D) text embeddings
  - joint_emb: (B, D) sum of image+text embeddings
  - cls_emb: (B, D) alias for image embeddings (temporal)
  - spatial: (B, P, D) patch embeddings (last vision layer, excluding CLS)
  - weighting: (B, 2) normalized image vs text embedding norm ratio
  - representational_alignment: (B,) sigmoid of CLIP logits_per_image diagonal
"""
from typing import Any, Dict, List, Union
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

class ModelHooks:
    def __init__(self, device: str = "cpu"):
        self.device = device
        # Load pretrained CLIP model and processor
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    @torch.no_grad()
    def forward(
        self,
        image: Union[Image.Image, List[Image.Image]],
        text: Union[str, List[str]]
    ) -> Dict[str, Any]:
        # Batch inputs
        imgs = image if isinstance(image, list) else [image]
        txts = text  if isinstance(text, list)  else [text]
        # Preprocess
        inputs = self.processor(
            images=imgs,
            text=txts,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Forward with hidden states
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            output_attentions=False
        )

        # Basic embeddings
        img_emb = outputs.image_embeds      # (B, D)
        txt_emb = outputs.text_embeds       # (B, D)
        joint_emb = img_emb + txt_emb       # (B, D)
        cls_emb = img_emb                   # (B, D)

        # Spatial: patch embeddings from final vision layer
        vis_hid = outputs.vision_model_output.hidden_states[-1]  # (B, seq_len, D)
        spatial = vis_hid[:, 1:, :].cpu()                        # drop CLS token

        # Modality weighting: image vs text norm ratio
        iv = img_emb.norm(dim=-1, keepdim=True)
        it = txt_emb.norm(dim=-1, keepdim=True)
        total = iv + it + 1e-8
        weighting = torch.cat([iv/total, it/total], dim=-1).cpu()

        # Representational alignment: sigmoid of logits_per_image diagonal
        sim = outputs.logits_per_image
        if sim.ndim == 2 and sim.size(0) == sim.size(1):
            diag = torch.diagonal(sim)
        else:
            diag = sim.view(-1)
        representational_alignment = torch.sigmoid(diag).cpu()

        return {
            "img_emb": img_emb.cpu(),
            "txt_emb": txt_emb.cpu(),
            "joint_emb": joint_emb.cpu(),
            "cls_emb": cls_emb.cpu(),
            "spatial": spatial,
            "weighting": weighting,
            "representational_alignment": representational_alignment,
        }