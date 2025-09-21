# src/stimuli.py
# =============================================================================
# Stimuli construction + reliability mapping (Rv, Rt)
# =============================================================================
# Responsibilities
#   • Deterministic COCO streaming with a manifest for exact replay.
#   • Image corruptions (Gaussian noise, Gaussian blur, cutout) and image
#     reliability Rv measured by PSNR→[0,1].
#   • Text corruptions (mask, shuffle, replace) and text reliability Rt
#     measured from GPT‑2 cross‑entropy (CE) via a clean/random anchor.
#   • A single helper `knob_from_sigma(mode, sigma)` exposes the *native*
#     knob per image mechanism, so both console prints and CSV reflect the
#     true control: gaussian_sigma | blur_radius_px | cutout_frac.
#
# The rest of the project (spokes, plots, metrics) reads *gauges* (Rv, Rt),
# not raw knobs, which keeps the analysis on a common [0,1] scale.  Tests
# cover invariants/monotonicity for the PSNR and CE mappings.  (See tests/.)
# =============================================================================

from __future__ import annotations

# ---- stdlib -----------------------------------------------------------------
import hashlib
import itertools
import json
import os
import random
from io import BytesIO
from typing import Iterator, Tuple

# ---- third‑party -------------------------------------------------------------
import numpy as np
from PIL import Image, ImageFilter
from datasets import load_dataset
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---- constants ---------------------------------------------------------------
MANIFEST = "manifest.json"          # where we cache COCO bytes + captions
PSNR_CAP_DB = 50.0                  # 50 dB ≈ near‑lossless → Rv=1
BLUR_MAX_RADIUS_PX = 5.0            # upper bound for Gaussian blur radius
CUTOUT_MAX_FRAC   = 0.5             # upper bound for cutout area fraction

# Reuse one HTTP session with light retries (faster, more robust)
_SESSION = requests.Session()
_SESSION.mount("http://",  HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.3)))
_SESSION.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.3)))


def _get(url: str, timeout: float = 20.0) -> bytes:
    """Small helper: GET with retries via the pooled session."""
    resp = _SESSION.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


# =============================================================================
# 1) Deterministic COCO streaming + manifest replay
# =============================================================================
def load_coco_streaming(
    sample_size: int = 1000,
    split: str = "train",
    manifest_path: str = MANIFEST,
    seed: int | None = None,
    offset: int = 0,
    extend_ok: bool = True,
) -> Iterator[Tuple[Image.Image, str]]:
    """
    Yield (image, caption) pairs deterministically.

    First run:
      • stream from phiyodr/coco2017 via HF, fetch each JPEG, and write a
        manifest with bytes (hex), caption, and sha256.
    Later runs:
      • replay from the manifest with a seeded shuffle + offset, so batches are
        reproducible and disjoint across seeds/offsets.  (This is what end‑to‑
        end scripts rely on.)  The manifest can be extended if more items are
        requested than it currently holds.  :contentReference[oaicite:6]{index=6}
    """
    rng = random.Random(0 if seed is None else seed)

    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as fh:
            manifest = json.load(fh)

        need = offset + sample_size
        if extend_ok and need > len(manifest):
            seen = {m["sha256"] for m in manifest}
            ds = load_dataset("phiyodr/coco2017", split=split, streaming=True)
            for ex in ds.shuffle(seed=seed or 0):
                img_bytes = _get(ex["coco_url"], timeout=20.0)
                sha = hashlib.sha256(img_bytes).hexdigest()
                if sha in seen:
                    continue
                manifest.append(
                    {
                        "id": ex["image_id"],
                        "caption": ex["captions"][0],
                        "bytes_hex": img_bytes.hex(),
                        "sha256": sha,
                    }
                )
                seen.add(sha)
                if len(manifest) >= need:
                    break
            with open(manifest_path, "w") as fh:
                json.dump(manifest, fh, indent=2)

        idxs = list(range(len(manifest)))
        rng.shuffle(idxs)
        for i in idxs[offset : offset + sample_size]:
            entry = manifest[i]
            img = Image.open(BytesIO(bytes.fromhex(entry["bytes_hex"]))).convert("RGB")
            yield img, entry["caption"]
        return

    # First run: build a minimal manifest while yielding
    ds = load_dataset("phiyodr/coco2017", split=split, streaming=True)
    manifest = []
    for ex in itertools.islice(ds.shuffle(seed=seed or 0), sample_size):
        img_bytes = _get(ex["coco_url"], timeout=20.0)
        caption = ex["captions"][0]
        sha = hashlib.sha256(img_bytes).hexdigest()
        manifest.append(
            {"id": ex["image_id"], "caption": caption, "bytes_hex": img_bytes.hex(), "sha256": sha}
        )
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        yield img, caption
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)


# =============================================================================
# 2) Image corruptions + image reliability Rv
# =============================================================================
def add_gaussian_noise(img, sigma: float) -> np.ndarray:
    """Add pixel‑wise Gaussian noise (std = sigma). Accepts PIL or ndarray; returns uint8 array."""
    arr = np.array(img) if isinstance(img, Image.Image) else img
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = arr.astype(float) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_gaussian_blur(img, radius: float) -> np.ndarray:
    """Gaussian blur with the given kernel radius (pixels)."""
    pil = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img))
    if radius <= 0:
        return np.asarray(pil)
    return np.asarray(pil.filter(ImageFilter.GaussianBlur(float(radius))))


def add_cutout(img, frac: float, fill: int = 0) -> np.ndarray:
    """
    Remove a single rectangle covering roughly `frac` of the image area (0–0.9).
    Fills with `fill` (0=black).  Location is random but bounded by image size.
    """
    pil = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img))
    w, h = pil.size
    frac = float(np.clip(frac, 0.0, 0.9))
    if frac <= 0.0:
        return np.asarray(pil)

    # Pick a rectangle with area ≈ frac·(w·h)
    A = max(1, int(w * h * frac))
    cw = max(1, int(np.sqrt(A)))
    ch = max(1, int(A / cw))
    cw = min(cw, w)
    ch = min(ch, h)

    x0 = random.randint(0, max(0, w - cw))
    y0 = random.randint(0, max(0, h - ch))
    arr = np.asarray(pil).copy()
    arr[y0 : y0 + ch, x0 : x0 + cw, :] = fill
    return arr


def image_reliability(img_clean, img_noisy, cap: float = PSNR_CAP_DB) -> float:
    """
    Gauge (image side): PSNR on the actually disturbed image, then map to Rv ∈ [0,1]
    by dividing by the cap (50 dB).  Clean images ⇒ Rv≈1.0.  :contentReference[oaicite:7]{index=7}
    """
    from src.image_fidelity import compute_psnr  # local import to avoid cycles
    clean_arr = np.array(img_clean)
    noisy_arr = np.array(img_noisy)
    psnr = compute_psnr(clean_arr, noisy_arr)
    return float(min(1.0, psnr / cap))


def knob_from_sigma(mode: str, sigma: float) -> Tuple[float | None, float | None, float | None]:
    """
    Map the *generic* image dial `sigma` to the *native* mechanism knob.

    Returns a triple (gaussian_sigma, blur_radius_px, cutout_frac), using None
    for non‑applicable fields.  We normalise blur/cutout strength by alpha=sigma/50,
    which keeps the human dial consistent across modes while letting us print
    true units for each mechanism.
    """
    alpha = float(np.clip(sigma / 50.0, 0.0, 1.0))
    if mode == "gaussian":
        return float(sigma), None, None
    if mode == "blur":
        return None, float(BLUR_MAX_RADIUS_PX * alpha), None
    if mode == "cutout":
        return None, None, float(CUTOUT_MAX_FRAC * alpha)
    raise ValueError(f"Unknown vision mode: {mode}")


def prepare_image(img_entry, sigma: float, *, mode: str = "gaussian"):
    """
    Return (clean_array, noisy_array, Rv) for the selected corruption mode.

    • `sigma` is the generic intensity dial we sweep in the evaluator.
      It is converted to the mechanism’s native knob via `knob_from_sigma`,
      the corruption is applied, and then **Rv is measured** from PSNR.
    """
    img = img_entry if isinstance(img_entry, Image.Image) else Image.open(img_entry).convert("RGB")
    arr_clean = np.asarray(img)

    gs, radius, frac = knob_from_sigma(mode, sigma)
    if mode == "gaussian":
        arr_noisy = add_gaussian_noise(arr_clean, gs)
    elif mode == "blur":
        arr_noisy = add_gaussian_blur(arr_clean, radius)
    elif mode == "cutout":
        arr_noisy = add_cutout(arr_clean, frac)
    else:
        raise ValueError(f"Unknown vision mode: {mode}")

    Rv = image_reliability(arr_clean, arr_noisy)
    return arr_clean, arr_noisy, Rv


# =============================================================================
# 3) Text corruptions + text reliability Rt
# =============================================================================
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load GPT‑2 once (CPU works; GPU speeds up CE.)
_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
_lm_model = GPT2LMHeadModel.from_pretrained("gpt2").eval()


def mask_ids(ids: list[int], p: float) -> list[int]:
    """Mask tokens with probability p by replacing with EOS (GPT‑2 has no [MASK])."""
    EOS = _tokenizer.eos_token_id
    return [tid if random.random() >= p else EOS for tid in ids]


def shuffle_ids(ids: list[int], frac: float) -> list[int]:
    """Randomly permute a fraction of token positions; preserves the token multiset."""
    n = len(ids)
    if n == 0 or frac <= 0.0:
        return ids[:]
    k = max(1, int(frac * n))
    idx = list(range(n))
    random.shuffle(idx)
    chosen = idx[:k]
    out = ids[:]
    pool = [out[i] for i in chosen]
    random.shuffle(pool)
    for i, v in zip(chosen, pool):
        out[i] = v
    return out


def replace_ids(ids: list[int], p: float) -> list[int]:
    """Randomly replace tokens with probability p by a random vocab id."""
    V = _tokenizer.vocab_size
    return [tid if random.random() >= p else random.randint(0, V - 1) for tid in ids]


def ce_loss(ids: list[int]) -> float:
    """Average per‑token cross‑entropy from GPT‑2 on the provided ids."""
    if not ids:
        return 0.0
    device = next(_lm_model.parameters()).device
    ids_t = torch.tensor([ids], device=device)
    with torch.no_grad():
        return float(_lm_model(ids_t, labels=ids_t).loss.item())


def text_reliability(ce_clean: float, ce_corr: float, ce_rand: float) -> float:
    """
    Gauge (text side): map cross‑entropy to Rt ∈ [0,1] using clean/random anchors:
      Rt = (ce_rand − ce_corr) / (ce_rand − ce_clean), clipped.
    Clean ⇒ Rt=1; random ⇒ Rt=0.  (Bounds/monotonicity are unit‑tested.)  :contentReference[oaicite:8]{index=8}
    """
    denom = max(ce_rand - ce_clean, 1e-8)
    return float(np.clip((ce_rand - ce_corr) / denom, 0.0, 1.0))


# Expose the tokenizer for encode/decode in the evaluator
tokenizer = _tokenizer


def prepare_text(caption: str):
    """Return `(ids_clean, ce_clean, ce_rand)` for a caption (used to compute Rt)."""
    ids_clean = tokenizer.encode(caption, add_special_tokens=False)
    ce_clean = ce_loss(ids_clean)
    ids_rand = [random.randint(0, tokenizer.vocab_size - 1) for _ in ids_clean]
    ce_rand = ce_loss(ids_rand)
    return ids_clean, ce_clean, ce_rand
