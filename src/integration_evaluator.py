# src/integration_evaluator.py
# =============================================================================
# End‑to‑end evaluator for four “spokes” of multimodal integration
# =============================================================================
# Spokes (what we test) — each compares a *model measurement* to a small *ideal*
# and appends tidy rows to results.csv:
#   A) Image fidelity                — S_fid vs I_img (I_img = Rv)
#   B) Modality weighting            — (S_v, S_t) vs (I_v, I_t)
#   C) Superadditivity (inv. eff.)   — Boost = S_joint − I_joint
#   D) Representational alignment    — S_align vs I_align (I_align = R_joint)
#
# Design:
#   • Image side uses *mechanism‑specific knobs*: gaussian_sigma | blur_radius_px |
#     cutout_frac (we print these).  Reliability Rv is *measured* by PSNR→[0,1].
#   • Text side uses one knob `ce_norm` for mask/shuffle/replace (we print this).
#     Reliability Rt is *measured* from GPT‑2 cross‑entropy via the standard
#     clean/random anchor mapping.
#   • Console prints mirror what we write to CSV: knobs + Rv/Rt + spoke outputs.
#     We *do not* print PSNR(dB) or raw CE by default.
#
# Downstream:
#   • Plots + metrics consume Rv, Rt, and spoke outputs; extra knob columns are
#     for reviewer traceability only and do not affect figures. :contentReference[oaicite:9]{index=9}
# =============================================================================

from __future__ import annotations

# ---- stdlib -----------------------------------------------------------------
import argparse
import csv
import hashlib
import logging
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# ---- third‑party -------------------------------------------------------------
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---- project imports ---------------------------------------------------------
from src.image_fidelity import noise_sigma_sweep, compute_I as compute_I_image, psnr_to_sigma  # :contentReference[oaicite:10]{index=10}
from src.stimuli import (  # :contentReference[oaicite:11]{index=11}
    load_coco_streaming,
    mask_ids,
    shuffle_ids,
    replace_ids,
    ce_loss,
    text_reliability,
    tokenizer,
    prepare_image,
    prepare_text,
    knob_from_sigma,  # native image knobs
)
from src import weighting, superadditivity, repr_align, model_hooks  # 

LOG = logging.getLogger("integration_evaluator")


# =============================================================================
# Reproducibility helpers
# =============================================================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Small utilities
# =============================================================================
def _fmt(x: float, p: int = 3) -> str:
    return f"{x:.{p}f}"


def _sigma_grid_for_spokes(vision_mode: str, n: int = 10) -> List[float]:
    """
    Sigma schedule for the 10×10 (Rv,Rt) grid in B–D.

    • gaussian: convert PSNR grid [0,50] dB to σ via psnr_to_sigma()
    • blur/cutout: use a generic strength grid [0,50]; mapping to radius/frac
      happens in stimuli.knob_from_sigma()/prepare_image(); Rv is *measured*
      from the actually disturbed image (PSNR).  :contentReference[oaicite:13]{index=13}
    """
    if vision_mode == "gaussian":
        return [float(psnr_to_sigma(p)) for p in np.linspace(0.0, 50.0, n)]
    return [float(s) for s in np.linspace(0.0, 50.0, n)]


def _vision_knob_cols(vision_mode: str, sigma: float) -> Tuple[str, str, str]:
    """
    CSV values for the three image knobs (empty string when N/A):
      gaussian → (σ, '', '')
      blur     → ('', radius_px, '')
      cutout   → ('', '', frac)
    """
    gs, rad, frac = knob_from_sigma(vision_mode, sigma)
    return (
        f"{gs:.1f}" if gs is not None else "",
        f"{rad:.2f}" if rad is not None else "",
        f"{frac:.3f}" if frac is not None else "",
    )


def _vision_knob_print(vision_mode: str, sigma: float) -> str:
    """Human‑readable knob snippet used in console logs."""
    gs, rad, frac = knob_from_sigma(vision_mode, sigma)
    if gs is not None:
        return f"gaussian_sigma={gs:.1f}"
    if rad is not None:
        return f"blur_radius_px={rad:.2f}"
    if frac is not None:
        return f"cutout_frac={frac:.3f}"
    return ""


# =============================================================================
# Spoke runners
# =============================================================================
def run_image_fidelity(
    hook: model_hooks.ModelHooks,
    pairs: Sequence[Tuple[Image.Image, str]],
    *,
    vision_mode: str,
    text_mode: str,
    run_tag: str,
    seed: int,
) -> List[Dict]:
    """
    A) Image fidelity — compare patch‑level cosine similarity S_fid (clean vs noisy)
       to the ideal I_img = Rv (identity).  Rows include per‑layer depth.  :contentReference[oaicite:14]{index=14}
    """
    LOG.info("\n=== Image Fidelity spoke ===")
    max_sig = 50.0 if vision_mode in ("blur", "cutout") else 255.0
    sigmas, _ = noise_sigma_sweep(max_sigma=max_sig)  # evenly spaced in [0,max_sig]  :contentReference[oaicite:15]{index=15}

    results: List[Dict] = []
    for pair_id, (img_entry, txt) in enumerate(pairs):
        # clean pass (once per image)
        img_pil = img_entry if isinstance(img_entry, Image.Image) else Image.open(img_entry).convert("RGB")
        rec_clean = hook.forward(image=img_pil, text=txt)
        layers = rec_clean["image_fidelity_layers"]

        for depth_idx, _ in enumerate(layers):
            clean_patches = layers[depth_idx][0]  # (seq, D)

            for sigma in sigmas:
                _, img_noisy_arr, Rv = prepare_image(img_entry, sigma, mode=vision_mode)
                img_noisy = Image.fromarray(img_noisy_arr).convert("RGB")

                rec_noisy = hook.forward(image=img_noisy, text=txt)
                noisy_patches = rec_noisy["image_fidelity_layers"][depth_idx][0]
                cos = F.cosine_similarity(clean_patches, noisy_patches, dim=-1)
                S_fid = float(((cos + 1.0) / 2.0).mean().item())

                I_img = compute_I_image(Rv)
                Delta = S_fid - I_img

                LOG.info(
                    "[%s+%s] spoke=image-fidelity | pair_id=%d | Depth=%d | %s | "
                    "Rv=%s | I_img=%s | S_fid=%s | Δ=%s | AbsError=%s",
                    vision_mode, text_mode, pair_id, depth_idx, _vision_knob_print(vision_mode, sigma),
                    _fmt(Rv), _fmt(I_img), _fmt(S_fid), _fmt(Delta), _fmt(abs(Delta)),
                )

                gsig, rad, frac = _vision_knob_cols(vision_mode, sigma)
                results.append(
                    {
                        "run_tag": run_tag,
                        "seed": seed,
                        "vision_mode": vision_mode,
                        "text_mode": text_mode,
                        "spoke": "image-fidelity",
                        "pair_id": int(pair_id),
                        "sigma": float(sigma),
                        "ce_norm": "",
                        "Depth": int(depth_idx),
                        "gaussian_sigma": gsig,
                        "blur_radius_px": rad,
                        "cutout_frac": frac,
                        "Rv": float(Rv),
                        "Rt": "",
                        "R_joint": "",
                        "I_img": float(I_img),
                        "I_v": "",
                        "I_t": "",
                        "I_joint": "",
                        "I_align": "",
                        "S_fid": float(S_fid),
                        "S_v": "",
                        "S_t": "",
                        "S_joint": "",
                        "S_align": "",
                        "Delta": float(Delta),
                        "Delta_v": "",
                        "Delta_t": "",
                        "AbsError": float(abs(Delta)),
                        "Boost": "",
                    }
                )
    return results


def _apply_text_corruption(ids_clean: List[int], ce_norm: float, text_mode: str) -> List[int]:
    if text_mode == "mask":
        return mask_ids(ids_clean, ce_norm)
    if text_mode == "shuffle":
        return shuffle_ids(ids_clean, frac=ce_norm)
    if text_mode == "replace":
        return replace_ids(ids_clean, p=ce_norm)
    raise ValueError(f"Unknown text_mode: {text_mode}")


def run_weighting(
    hook: model_hooks.ModelHooks,
    pairs: Sequence[Tuple[Image.Image, str]],
    *,
    vision_mode: str,
    text_mode: str,
    run_tag: str,
    seed: int,
) -> List[Dict]:
    """
    B) Modality weighting — compare CLIP’s relative raw‑norm weights (S_v,S_t)
       to the ideal I_v = Rv/(Rv+Rt), I_t = Rt/(Rv+Rt).  :contentReference[oaicite:16]{index=16}
    """
    LOG.info("\n=== Modality weighting spoke ===")
    sigma_grid = _sigma_grid_for_spokes(vision_mode, n=10)
    ce_norms = np.linspace(0.0, 1.0, 10)

    results: List[Dict] = []
    for pair_id, (img_entry, caption) in enumerate(pairs):
        ids_clean, ce_clean, ce_rand = prepare_text(caption)

        for sigma in sigma_grid:
            _, img_noisy_arr, Rv = prepare_image(img_entry, sigma, mode=vision_mode)
            img_noisy = Image.fromarray(img_noisy_arr).convert("RGB")

            for ce_norm in ce_norms:
                ids_corr = _apply_text_corruption(ids_clean, float(ce_norm), text_mode)
                ce_corr = ce_loss(ids_corr)
                Rt = text_reliability(ce_clean, ce_corr, ce_rand)

                txt_corr = tokenizer.decode(ids_corr, skip_special_tokens=True)
                Sv, St = hook.forward(image=img_noisy, text=txt_corr)["weighting"][0].tolist()

                I_v, I_t = weighting.compute_I(Rv, Rt)
                Delta_v, Delta_t = Sv - I_v, St - I_t
                AbsErr = (abs(Delta_v) + abs(Delta_t)) / 2.0

                LOG.info(
                    "[%s+%s] spoke=weighting | pair_id=%d | %s | ce_norm=%.2f | "
                    "Rv=%s | Rt=%s | Iv=%s | Sv=%s | Δv=%s | It=%s | St=%s | Δt=%s | AbsError=%s",
                    vision_mode, text_mode, pair_id, _vision_knob_print(vision_mode, sigma), ce_norm,
                    _fmt(Rv), _fmt(Rt), _fmt(I_v), _fmt(Sv), _fmt(Delta_v),
                    _fmt(I_t), _fmt(St), _fmt(Delta_t), _fmt(AbsErr),
                )

                gsig, rad, frac = _vision_knob_cols(vision_mode, sigma)
                results.append(
                    {
                        "run_tag": run_tag,
                        "seed": seed,
                        "vision_mode": vision_mode,
                        "text_mode": text_mode,
                        "spoke": "weighting",
                        "pair_id": int(pair_id),
                        "sigma": float(sigma),
                        "ce_norm": float(ce_norm),
                        "Depth": "",
                        "gaussian_sigma": gsig,
                        "blur_radius_px": rad,
                        "cutout_frac": frac,
                        "Rv": float(Rv),
                        "Rt": float(Rt),
                        "R_joint": "",
                        "I_img": "",
                        "I_v": float(I_v),
                        "I_t": float(I_t),
                        "I_joint": "",
                        "I_align": "",
                        "S_fid": "",
                        "S_v": float(Sv),
                        "S_t": float(St),
                        "S_joint": "",
                        "S_align": "",
                        "Delta": "",
                        "Delta_v": float(Delta_v),
                        "Delta_t": float(Delta_t),
                        "AbsError": float(AbsErr),
                        "Boost": "",
                    }
                )
    return results


def run_superadditivity(
    hook: model_hooks.ModelHooks,
    pairs: Sequence[Tuple[Image.Image, str]],
    *,
    vision_mode: str,
    text_mode: str,
    run_tag: str,
    seed: int,
) -> List[Dict]:
    """
    C) Superadditivity / inverse effectiveness — compare S_joint to
       I_joint = Rv + Rt − Rv*Rt.  (Clamped to [0,1].)  :contentReference[oaicite:17]{index=17}
    """
    LOG.info("\n=== Superadditivity spoke ===")
    sigma_grid = _sigma_grid_for_spokes(vision_mode, n=10)
    ce_norms = np.linspace(0.0, 1.0, 10)

    results: List[Dict] = []
    for pair_id, (img_entry, caption) in enumerate(pairs):
        ids_clean, ce_clean, ce_rand = prepare_text(caption)

        for sigma in sigma_grid:
            _, img_noisy_arr, Rv = prepare_image(img_entry, sigma, mode=vision_mode)
            img_noisy = Image.fromarray(img_noisy_arr).convert("RGB")

            for ce_norm in ce_norms:
                ids_corr = _apply_text_corruption(ids_clean, float(ce_norm), text_mode)
                ce_corr = ce_loss(ids_corr)
                Rt = text_reliability(ce_clean, ce_corr, ce_rand)

                txt_corr = tokenizer.decode(ids_corr, skip_special_tokens=True)
                rec = hook.forward(image=img_noisy, text=txt_corr)
                i, t, j = rec["img_emb"][0], rec["txt_emb"][0], rec["joint_emb"][0]
                S_joint = float(j.norm().item() / (i.norm().item() + t.norm().item() + 1e-8))

                I_joint = superadditivity.compute_I(Rv, Rt)
                Boost = S_joint - I_joint

                LOG.info(
                    "[%s+%s] spoke=superadditivity | pair_id=%d | %s | ce_norm=%.2f | "
                    "Rv=%s | Rt=%s | I_joint=%s | S_joint=%s | Boost=%s",
                    vision_mode, text_mode, pair_id, _vision_knob_print(vision_mode, sigma), ce_norm,
                    _fmt(Rv), _fmt(Rt), _fmt(I_joint), _fmt(S_joint), _fmt(Boost),
                )

                gsig, rad, frac = _vision_knob_cols(vision_mode, sigma)
                results.append(
                    {
                        "run_tag": run_tag,
                        "seed": seed,
                        "vision_mode": vision_mode,
                        "text_mode": text_mode,
                        "spoke": "superadditivity",
                        "pair_id": int(pair_id),
                        "sigma": float(sigma),
                        "ce_norm": float(ce_norm),
                        "Depth": "",
                        "gaussian_sigma": gsig,
                        "blur_radius_px": rad,
                        "cutout_frac": frac,
                        "Rv": float(Rv),
                        "Rt": float(Rt),
                        "R_joint": "",
                        "I_img": "",
                        "I_v": "",
                        "I_t": "",
                        "I_joint": float(I_joint),
                        "I_align": "",
                        "S_fid": "",
                        "S_v": "",
                        "S_t": "",
                        "S_joint": float(S_joint),
                        "S_align": "",
                        "Delta": "",
                        "Delta_v": "",
                        "Delta_t": "",
                        "AbsError": "",
                        "Boost": float(Boost),
                    }
                )
    return results


def run_repr_alignment(
    hook: model_hooks.ModelHooks,
    pairs: Sequence[Tuple[Image.Image, str]],
    *,
    vision_mode: str,
    text_mode: str,
    run_tag: str,
    seed: int,
) -> List[Dict]:
    """
    D) Representational alignment — S_align compares cosine(match) − mean cosine(mismatch, K=2)
       (mapped to [0,1]) against I_align = R_joint where R_joint = min(Rv,Rt).  :contentReference[oaicite:18]{index=18}
    """
    LOG.info("\n=== Representational alignment spoke ===")
    sigma_grid = _sigma_grid_for_spokes(vision_mode, n=10)
    ce_norms = np.linspace(0.0, 1.0, 10)
    all_captions = [t for _, t in pairs]

    results: List[Dict] = []
    for pair_id, (img_entry, caption) in enumerate(pairs):
        ids_clean, ce_clean, ce_rand = prepare_text(caption)

        for sigma in sigma_grid:
            _, img_noisy_arr, Rv = prepare_image(img_entry, sigma, mode=vision_mode)
            img_noisy = Image.fromarray(img_noisy_arr).convert("RGB")
            img_emb_noisy = hook.forward(image=img_noisy, text=caption)["img_emb"]

            for ce_norm in ce_norms:
                ids_corr = _apply_text_corruption(ids_clean, float(ce_norm), text_mode)
                ce_corr = ce_loss(ids_corr)
                Rt = text_reliability(ce_clean, ce_corr, ce_rand)
                txt_corr = tokenizer.decode(ids_corr, skip_special_tokens=True)

                rec_match = hook.forward(image=img_noisy, text=txt_corr)
                sim_match = float(F.cosine_similarity(img_emb_noisy, rec_match["txt_emb"]).item())

                # K=2 random mismatches (avoid trivial self‑match)
                K = 2
                pool = [c for c in all_captions if c != caption] or all_captions
                mis = []
                for _ in range(K):
                    rec_m = hook.forward(image=img_noisy, text=random.choice(pool))
                    mis.append(float(F.cosine_similarity(img_emb_noisy, rec_m["txt_emb"]).item()))
                sim_mismatch = float(np.mean(mis))

                match01, mismatch01 = (sim_match + 1.0) / 2.0, (sim_mismatch + 1.0) / 2.0
                S_align = float(np.clip(match01 - mismatch01, 0.0, 1.0))
                R_joint = repr_align.joint_reliability(Rv, Rt)
                I_align = repr_align.compute_I(R_joint)

                LOG.info(
                    "[%s+%s] spoke=repr-align | pair_id=%d | %s | ce_norm=%.2f | "
                    "Rv=%s | Rt=%s | R_joint=%s | I_align=%s | S_align=%s",
                    vision_mode, text_mode, pair_id, _vision_knob_print(vision_mode, sigma), ce_norm,
                    _fmt(Rv), _fmt(Rt), _fmt(R_joint), _fmt(I_align), _fmt(S_align),
                )

                gsig, rad, frac = _vision_knob_cols(vision_mode, sigma)
                results.append(
                    {
                        "run_tag": run_tag,
                        "seed": seed,
                        "vision_mode": vision_mode,
                        "text_mode": text_mode,
                        "spoke": "repr-align",
                        "pair_id": int(pair_id),
                        "sigma": float(sigma),
                        "ce_norm": float(ce_norm),
                        "Depth": "",
                        "gaussian_sigma": gsig,
                        "blur_radius_px": rad,
                        "cutout_frac": frac,
                        "Rv": float(Rv),
                        "Rt": float(Rt),
                        "R_joint": float(R_joint),
                        "I_img": "",
                        "I_v": "",
                        "I_t": "",
                        "I_joint": "",
                        "I_align": float(I_align),
                        "S_fid": "",
                        "S_v": "",
                        "S_t": "",
                        "S_joint": "",
                        "S_align": float(S_align),
                        "Delta": "",
                        "Delta_v": "",
                        "Delta_t": "",
                        "AbsError": "",
                        "Boost": "",
                    }
                )
    return results


# =============================================================================
# CSV writer
# =============================================================================
def _write_results_csv(rows: Iterable[Dict], out_path: Path) -> None:
    """
    Append rows to CSV with a stable header; new columns are included for the
    three image knobs (for reviewer traceability).
    """
    fieldnames = [
        "run_tag",
        "seed",
        "vision_mode",
        "text_mode",
        "spoke",
        "pair_id",
        "sigma",
        "ce_norm",
        "Depth",
        "gaussian_sigma",
        "blur_radius_px",
        "cutout_frac",
        "Rv",
        "Rt",
        "R_joint",
        "I_img",
        "I_v",
        "I_t",
        "I_joint",
        "I_align",
        "S_fid",
        "S_v",
        "S_t",
        "S_joint",
        "S_align",
        "Delta",
        "Delta_v",
        "Delta_t",
        "AbsError",
        "Boost",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = (not out_path.exists()) or (out_path.stat().st_size == 0)
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if header_needed:
            writer.writeheader()
        writer.writerows([{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])


# =============================================================================
# CLI
# =============================================================================
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision-mode", choices=["gaussian", "blur", "cutout"], default=os.getenv("V_MODE", "gaussian"))
    ap.add_argument("--text-mode", choices=["mask", "shuffle", "replace"], default=os.getenv("T_MODE", "mask"))
    ap.add_argument("--pairs", type=int, default=int(os.getenv("PAIRS", "50")))
    ap.add_argument("--seed", type=int, default=int(os.getenv("SEED", "0")))
    ap.add_argument("--manifest-seed", type=int, default=None)
    ap.add_argument("--offset", type=int, default=int(os.getenv("OFFSET", "0")))
    ap.add_argument("--save", default=os.getenv("RESULTS_CSV", "results.csv"))
    ap.add_argument("--run-tag", default=os.getenv("RUN_TAG"))
    ap.add_argument("--batch-all", action="store_true", default=os.getenv("BATCH_ALL", "0") == "1")
    ap.add_argument("--build-manifest", type=int, default=int(os.getenv("BUILD_MANIFEST", "0")))
    ap.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"), choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hook = model_hooks.ModelHooks(device=device)

    V_MODE = args.vision_mode
    T_MODE = args.text_mode
    SEED = args.seed
    MSEED = args.manifest_seed if args.manifest_seed is not None else SEED
    OFFSET = args.offset
    OUT_CSV = Path(args.save)
    RUN_TAG = args.run_tag or f"{V_MODE}+{T_MODE}"

    LOG.info("\nRun tag: %s | vision: %s | text: %s", RUN_TAG, V_MODE, T_MODE)
    LOG.info("pairs=%d seed=%d manifest_seed=%s offset=%d", args.pairs, SEED, MSEED, OFFSET)

    # Optional: prebuild manifest and exit (useful before multi‑seed runs).
    if args.build_manifest and args.build_manifest > 0:
        LOG.info("[build-manifest] Target ≈ %d images…", args.build_manifest)
        _ = list(load_coco_streaming(sample_size=args.build_manifest, split="train", seed=MSEED, offset=0))
        LOG.info("[build-manifest] Done.")
        return

    # Convenience: ABC sweep in one command appending to the same CSV
    if args.batch_all:
        import subprocess, sys

        combos = [("A", "gaussian", "mask"), ("B", "blur", "shuffle"), ("C", "cutout", "replace")]
        for tag, vmode, tmode in combos:
            cmd = [
                sys.executable,
                "-m",
                "src.integration_evaluator",
                "--vision-mode",
                vmode,
                "--text-mode",
                tmode,
                "--seed",
                str(SEED),
                "--manifest-seed",
                str(MSEED),
                "--pairs",
                str(args.pairs),
                "--offset",
                str(OFFSET),
                "--run-tag",
                tag,
                "--save",
                str(OUT_CSV),
                "--log-level",
                args.log_level,
            ]
            LOG.info("\n[batch-all] Running: %s", " ".join(cmd))
            subprocess.run(cmd, check=True)
        return

    # 1) Load pairs deterministically (stream or manifest replay)
    LOG.info("\n=== Loading COCO pairs via HF streaming ===")
    pairs = list(load_coco_streaming(sample_size=args.pairs, split="train", seed=MSEED, offset=OFFSET))

    # 2) Run all four spokes
    rows: List[Dict] = []
    rows += run_image_fidelity(hook, pairs, vision_mode=V_MODE, text_mode=T_MODE, run_tag=RUN_TAG, seed=SEED)
    rows += run_weighting(hook, pairs, vision_mode=V_MODE, text_mode=T_MODE, run_tag=RUN_TAG, seed=SEED)
    rows += run_superadditivity(hook, pairs, vision_mode=V_MODE, text_mode=T_MODE, run_tag=RUN_TAG, seed=SEED)
    rows += run_repr_alignment(hook, pairs, vision_mode=V_MODE, text_mode=T_MODE, run_tag=RUN_TAG, seed=SEED)

    # 3) Write results and print a content hash for provenance
    _write_results_csv(rows, OUT_CSV)
    sha = hashlib.sha256(OUT_CSV.read_bytes()).hexdigest()
    LOG.info("sha256(%s) = %s", OUT_CSV.name, sha)
    LOG.info("✅ Appended %d rows → %s", len(rows), OUT_CSV.resolve())


if __name__ == "__main__":
    main()