# src/integration_evaluator.py

import csv
import os
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F

from src import (
    spatial,
    temporal,
    weighting,
    superadditivity,
    repr_align,
    stimuli,
    model_hooks,
)


def make_base_pair():
    """
    Dummy white‐square pair if COCO loading fails.
    """
    img = Image.new("RGB", (224, 224), "white")
    txt = "a white square"
    return img, txt


def main():
    hook = model_hooks.ModelHooks(device="cpu")  # change to "cuda" if you have GPU

    # ─── 1. Load 100 COCO samples via local JSON + HTTP ───────
    ann = os.path.join("coco", "annotations", "captions_train2017.json")
    try:
        pairs = stimuli.load_coco_remote(sample_size=1, ann_file=ann)
        print(f"Loaded {len(pairs)} COCO samples via local JSON + HTTP")
    except Exception as e:
        print("Failed to load COCO locally, falling back to dummy pair:", e)
        pairs = [make_base_pair()]

    results = []

    # ─── 2. Spatial congruence ───────────────────────────────
    print("\n=== Spatial spoke ===")
    for img_entry, txt in pairs:
        img = img_entry if isinstance(img_entry, Image.Image) else Image.open(img_entry).convert("RGB")
        for shift, R in spatial.noise_sweep(max_shift=8):
            img_p = stimuli.shift_image(img, dx=shift, dy=0)
            rec = hook.forward(image=img_p, text=txt)
            S = float(rec["spatial"].mean().item())
            I = spatial.compute_I(R)
            score = 1 - abs(S - I)
            print(f"shift={shift:>2}px | R={R:.2f} | I={I:.2f} | S={S:.2f} | score={score:.2f}")
            results.append(("spatial", shift, R, I, S, score))

    # ─── 3. Temporal congruence (CLS stability) ─────────────
    print("\n=== Temporal spoke ===")
    for img_entry, txt in pairs:
        img = img_entry if isinstance(img_entry, Image.Image) else Image.open(img_entry).convert("RGB")
        for lag, R in temporal.noise_sweep(max_lag=8, sigma=4.0):
            video = stimuli.make_shifted_video(img, dx=lag, dy=0, num_frames=2)
            emb1 = hook.forward(image=video[0], text=txt)["cls_emb"]
            emb2 = hook.forward(image=video[1], text=txt)["cls_emb"]
            S = float(F.cosine_similarity(emb1, emb2, dim=-1).item())
            I = temporal.compute_I(R)
            score = 1 - abs(S - I)
            print(f"lag={lag:>2}  | R={R:.2f} | I={I:.2f} | S={S:.2f} | score={score:.2f}")
            results.append(("temporal", lag, R, I, S, score))

    # ─── 4. Modality weighting ────────────────────────────────
    print("\n=== Modality weighting spoke ===")
    for img_entry, txt in pairs:
        img = img_entry if isinstance(img_entry, Image.Image) else Image.open(img_entry).convert("RGB")
        for Rv, Rt in weighting.noise_sweep():
            img_n = stimuli.add_gaussian_noise(img, sigma=weighting.psnr_to_sigma(Rv))
            txt_n = stimuli.shuffle_words(txt)
            rec = hook.forward(image=img_n, text=txt_n)
            Sv, St = rec["weighting"][0].tolist()
            Iw = weighting.compute_I(Rv, Rt)
            score = 1 - (abs(Sv - Iw[0]) + abs(St - Iw[1])) / 2
            print(f"Rv={Rv:.2f}, Rt={Rt:.2f} | I=({Iw[0]:.2f},{Iw[1]:.2f}) | "
                  f"S=({Sv:.2f},{St:.2f}) | score={score:.2f}")
            results.append(("weighting", f"{Rv:.2f},{Rt:.2f}", (Rv, Rt), Iw, (Sv, St), score))

    # ─── 5. Superadditivity ─────────────────────────────────
    print("\n=== Superadditivity spoke ===")
    for img_entry, txt in pairs:
        img = img_entry if isinstance(img_entry, Image.Image) else Image.open(img_entry).convert("RGB")
        for Rv, Rt in superadditivity.noise_sweep():
            img_n = stimuli.add_gaussian_noise(img, sigma=weighting.psnr_to_sigma(Rv))
            txt_n = stimuli.shuffle_words(txt)
            rec = hook.forward(image=img_n, text=txt_n)
            img_e, txt_e, joint_e = rec["img_emb"], rec["txt_emb"], rec["joint_emb"]
            ε = 1e-8
            ni, nt = img_e.norm(dim=-1), txt_e.norm(dim=-1)
            nj = joint_e.norm(dim=-1)
            base = torch.max(ni, nt) + ε
            S = float(((nj - base) / base).item())
            R = superadditivity.compute_R(Rv, Rt)
            I = superadditivity.compute_I(R)
            score = 1 - abs(S - I)
            print(f"Rv={Rv:.2f}, Rt={Rt:.2f} | R={R:.2f} | I={I:.2f} | S={S:.2f} | score={score:.2f}")
            results.append(("superadditivity", f"{Rv:.2f},{Rt:.2f}", R, I, S, score))

    # ─── 6. Representational alignment ───────────────────────
    print("\n=== Representational alignment spoke ===")
    for img_entry, txt in pairs:
        img = img_entry if isinstance(img_entry, Image.Image) else Image.open(img_entry).convert("RGB")
        for noise, R in repr_align.noise_sweep():
            img_n = stimuli.add_gaussian_noise(img, sigma=noise)
            rec = hook.forward(image=img_n, text=txt)
            S = float(rec["representational_alignment"].item())
            I = repr_align.compute_I(R)
            score = 1 - abs(S - I)
            print(f"noise={noise:.2f} | R={R:.2f} | I={I:.2f} | S={S:.2f} | score={score:.2f}")
            results.append(("alignment", noise, R, I, S, score))

    # ─── Write out CSV ───────────────────────────────────────
    out = Path("results.csv")
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["spoke", "param", "R", "I", "S", "score"])
        writer.writerows(results)
    print(f"\n✅ Wrote results to {out.resolve()}")


if __name__ == "__main__":
    main()