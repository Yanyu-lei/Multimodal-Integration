# src/stimuli.py

import os
import random
import json
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from pycocotools.coco import COCO


def shift_image(img: Image.Image, dx: int, dy: int) -> Image.Image:
    w, h = img.size
    # create a black canvas
    canvas = Image.new(img.mode, (w, h))
    # paste the original image shifted by (dx, dy)
    canvas.paste(img, (dx, dy))
    return canvas



def add_gaussian_noise(img, sigma: float) -> np.ndarray:
    """
    Add pixel-wise Gaussian noise with standard deviation sigma.
    Accepts either a PIL.Image.Image or an np.ndarray.
    Returns a uint8 numpy array.
    """
    if isinstance(img, Image.Image):
        arr = np.array(img)
    else:
        arr = img
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = arr.astype(float) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def shuffle_words(text: str, shuffle_frac: float = 0.3) -> str:
    """
    Shuffle a fraction of the words in the text.
    """
    words = text.split()
    n = max(1, int(len(words) * shuffle_frac))
    idxs = list(range(len(words)))
    swap = random.sample(idxs, n)
    random.shuffle(swap)
    for i, j in zip(idxs[:n], swap):
        words[i], words[j] = words[j], words[i]
    return " ".join(words)


def make_shifted_video(
    img: Image.Image,
    dx: int,
    dy: int,
    num_frames: int = 2
) -> list[Image.Image]:
    """
    Create a synthetic 'video' of num_frames by shifting each frame by (dx*t, dy*t).
    """
    frames = []
    for t in range(num_frames):
        frames.append(shift_image(img, dx * t, dy * t))
    return frames


def load_coco_image_caption(
    ann_file: str,
    images_dir: str,
    sample_size: int = 1000
) -> list[tuple[str, str]]:
    """
    Load a random sample of (image_path, caption) pairs from local COCO.
    Used by unit tests.
    """
    coco = COCO(ann_file)
    ids = random.sample(coco.getImgIds(), sample_size)
    pairs = []
    for i in ids:
        meta = coco.loadImgs(i)[0]
        img_path = os.path.join(images_dir, meta["file_name"])
        anns = coco.loadAnns(coco.getAnnIds(imgIds=i))
        caps = [a["caption"] for a in anns]
        pairs.append((img_path, random.choice(caps)))
    return pairs


def load_coco_remote(
    sample_size: int = 100,
    ann_file: str = None
) -> list[tuple[Image.Image, str]]:
    """
    1) Read the local captions JSON at ann_file (~250 MB).
    2) Randomly sample sample_size image IDs.
    3) For each, fetch the JPEG via its public URL.
    Returns list of (PIL.Image, caption).
    """
    if ann_file is None:
        raise ValueError("Please provide ann_file path to captions_train2017.json")
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    annos  = data["annotations"]

    # build mapping image_id → captions
    caps_map = {}
    for a in annos:
        caps_map.setdefault(a["image_id"], []).append(a["caption"])

    sampled = random.sample(images, sample_size)
    pairs = []
    for meta in sampled:
        img_id = meta["id"]
        url    = meta["coco_url"]
        cap    = random.choice(caps_map[img_id])

        resp = requests.get(url)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        pairs.append((img, cap))

    return pairs
