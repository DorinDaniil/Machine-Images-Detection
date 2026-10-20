"""PairDataset: training samples made from (real, re-decoded) pairs.
ImageDataset: plain images with labels and optional masks, for testing.

Three kinds of training samples, target map T in [0, 1] per pixel:
    real   the real photo, T = 0
    full   real mixed with the re-decoded copy: (1 - a) * real + a * fake, T = a everywhere
    local  the same mix only inside a random region M, pixels outside stay original, T = a * M
"""

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset


def random_mask(h: int, w: int, rng: np.random.Generator, area=(0.05, 0.6)) -> np.ndarray:
    """1-3 rectangles or ellipses covering a random share of the image."""
    mask = np.zeros((h, w), dtype=bool)
    yy, xx = np.mgrid[:h, :w]
    n = rng.integers(1, 4)
    for _ in range(n):
        a = rng.uniform(*area) / n * h * w  # pixels in this shape
        ratio = rng.uniform(0.5, 2.0)       # width / height
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        if rng.random() < 0.5:
            rh = np.sqrt(a / ratio)
            mask |= (abs(yy - cy) < rh / 2) & (abs(xx - cx) < rh * ratio / 2)
        else:
            ry = np.sqrt(a / (np.pi * ratio))
            mask |= ((yy - cy) / ry) ** 2 + ((xx - cx) / (ry * ratio)) ** 2 < 1
    return mask.astype(np.float32)


def blend(real: np.ndarray, fake: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """weight [H, W]: share of the fake in every pixel."""
    out = real * (1 - weight[..., None]) + fake * weight[..., None]
    return np.clip(out.round(), 0, 255).astype(np.uint8)


def degrade(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """What images get on the way through the internet: down-and-up scaling, blur, JPEG."""
    if rng.random() < 0.5:
        s = rng.uniform(0.5, 1.0)
        size = image.size
        image = image.resize((round(size[0] * s), round(size[1] * s)), Image.BICUBIC).resize(size, Image.BICUBIC)
    if rng.random() < 0.2:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
    if rng.random() < 0.7:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=int(rng.integers(40, 96)))
        image = Image.open(buf).convert("RGB")
    return image


def to_tensor(image) -> torch.Tensor:
    return torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)


class PairDataset(Dataset):
    def __init__(self, pairs: list[Path], vaes: list[str], crop: int, p_real=0.3, p_full=0.3, p_local=0.4, seed=0):
        self.pairs, self.vaes, self.crop = list(pairs), list(vaes), crop
        self.probs = np.array([p_real, p_full, p_local]) / (p_real + p_full + p_local)
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        pair = Path(self.pairs[i])
        real = np.asarray(Image.open(pair / "real.png").convert("RGB"), dtype=np.float32)
        h, w = real.shape[:2]

        mode = rng.choice(["real", "full", "local"], p=self.probs)
        if mode == "real":
            weight = np.zeros((h, w), np.float32)
            fake = real
        else:
            fake = np.asarray(Image.open(pair / f"{rng.choice(self.vaes)}.png").convert("RGB"), dtype=np.float32)
            mask = np.ones((h, w), np.float32) if mode == "full" else random_mask(h, w, rng)
            weight = rng.uniform(0.3, 1.0) * mask
        image = blend(real, fake, weight)

        c = min(self.crop, h, w)
        y, x = rng.integers(0, h - c + 1), rng.integers(0, w - c + 1)
        image = degrade(Image.fromarray(image[y:y + c, x:x + c]), rng)
        target = torch.from_numpy(weight[y:y + c, x:x + c].copy())[None]
        return {"image": to_tensor(image), "target": target, "label": target.max()}


class ImageDataset(Dataset):
    """items: dicts with "path", "label" and optionally "mask" (PNG, white = generated).

    Images keep their resolution: shrunk to max_side if larger, then cropped to the patch grid.
    """

    def __init__(self, items: list[dict], patch: int, max_side: int = 1024):
        self.items, self.patch, self.max_side = items, patch, max_side

    def __len__(self):
        return len(self.items)

    def _fit(self, image: Image.Image, resample) -> Image.Image:
        scale = self.max_side / max(image.size)
        if scale < 1:
            image = image.resize((round(image.width * scale), round(image.height * scale)), resample)
        w, h = image.width // self.patch * self.patch, image.height // self.patch * self.patch
        return image.crop(((image.width - w) // 2, (image.height - h) // 2, (image.width + w) // 2, (image.height + h) // 2))

    def __getitem__(self, i):
        item = self.items[i]
        image = self._fit(Image.open(item["path"]).convert("RGB"), Image.BICUBIC)
        out = {"image": to_tensor(image), "label": float(item["label"]), "has_mask": bool(item.get("mask"))}
        if item.get("mask"):
            mask = self._fit(Image.open(item["mask"]).convert("L"), Image.NEAREST)
            out["mask"] = torch.from_numpy(np.array(mask, dtype=np.float32) / 255)[None]
        else:
            out["mask"] = torch.zeros(1, image.height, image.width)
        return out
