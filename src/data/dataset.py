"""PairDataset: training samples made from (real, re-decoded) pairs.
ImageDataset: plain images with labels and optional masks, for testing.

Three kinds of training samples, target map T in [0, 1] per pixel:
    real   the real image, T = 0
    full   real mixed with a re-decoded copy: (1 - a) * real + a * fake, T = a everywhere
    local  the same mix only inside a random region M, pixels outside stay original, T = a * M
The real and its copies share the JPEG history (see decoders.save_like), so only the decoder trace differs.
"""

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from .decoders import find


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


def jpeg(image: Image.Image, quality: int, subsampling: int = 2) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(quality), subsampling=subsampling)
    return Image.open(buf).convert("RGB")


def degrade(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """What images get on the way through the internet. Same distribution for real and fake samples."""
    if rng.random() < 0.4:  # down-and-up or plain upscale, size kept
        s = rng.uniform(0.5, 0.9) if rng.random() < 0.7 else rng.uniform(1.2, 2.0)
        size = image.size
        image = image.resize((round(size[0] * s), round(size[1] * s)), Image.BICUBIC).resize(size, Image.BICUBIC)
    if rng.random() < 0.15:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    if rng.random() < 0.25:  # sensor-like noise, so "noisy = real" is not a shortcut
        x = np.asarray(image, dtype=np.float32) + rng.normal(0, rng.uniform(0.5, 4), (image.height, image.width, 1))
        image = Image.fromarray(np.clip(x.round(), 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        image = jpeg(image, rng.integers(50, 96), subsampling=int(rng.choice([0, 2])))
    return image


def to_tensor(image) -> torch.Tensor:
    return torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)


def load_rgb(path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32)


class PairDataset(Dataset):
    def __init__(self, pairs: list, decoders: list[str], crop: int, p_real=0.3, p_full=0.3, p_local=0.4, seed=0):
        self.pairs, self.decoders, self.crop = [Path(p) for p in pairs], list(decoders), crop
        self.probs = np.array([p_real, p_full, p_local]) / (p_real + p_full + p_local)
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        pair = self.pairs[i]
        real = load_rgb(find(pair, "real"))
        h, w = real.shape[:2]

        mode = rng.choice(["real", "full", "local"], p=self.probs)
        if mode == "real":
            weight, fake = np.zeros((h, w), np.float32), real
        else:
            fake = load_rgb(find(pair, str(rng.choice(self.decoders))))
            mask = np.ones((h, w), np.float32) if mode == "full" else random_mask(h, w, rng)
            weight = rng.uniform(0.3, 1.0) * mask
        image = blend(real, fake, weight)

        c = self.crop
        y, x = rng.integers(0, h - c + 1), rng.integers(0, w - c + 1)
        image = degrade(Image.fromarray(image[y:y + c, x:x + c]), rng)
        target = torch.from_numpy(weight[y:y + c, x:x + c].copy())[None]
        return {"image": to_tensor(image), "target": target, "label": target.max()}


class ImageDataset(Dataset):
    """items: dicts with "path", "label" and optionally "mask" (PNG, white = generated).

    protocol "native":    keep resolution and bytes; shrink only above max_side; crop to the patch grid.
    protocol "equalized": every image, real or fake, gets the same processing: short side to `size`,
                          centre crop size x size, JPEG q90 4:2:0. Removes size and format shortcuts.
    """

    def __init__(self, items: list[dict], patch: int, protocol: str = "native", max_side: int = 1024, size: int = 512):
        self.items, self.patch, self.protocol, self.max_side, self.size = items, patch, protocol, max_side, size

    def __len__(self):
        return len(self.items)

    def _native(self, image: Image.Image, resample) -> Image.Image:
        scale = self.max_side / max(image.size)
        if scale < 1:
            image = image.resize((round(image.width * scale), round(image.height * scale)), resample)
        w, h = image.width // self.patch * self.patch, image.height // self.patch * self.patch
        left, top = (image.width - w) // 2, (image.height - h) // 2
        return image.crop((left, top, left + w, top + h))

    def _equalized(self, image: Image.Image, resample) -> Image.Image:
        scale = self.size / min(image.size)
        image = image.resize((max(self.size, round(image.width * scale)), max(self.size, round(image.height * scale))), resample)
        left, top = (image.width - self.size) // 2, (image.height - self.size) // 2
        return image.crop((left, top, left + self.size, top + self.size))

    def __getitem__(self, i):
        item = self.items[i]
        fit = self._native if self.protocol == "native" else self._equalized
        with Image.open(item["path"]) as im:
            image = fit(im.convert("RGB"), Image.BICUBIC)
        if self.protocol == "equalized":
            image = jpeg(image, 90, subsampling=2)
        out = {"image": to_tensor(image), "label": float(item["label"]), "has_mask": bool(item.get("mask"))}
        if item.get("mask"):
            with Image.open(item["mask"]) as m:
                mask = fit(m.convert("L"), Image.NEAREST)
            out["mask"] = torch.from_numpy(np.array(mask, dtype=np.float32) / 255)[None]
        else:
            out["mask"] = torch.zeros(1, image.height, image.width)
        return out
