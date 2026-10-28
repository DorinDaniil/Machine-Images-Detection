"""TrainDataset: real and generated images with optional masks of generated regions.
ImageDataset: the same for testing, under a fixed protocol.

Every image is treated by the same random rules whatever its label, so the only thing that differs
between the classes is how the pixels were made. Files on disk are untouched originals (fakes mostly PNG,
reals mostly JPEG), so every training image is JPEG-compressed here at a random quality: the presence of
JPEG traces then says nothing about the label.
"""

import io

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset


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
    return jpeg(image, rng.integers(50, 96), subsampling=int(rng.choice([0, 2])))  # always, see module docstring


def to_tensor(image) -> torch.Tensor:
    return torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)


class TrainDataset(Dataset):
    """rows: dicts with "path", "label" (0 real, 1 generated) and "mask" (path or empty).

    View: the image at its own resolution, never resized. A side longer than max_side is cut to max_side by a
    random crop, and both sides are cut to the patch grid; then a random flip and degradations.
    Target map: the mask if the image has one (only that region is generated), else the label everywhere.
    Sizes differ from image to image, so the loader gives one image at a time (see train.py).
    """

    def __init__(self, rows: list[dict], max_side: int = 1024, patch: int = 16, seed: int = 0):
        self.rows, self.max_side, self.patch, self.seed, self.epoch = rows, max_side, patch, seed, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        row = self.rows[i]
        with Image.open(row["path"]) as im:
            image = im.convert("RGB")
        if row.get("mask"):
            with Image.open(row["mask"]) as m:
                target = m.convert("L")
        else:
            target = Image.new("L", image.size, 255 * int(row["label"]))

        w = min(image.width, self.max_side) // self.patch * self.patch
        h = min(image.height, self.max_side) // self.patch * self.patch
        x, y = int(rng.integers(0, image.width - w + 1)), int(rng.integers(0, image.height - h + 1))
        box = (x, y, x + w, y + h)
        image, target = image.crop(box), target.crop(box)
        if rng.random() < 0.5:
            image, target = image.transpose(Image.FLIP_LEFT_RIGHT), target.transpose(Image.FLIP_LEFT_RIGHT)

        target = torch.from_numpy(np.array(target, dtype=np.float32) / 255)[None]
        return {"image": to_tensor(degrade(image, rng)), "target": target, "label": float(row["label"])}


class ImageDataset(Dataset):
    """items: dicts with "path", "label" and optionally "mask" (PNG, white = generated).

    protocol "native":    own resolution, as in training: never resized, sides above max_side cut by a centre
                          crop, both sides cut to the patch grid.
    protocol "equalized": short side to `size`, centre crop size x size. Removes size and resolution
                          shortcuts.
    Both end with JPEG q90 4:2:0: in training every image is JPEG-compressed, so an image without JPEG
    traces (a PNG fake) would be outside what the model has seen.
    """

    def __init__(self, items: list[dict], patch: int, protocol: str = "native", max_side: int = 1024, size: int = 512):
        self.items, self.patch, self.protocol, self.max_side, self.size = items, patch, protocol, max_side, size

    def __len__(self):
        return len(self.items)

    def _native(self, image: Image.Image, resample) -> Image.Image:
        w = min(image.width, self.max_side) // self.patch * self.patch
        h = min(image.height, self.max_side) // self.patch * self.patch
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
            image = jpeg(fit(im.convert("RGB"), Image.BICUBIC), 90, subsampling=2)
        out = {"image": to_tensor(image), "label": float(item["label"]), "has_mask": bool(item.get("mask"))}
        if item.get("mask"):
            with Image.open(item["mask"]) as m:
                mask = fit(m.convert("L"), Image.NEAREST)
            out["mask"] = torch.from_numpy(np.array(mask, dtype=np.float32) / 255)[None]
        else:
            out["mask"] = torch.zeros(1, image.height, image.width)
        return out
