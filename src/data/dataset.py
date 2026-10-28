"""TrainDataset: real and generated images with optional masks of generated regions.
ImageDataset: the same for testing, under a fixed protocol.

Images are used whole at their own resolution. Every image is treated by the same random rules whatever its
label, so the only thing that differs between the classes is how the pixels were made. Files on disk are
untouched originals: fakes mostly PNG, reals mostly JPEG. So an image that is not stored as JPEG is first
JPEG-compressed once (random quality in training, q90 in testing), as if it had been published before: both
classes then have a compression history, and compressing again later leaves double-compression traces in
both. Every image is then compressed once more (random JPEG / WebP in training, JPEG q90 in testing). In
training a random 0-7 px shift keeps the 8 x 8 JPEG grids of the two compressions from always lining up.
"""

import io
import warnings

import numpy as np
import torch
from PIL import Image, ImageFilter, features
from torch.utils.data import Dataset

RESAMPLE = (Image.BILINEAR, Image.BICUBIC, Image.LANCZOS)
WEBP = features.check("webp")
warnings.filterwarnings("ignore", message="Palette images with Transparency")  # harmless, floods the training log
warnings.filterwarnings("ignore", message="Corrupt EXIF data")


def compress(image: Image.Image, fmt: str = "JPEG", quality: int = 90, subsampling: int = 2) -> Image.Image:
    buf = io.BytesIO()
    if fmt == "JPEG":
        image.save(buf, format="JPEG", quality=int(quality), subsampling=int(subsampling))
    else:
        image.save(buf, format=fmt, quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as im:
        return im.convert("RGB")


def degrade(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """What images get on the way through the internet, size kept. Same probabilities for real and fake."""
    if rng.random() < 0.2:  # resampled and brought back to the same size
        s = rng.uniform(0.5, 0.9) if rng.random() < 0.7 else rng.uniform(1.2, 2.0)
        size = image.size
        image = image.resize((max(1, round(size[0] * s)), max(1, round(size[1] * s))), RESAMPLE[rng.integers(3)])
        image = image.resize(size, RESAMPLE[rng.integers(3)])
    if rng.random() < 0.1:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    if rng.random() < 0.2:  # sensor-like noise, so "noisy = real" is not a shortcut
        x = np.asarray(image, dtype=np.float32) + rng.normal(0, rng.uniform(0.5, 4), (image.height, image.width, 1))
        image = Image.fromarray(np.clip(x.round(), 0, 255).astype(np.uint8))
    # compression, always (see the module docstring): JPEG or WebP, sometimes a second JPEG as after re-sharing
    if WEBP and rng.random() < 0.2:
        image = compress(image, "WEBP", rng.integers(60, 96))
    else:
        image = compress(image, "JPEG", rng.integers(60, 96), subsampling=rng.choice([0, 2]))
    if rng.random() < 0.25:
        image = compress(image, "JPEG", rng.integers(70, 96))
    return image


def load(path, rng: np.random.Generator | None = None) -> Image.Image:
    """RGB image; one JPEG pass (random quality, or q90 without rng) if the file is not a JPEG."""
    with Image.open(path) as im:
        fmt, image = im.format, im.convert("RGB")
    if fmt != "JPEG":
        image = compress(image, "JPEG", 90 if rng is None else rng.integers(75, 99), subsampling=2)
    return image


def to_tensor(image) -> torch.Tensor:
    return torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)


def to_grid(image: Image.Image, patch: int) -> Image.Image:
    """Cuts the right and bottom edge to the patch grid: at most patch - 1 px."""
    return image.crop((0, 0, image.width // patch * patch, image.height // patch * patch))


class TrainDataset(Dataset):
    """rows: dicts with "path", "label" (0 real, 1 generated) and "mask" (path or empty).

    View: the whole image at its own resolution, first JPEG-compressed once if not stored as JPEG (see the
    module docstring) and shifted by a random 0-7 px. Augmentations, same for both classes:
      crop_p  a random part of the image, each side 40-100 % (at least 256 px), never resized;
      down_p  the view downscaled by 0.5-1 and kept smaller (at least 256 px), as images shared online;
      a horizontal flip; then `degrade`: resampling, blur, noise, JPEG / WebP compression.
    Two cuts that are not augmentation: a side above max_side is cut to max_side by a random crop (memory;
    only large photos reach it, and a crop of a real photo is still real, of a mask still its mask) and both
    sides are cut to the patch grid.
    Target map: the mask if the image has one (only that region is generated), else the label everywhere;
    it goes through the same crop, resize and flip as the image.
    Sizes differ from image to image, so the loader gives one image at a time (see train.py).
    """

    def __init__(self, rows: list[dict], max_side: int = 1024, patch: int = 16, seed: int = 0,
                 crop_p: float = 0.5, down_p: float = 0.3):
        self.rows, self.max_side, self.patch, self.seed = rows, max_side, patch, seed
        self.crop_p, self.down_p, self.epoch = crop_p, down_p, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        row = self.rows[i]
        image = load(row["path"], rng)
        if row.get("mask"):
            with Image.open(row["mask"]) as m:
                target = m.convert("L").resize(image.size, Image.NEAREST)
        else:
            target = Image.new("L", image.size, 255 * int(row["label"]))
        dx, dy = (int(v) for v in rng.integers(0, 8, 2))  # the JPEG grid at a random offset
        box = (dx, dy, image.width, image.height)
        image, target = image.crop(box), target.crop(box)

        w, h = min(image.width, self.max_side), min(image.height, self.max_side)
        if rng.random() < self.crop_p:
            w = int(rng.uniform(min(w, max(256, 0.4 * w)), w + 1))
            h = int(rng.uniform(min(h, max(256, 0.4 * h)), h + 1))
        x, y = int(rng.integers(0, image.width - w + 1)), int(rng.integers(0, image.height - h + 1))
        image, target = image.crop((x, y, x + w, y + h)), target.crop((x, y, x + w, y + h))
        if rng.random() < self.down_p:
            s = rng.uniform(0.5, 1.0)
            size = (min(w, max(256, round(w * s))), min(h, max(256, round(h * s))))
            image, target = image.resize(size, RESAMPLE[rng.integers(3)]), target.resize(size, Image.NEAREST)
        if rng.random() < 0.5:
            image, target = image.transpose(Image.FLIP_LEFT_RIGHT), target.transpose(Image.FLIP_LEFT_RIGHT)
        image, target = to_grid(image, self.patch), to_grid(target, self.patch)

        target = torch.from_numpy(np.array(target, dtype=np.float32) / 255)[None]
        return {"image": to_tensor(degrade(image, rng)), "target": target, "label": float(row["label"])}


class ImageDataset(Dataset):
    """items: dicts with "path", "label" and optionally "mask" (PNG, white = generated).

    protocol "native":    the whole image at its own resolution, as in training: nothing is cropped or resized
                          but the patch grid (at most patch - 1 px at the right and bottom). Score it with
                          Detector.full, which looks at large images tile by tile.
    protocol "equalized": the whole image resized to `area` pixels, proportions kept: every image has the
                          same size, a check that the model does not rely on size or resolution.
    An image not stored as JPEG is JPEG-compressed once first (q90), as in training; both protocols end with
    JPEG q90 4:2:0, since every training image is compressed. No shift: nothing is cut but the patch grid.
    """

    def __init__(self, items: list[dict], patch: int, protocol: str = "native", area: int = 512 * 512):
        self.items, self.patch, self.protocol, self.area = items, patch, protocol, area

    def __len__(self):
        return len(self.items)

    def _fit(self, image: Image.Image, resample) -> Image.Image:
        if self.protocol == "equalized":
            p, s = self.patch, (self.area / (image.width * image.height)) ** 0.5
            image = image.resize((max(p, round(image.width * s / p) * p), max(p, round(image.height * s / p) * p)), resample)
        return to_grid(image, self.patch)

    def __getitem__(self, i):
        item = self.items[i]
        image = load(item["path"])
        size = image.size
        image = compress(self._fit(image, Image.BICUBIC), "JPEG", 90)
        out = {"image": to_tensor(image), "label": float(item["label"]), "has_mask": bool(item.get("mask"))}
        if item.get("mask"):
            with Image.open(item["mask"]) as m:
                mask = self._fit(m.convert("L").resize(size, Image.NEAREST), Image.NEAREST)
            out["mask"] = torch.from_numpy(np.array(mask, dtype=np.float32) / 255)[None]
        else:
            out["mask"] = torch.zeros(1, image.height, image.width)
        return out
