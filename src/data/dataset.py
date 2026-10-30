"""TrainDataset: fixed-size views of real and generated images, with optional masks of generated regions.
ImageDataset: whole images for testing.

Pixels are never resized to fit the model: a training view is a size x size crop at the image's own resolution
(only the downscale augmentation changes the scale), a test image is used whole and scored tile by tile.
Every image is treated by the same random rules whatever its label, so the only thing that differs between the
classes is how the pixels were made. Files on disk are untouched originals: fakes mostly PNG, reals mostly
JPEG. So an image that is not stored as JPEG is first JPEG-compressed once (random quality in training, q90 in
testing), as if it had been published before: both classes then have a compression history. Every image is
then compressed once more (random JPEG / WebP in training, JPEG q90 in testing). In training the random crop
puts the 8 x 8 grids of the two compressions at a random offset.
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
    """rows: dicts with "path", "label" (0 real, 1 generated) and "mask" (path or empty). Every item is a
    size x size view, so images are batched as usual.

    train=True, in this order:
      read             one JPEG pass first if the file is not a JPEG (see the module docstring);
      down_p           augmentation before the crop: the image downscaled by 0.5-1, its short side kept >= size,
                       as images shared online;
      crop             a random size x size crop at that resolution: the fixed view, never resized;
      flip             horizontal, half of the views;
      degrade          resampling, blur, noise, then JPEG / WebP compression, always.
    train=False (validation): the centre size x size crop and JPEG q90, as in testing.
    Target map: the mask if the image has one (only that region is generated), else the label everywhere;
    it goes through the same resize, crop and flip as the image.
    """

    def __init__(self, rows: list[dict], size: int = 512, seed: int = 0, down_p: float = 0.3, train: bool = True):
        self.rows, self.size, self.seed, self.down_p, self.train, self.epoch = rows, size, seed, down_p, train, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        row, size = self.rows[i], self.size
        image = load(row["path"], rng if self.train else None)
        if row.get("mask"):
            with Image.open(row["mask"]) as m:
                target = m.convert("L").resize(image.size, Image.NEAREST)
        else:
            target = Image.new("L", image.size, 255 * int(row["label"]))

        short = min(image.size)
        s = size / short if short < size else 1.0  # smaller images are skipped at download; this is a guard
        if self.train and rng.random() < self.down_p:
            s = rng.uniform(max(0.5, size / short), 1.0)
        if s != 1.0:
            new = (max(size, round(image.width * s)), max(size, round(image.height * s)))
            image, target = image.resize(new, RESAMPLE[rng.integers(3)]), target.resize(new, Image.NEAREST)

        if self.train:
            x, y = int(rng.integers(0, image.width - size + 1)), int(rng.integers(0, image.height - size + 1))
        else:
            x, y = (image.width - size) // 2, (image.height - size) // 2
        image, target = image.crop((x, y, x + size, y + size)), target.crop((x, y, x + size, y + size))
        if self.train and rng.random() < 0.5:
            image, target = image.transpose(Image.FLIP_LEFT_RIGHT), target.transpose(Image.FLIP_LEFT_RIGHT)
        image = degrade(image, rng) if self.train else compress(image, "JPEG", 90)

        target = torch.from_numpy(np.array(target, dtype=np.float32) / 255)[None]
        return {"image": to_tensor(image), "target": target, "label": float(row["label"])}


class ImageDataset(Dataset):
    """items: dicts with "path", "label" and optionally "mask" (PNG, white = generated).

    The whole image at its own resolution, as in training: nothing is cropped or resized but the patch grid
    (at most patch - 1 px at the right and bottom). An image not stored as JPEG is JPEG-compressed once first
    (q90), then every image gets JPEG q90 4:2:0, as every training image is compressed. Score it with
    Detector.full, which looks at it in tiles of the training view size.
    """

    def __init__(self, items: list[dict], patch: int):
        self.items, self.patch = items, patch

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        image = load(item["path"])
        size = image.size
        image = compress(to_grid(image, self.patch), "JPEG", 90)
        out = {"image": to_tensor(image), "label": float(item["label"]), "has_mask": bool(item.get("mask"))}
        if item.get("mask"):
            with Image.open(item["mask"]) as m:
                mask = to_grid(m.convert("L").resize(size, Image.NEAREST), self.patch)
            out["mask"] = torch.from_numpy(np.array(mask, dtype=np.float32) / 255)[None]
        else:
            out["mask"] = torch.zeros(1, image.height, image.width)
        return out
