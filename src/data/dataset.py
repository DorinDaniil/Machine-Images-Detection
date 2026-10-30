"""TrainDataset: size x size views of real and generated images, with optional masks of generated regions.
ImageDataset: the same view of whole images for testing.

Every image, whatever its size and shape, is resized whole to size x size (aspect not kept), as usual for vision
models: the model sees every picture at one scale and nothing is cut off at test time. Every image is treated
by the same random rules whatever its label, so the only thing that differs between the classes is how the
pixels were made. Files on disk are untouched originals: fakes mostly PNG, reals mostly JPEG. So an image that
is not stored as JPEG is first JPEG-compressed once (random quality in training, q90 in testing), as if it had
been published before: both classes then have a compression history. In training the augmentations come
before the resize, at the image's own resolution, as images get them on the way through the internet before
anyone resizes them for a model; a test image is resized as it is.
"""

import io
import warnings

import numpy as np
import torch
from PIL import Image, ImageFilter, features
from torch.utils.data import Dataset

RESAMPLE = (Image.BILINEAR, Image.BICUBIC, Image.LANCZOS)
MAX_SIDE = 2048  # training: larger images are shrunk to this before the degradations, as social networks do
WEBP = features.check("webp")
warnings.filterwarnings("ignore", message="Palette images with Transparency")  # harmless, floods the training log
warnings.filterwarnings("ignore", message="Corrupt EXIF data")


def compress(image: Image.Image, fmt: str = "JPEG", quality: int = 90, subsampling: int = 2) -> Image.Image:
    buf = io.BytesIO()
    if fmt == "JPEG":
        image.save(buf, format="JPEG", quality=int(quality), subsampling=int(subsampling))
    else:
        image.save(buf, format=fmt, quality=int(quality), method=0)  # WebP: the fastest encoder setting
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
        noise = rng.standard_normal((image.height, image.width, 1), dtype=np.float32) * rng.uniform(0.5, 4)
        x = np.asarray(image, dtype=np.float32) + noise
        image = Image.fromarray(np.clip(x.round(), 0, 255).astype(np.uint8))
    # compression, always: JPEG or WebP, sometimes a second JPEG as after re-sharing
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


class TrainDataset(Dataset):
    """rows: dicts with "path", "label" (0 real, 1 generated) and "mask" (path or empty). Every item is a
    size x size view, so images are batched as usual.

    train=True, in this order:
      read             one JPEG pass first if the file is not a JPEG (see the module docstring);
      crop_p           augmentation: a random part of 50-100 % of the image, aspect 3:4-4:3 (as RandomResizedCrop
                       on ImageNet); the other views are the whole image;
      degrade          at the image's own resolution (above MAX_SIDE shrunk to it first): resampling, blur, noise,
                       then JPEG / WebP compression, always;
      resize           to size x size, aspect not kept, with a random resampling filter;
      flip             horizontal, half of the views.
    train=False (validation): the whole image resized to size x size (bicubic), as in testing.
    Target map: the mask if the image has one (only that region is generated), else the label everywhere;
    it goes through the same crop, resize and flip as the image.
    """

    def __init__(self, rows: list[dict], size: int = 512, seed: int = 0, crop_p: float = 0.5, train: bool = True):
        self.rows, self.size, self.seed, self.crop_p, self.train, self.epoch = rows, size, seed, crop_p, train, 0

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

        if self.train and rng.random() < self.crop_p:
            area = image.width * image.height * rng.uniform(0.5, 1.0)
            aspect = np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3)))
            w, h = min(image.width, round((area * aspect) ** 0.5)), min(image.height, round((area / aspect) ** 0.5))
            x, y = int(rng.integers(0, image.width - w + 1)), int(rng.integers(0, image.height - h + 1))
            image, target = image.crop((x, y, x + w, y + h)), target.crop((x, y, x + w, y + h))
        if self.train:
            s = MAX_SIDE / max(image.size)
            if s < 1:  # the target needs no resize here: below it goes to size x size over the same extent
                image = image.resize((round(image.width * s), round(image.height * s)), RESAMPLE[rng.integers(3)])
            image = degrade(image, rng)
        resample = RESAMPLE[rng.integers(3)] if self.train else Image.BICUBIC
        image, target = image.resize((size, size), resample), target.resize((size, size), Image.NEAREST)
        if self.train and rng.random() < 0.5:
            image, target = image.transpose(Image.FLIP_LEFT_RIGHT), target.transpose(Image.FLIP_LEFT_RIGHT)

        target = torch.from_numpy(np.array(target, dtype=np.float32) / 255)[None]
        return {"image": to_tensor(image), "target": target, "label": float(row["label"])}


class ImageDataset(Dataset):
    """items: dicts with "path", "label" and optionally "mask" (PNG, white = generated).

    The whole image resized to size x size (bicubic), as the validation views in training: nothing is cut off.
    An image not stored as JPEG is JPEG-compressed once first (q90), see the module docstring.
    """

    def __init__(self, items: list[dict], size: int = 512):
        self.items, self.size = items, size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item, size = self.items[i], (self.size, self.size)
        image = load(item["path"]).resize(size, Image.BICUBIC)
        out = {"image": to_tensor(image), "label": float(item["label"]), "has_mask": bool(item.get("mask"))}
        if item.get("mask"):
            with Image.open(item["mask"]) as m:
                mask = m.convert("L").resize(size, Image.NEAREST)
            out["mask"] = torch.from_numpy(np.array(mask, dtype=np.float32) / 255)[None]
        else:
            out["mask"] = torch.zeros(1, *size)
        return out
