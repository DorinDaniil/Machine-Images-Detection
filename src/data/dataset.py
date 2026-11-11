"""ImageDataset: real and generated images as size x size views.

Every image, whatever its size and shape, is resized whole to size x size (aspect not kept), as usual for vision
models: the model sees every picture at one scale and nothing is cut off at test time. Every image is treated
by the same random rules whatever its label, so the only thing that differs between the classes is how the
pixels were made. Files on disk are untouched originals: fakes mostly PNG, reals mostly JPEG. So an image that
is not stored as JPEG is first JPEG-compressed once (random quality in training, q90 otherwise), as if it had
been published before: both classes then have a compression history. In training the augmentations come
before the resize, at the image's own resolution, as images get them on the way through the internet before
anyone resizes them for a model.
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


def random_crop(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """A random part of the image: 50-100 % of its area, aspect from 1:2 to 2:1 whatever the image's own
    aspect. So after the resize to a square the stretch is the same for both classes: generated images are
    often square and photos are not, and without this the stretch alone would give the label away."""
    area = image.width * image.height * rng.uniform(0.5, 1.0)
    aspect = np.exp(rng.uniform(-np.log(2), np.log(2)))
    w, h = min(image.width, round((area * aspect) ** 0.5)), min(image.height, round((area / aspect) ** 0.5))
    x, y = int(rng.integers(0, image.width - w + 1)), int(rng.integers(0, image.height - h + 1))
    return image.crop((x, y, x + w, y + h))


class ImageDataset(Dataset):
    """rows: dicts with "path" and "label". Item: image [3, size, size] in [0, 1] and the label.

    train=False (validation, testing): the whole image resized to size x size (bicubic).
    train=True, in this order:
      read      one JPEG pass first if the file is not a JPEG (see the module docstring);
      crop      a random part of the image (random_crop);
      degrade   at the image's own resolution (above MAX_SIDE shrunk to it first): resampling, blur, noise,
                then JPEG / WebP compression, always;
      resize    to size x size, aspect not kept, with a random resampling filter;
      flip      horizontal, half of the views.
    """

    def __init__(self, rows: list[dict], size: int = 512, train: bool = False, seed: int = 0):
        self.rows, self.size, self.train, self.seed, self.epoch = rows, size, train, seed, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row, size = self.rows[i], (self.size, self.size)
        if not self.train:
            return to_tensor(load(row["path"]).resize(size, Image.BICUBIC)), row["label"]
        rng = np.random.default_rng([self.seed, self.epoch, i])
        image = random_crop(load(row["path"], rng), rng)
        s = MAX_SIDE / max(image.size)
        if s < 1:
            image = image.resize((round(image.width * s), round(image.height * s)), RESAMPLE[rng.integers(3)])
        image = degrade(image, rng).resize(size, RESAMPLE[rng.integers(3)])
        if rng.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return to_tensor(image), row["label"]
