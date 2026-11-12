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
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps, features
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


def censor(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Something hidden before posting, as over a face, a plate or a name: a strong blur, big pixels or a flat
    box over a region of 2-20 % of the image, or a thin bar of 1-6 % as over the eyes. The region is a rectangle,
    an ellipse or a smooth blob. After the fragment edits of Evidential-Copy-Detection (overlays.py)."""
    kind = rng.choice(["blur", "pixels", "box", "bar"])
    area, aspect = (rng.uniform(0.01, 0.06), rng.uniform(3, 7)) if kind == "bar" else (rng.uniform(0.02, 0.2), rng.uniform(0.6, 1.7))
    w = int(min(0.9 * image.width, max(8, (image.width * image.height * area * aspect) ** 0.5)))
    h = int(min(0.9 * image.height, max(8, image.width * image.height * area / w)))
    x, y = int(rng.integers(0, image.width - w + 1)), int(rng.integers(0, image.height - h + 1))
    region = image.crop((x, y, x + w, y + h))
    if kind == "blur":
        region = region.filter(ImageFilter.GaussianBlur(max(3.0, min(w, h) * rng.uniform(0.15, 0.35))))
    elif kind == "pixels":
        block = max(3, int(min(w, h) * rng.uniform(0.1, 0.25)))
        region = region.resize((max(1, w // block), max(1, h // block)), Image.BILINEAR).resize((w, h), Image.NEAREST)
    else:  # mostly black, sometimes white or any colour
        r = rng.random()
        colour = (0, 0, 0) if r < 0.7 else (255, 255, 255) if r < 0.85 else tuple(int(v) for v in rng.integers(0, 256, 3))
        region = Image.new("RGB", (w, h), colour)

    shape = "rectangle" if kind == "bar" else rng.choice(["rectangle", "ellipse", "blob"])
    mask = Image.new("L", (w, h), 255 if shape == "rectangle" else 0)
    if shape == "ellipse":
        ImageDraw.Draw(mask).ellipse((0, 0, w - 1, h - 1), fill=255)
    elif shape == "blob":  # an ellipse whose radius wobbles: a few random sine waves around it
        angle = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        radius = 1 + sum(rng.uniform(0.06, 0.28) * np.sin(rng.integers(2, 8) * angle + rng.uniform(0, 2 * np.pi))
                         for _ in range(rng.integers(2, 6)))
        radius = np.clip(radius, 0.45, 1.0)
        outline = zip(w / 2 * (1 + radius * np.cos(angle)), h / 2 * (1 + radius * np.sin(angle)))
        ImageDraw.Draw(mask).polygon([(float(px), float(py)) for px, py in outline], fill=255)
    if kind == "blur":  # a blur fades out at its edge
        mask = mask.filter(ImageFilter.GaussianBlur(min(w, h) * rng.uniform(0.02, 0.08)))
    image = image.copy()
    image.paste(region, (x, y), mask)
    return image


def degrade(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """What images get on the way through the internet, size kept. Same probabilities for real and fake.
    First what an author does (colour, sharpening, something censored), then what sharing does
    (resampling, blur, noise, compression). About a third of the views get nothing but the compression."""
    if rng.random() < 0.25:  # filters and auto-correction: brightness, contrast, saturation
        for enhance in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            image = enhance(image).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.03:
        image = ImageOps.grayscale(image).convert("RGB")
    if rng.random() < 0.1:  # sharpening, as phone cameras and many resizers do
        image = image.filter(ImageFilter.UnsharpMask(rng.uniform(1, 2), int(rng.integers(50, 150)), 2))
    if rng.random() < 0.1:
        image = censor(image, rng)
    if rng.random() < 0.25:  # resampled and brought back to the same size
        s = rng.uniform(0.5, 0.9) if rng.random() < 0.7 else rng.uniform(1.2, 2.0)
        size = image.size
        image = image.resize((max(1, round(size[0] * s)), max(1, round(size[1] * s))), RESAMPLE[rng.integers(3)])
        image = image.resize(size, RESAMPLE[rng.integers(3)])
    if rng.random() < 0.15:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
    if rng.random() < 0.2:  # sensor-like noise, so "noisy = real" is not a shortcut
        noise = rng.standard_normal((image.height, image.width, 1), dtype=np.float32) * rng.uniform(0.5, 4)
        x = np.asarray(image, dtype=np.float32) + noise
        image = Image.fromarray(np.clip(x.round(), 0, 255).astype(np.uint8))
    # compression, always: JPEG or WebP, sometimes a second JPEG as after re-sharing
    if WEBP and rng.random() < 0.2:
        image = compress(image, "WEBP", rng.integers(50, 96))
    else:
        image = compress(image, "JPEG", rng.integers(50, 96), subsampling=rng.choice([0, 2]))
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
      degrade   at the image's own resolution (above MAX_SIDE shrunk to it first): colour, sharpening, a censored
                region, resampling, blur, noise, then JPEG / WebP compression, always;
      resize    to size x size, aspect not kept, with a random resampling filter;
      flip      horizontal, half of the views; a turn by 90, 180 or 270 degrees, 5 % of the views.
    A file that cannot be read is reported and the next image is returned instead of it.
    """

    def __init__(self, rows: list[dict], size: int = 512, train: bool = False, seed: int = 0):
        self.rows, self.size, self.train, self.seed, self.epoch = rows, size, train, seed, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        try:
            return self.view(i), self.rows[i]["label"]
        except Exception as e:  # one broken file must not stop hours of training
            print(f"unreadable, skipped: {self.rows[i]['path']} ({e!r})")
            return self[(i + 1) % len(self)]

    def view(self, i) -> torch.Tensor:
        row, size = self.rows[i], (self.size, self.size)
        if not self.train:
            return to_tensor(load(row["path"]).resize(size, Image.BICUBIC))
        rng = np.random.default_rng([self.seed, self.epoch, i])
        image = random_crop(load(row["path"], rng), rng)
        s = MAX_SIDE / max(image.size)
        if s < 1:
            image = image.resize((round(image.width * s), round(image.height * s)), RESAMPLE[rng.integers(3)])
        image = degrade(image, rng).resize(size, RESAMPLE[rng.integers(3)])
        if rng.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if rng.random() < 0.05:
            image = image.transpose([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270][rng.integers(3)])
        return to_tensor(image)
