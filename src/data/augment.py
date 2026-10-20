"""Degradations that images pick up on the way through the internet.

Applied to the composed training image (and its target map where geometry changes).
Everything works on PIL images and numpy masks, no torchvision needed.
"""

import io

import numpy as np
from PIL import Image, ImageFilter

DEFAULT_DEGRADE = {
    "jpeg_p": 0.7,
    "jpeg_quality": [40, 95],
    "resize_p": 0.5,
    "resize_range": [0.5, 1.0],
    "blur_p": 0.2,
    "blur_sigma": [0.3, 1.5],
    "launder_p": 0.3,
}


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resize_pair(image: Image.Image, mask: np.ndarray | None, scale: float):
    w, h = image.size
    new_size = (max(16, int(round(w * scale))), max(16, int(round(h * scale))))
    image = image.resize(new_size, Image.BICUBIC)
    if mask is not None:
        mask = np.asarray(Image.fromarray(mask).resize(new_size, Image.BILINEAR), dtype=np.float32)
    return image, mask


class Degrade:
    """Random chain: JPEG, down-scaling, blur, and "laundering" (down, JPEG, up, JPEG)."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = {**DEFAULT_DEGRADE, **(cfg or {})}

    def __call__(self, image: Image.Image, mask: np.ndarray | None, rng: np.random.Generator):
        c = self.cfg

        if rng.random() < c["launder_p"]:
            w, h = image.size
            down = rng.uniform(0.4, 0.8)
            image, mask = resize_pair(image, mask, down)
            image = jpeg_roundtrip(image, rng.integers(50, 90))
            image = jpeg_roundtrip(image.resize((w, h), Image.BICUBIC), rng.integers(50, 90))
            if mask is not None:
                mask = np.asarray(Image.fromarray(mask).resize((w, h), Image.BILINEAR), dtype=np.float32)

        if rng.random() < c["resize_p"]:
            image, mask = resize_pair(image, mask, rng.uniform(*c["resize_range"]))

        if rng.random() < c["blur_p"]:
            image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(*c["blur_sigma"]))))

        if rng.random() < c["jpeg_p"]:
            image = jpeg_roundtrip(image, rng.integers(c["jpeg_quality"][0], c["jpeg_quality"][1] + 1))

        return image, mask
