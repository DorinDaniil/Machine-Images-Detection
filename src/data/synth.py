"""Aligned negatives: fakes made from a real image and its re-decoded copy.

A re-decoded copy is decode(encode(real)) through a generator's VAE: same content,
only the decoder trace differs. From one aligned pair we build three kinds of samples:

    real   the original, target map = 0 everywhere
    full   the re-decoded copy blended with ratio alpha, target = alpha everywhere
    local  re-decoded pixels pasted into a random region, original pixels kept outside,
           target = alpha inside the region, 0 outside (INP-X style: no global VAE trace)
"""

import numpy as np
from PIL import Image, ImageFilter

MODES = ("real", "full", "local")


def sample_mode(rng: np.random.Generator, p_real: float, p_full: float, p_local: float) -> str:
    probs = np.array([p_real, p_full, p_local], dtype=np.float64)
    return str(rng.choice(MODES, p=probs / probs.sum()))


def _blob(h: int, w: int, rng: np.random.Generator, area: float) -> np.ndarray:
    """Smooth random shape: low-frequency noise thresholded to the requested area."""
    coarse = rng.normal(size=(8, 8)).astype(np.float32)
    noise = np.asarray(Image.fromarray(coarse).resize((w, h), Image.BICUBIC), dtype=np.float32)
    threshold = np.quantile(noise, 1 - area)
    return (noise >= threshold).astype(np.float32)


def _rect(h: int, w: int, rng: np.random.Generator, area: float) -> np.ndarray:
    aspect = rng.uniform(0.5, 2.0)
    rh = int(np.clip(np.sqrt(area * h * w / aspect), 8, h))
    rw = int(np.clip(rh * aspect, 8, w))
    y, x = rng.integers(0, h - rh + 1), rng.integers(0, w - rw + 1)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y : y + rh, x : x + rw] = 1.0
    return mask


def _ellipse(h: int, w: int, rng: np.random.Generator, area: float) -> np.ndarray:
    aspect = rng.uniform(0.5, 2.0)
    ry = np.clip(np.sqrt(area * h * w / (np.pi * aspect)), 4, h / 2)
    rx = np.clip(ry * aspect, 4, w / 2)
    cy, cx = rng.uniform(ry, h - ry), rng.uniform(rx, w - rx)
    yy, xx = np.mgrid[0:h, 0:w]
    return ((((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1).astype(np.float32)


SHAPES = (_blob, _rect, _ellipse)


def random_region_mask(
    h: int,
    w: int,
    rng: np.random.Generator,
    area_range: tuple[float, float] = (0.05, 0.6),
    feather: float = 0.0,
) -> np.ndarray:
    """Union of 1-3 random shapes covering a fraction of the image in `area_range`."""
    target_area = rng.uniform(*area_range)
    n_shapes = int(rng.integers(1, 4))
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(n_shapes):
        shape = SHAPES[rng.integers(len(SHAPES))]
        mask = np.maximum(mask, shape(h, w, rng, target_area / n_shapes))
    if feather > 0:
        blurred = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(feather))
        mask = np.asarray(blurred, dtype=np.float32) / 255.0
    return mask


def compose(real: np.ndarray, fake: np.ndarray, mask: np.ndarray, alpha: float):
    """Blend `fake` into `real` inside `mask` with ratio `alpha`.

    real, fake: uint8 [H, W, 3] of the same size.  mask: float [H, W] in [0, 1].
    Returns (uint8 image, float32 target map = mask * alpha).
    """
    weight = (mask * alpha).astype(np.float32)
    out = real.astype(np.float32) * (1 - weight[..., None]) + fake.astype(np.float32) * weight[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), weight


def make_sample(
    real: np.ndarray,
    fake: np.ndarray | None,
    mode: str,
    rng: np.random.Generator,
    alpha_range: tuple[float, float] = (0.3, 1.0),
    area_range: tuple[float, float] = (0.05, 0.6),
    feather_p: float = 0.5,
):
    """Build one training sample from an aligned pair. Returns (uint8 image, float32 target)."""
    h, w = real.shape[:2]
    if mode == "real":
        return real, np.zeros((h, w), dtype=np.float32)

    alpha = float(rng.uniform(*alpha_range))
    if mode == "full":
        mask = np.ones((h, w), dtype=np.float32)
    else:
        feather = float(rng.uniform(1, 6)) if rng.random() < feather_p else 0.0
        mask = random_region_mask(h, w, rng, area_range, feather)
    return compose(real, fake, mask, alpha)
