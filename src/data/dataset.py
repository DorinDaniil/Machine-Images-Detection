"""Datasets.

AlignedPairsDataset  training; composes real / full / local samples from aligned pairs
ManifestDataset      validation and testing; reads rows of a manifest CSV
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augment import Degrade
from .synth import make_sample, sample_mode

PAIR_COLUMNS = ["pair_dir", "decoders", "content_type", "source", "split"]


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """uint8 [H, W, 3] -> float [3, H, W] in [0, 1]."""
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0


def random_crop(image: np.ndarray, target: np.ndarray, size: int, rng: np.random.Generator):
    """Crop `size` x `size` at a random position. Upscales first if the image is smaller."""
    h, w = image.shape[:2]
    if min(h, w) < size:
        scale = size / min(h, w)
        new_size = (int(np.ceil(w * scale)), int(np.ceil(h * scale)))
        image = np.asarray(Image.fromarray(image).resize(new_size, Image.BICUBIC))
        target = np.asarray(Image.fromarray(target).resize(new_size, Image.BILINEAR), dtype=np.float32)
        h, w = image.shape[:2]
    y, x = rng.integers(0, h - size + 1), rng.integers(0, w - size + 1)
    return image[y : y + size, x : x + size], target[y : y + size, x : x + size]


def fit_to_multiple(image: Image.Image, max_side: int, multiple: int) -> Image.Image:
    """Shrink only if the longer side exceeds max_side, then center-crop to a multiple of the patch."""
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize((max(multiple, int(w * scale)), max(multiple, int(h * scale))), Image.BICUBIC)
        w, h = image.size
    cw, ch = max(multiple, w // multiple * multiple), max(multiple, h // multiple * multiple)
    left, top = (w - cw) // 2, (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch)) if (cw, ch) != (w, h) else image


class AlignedPairsDataset(Dataset):
    """Training set from `aligned_pairs.csv` written by src/data/prepare_pairs.py.

    Each pair directory holds real.png and <decoder>.png re-decoded copies of the same size.
    """

    def __init__(self, pairs_csv: str | Path, cfg: dict, split: str = "train", seed: int = 0):
        df = pd.read_csv(pairs_csv, dtype=str, keep_default_na=False)
        self.rows = df[df["split"] == split].reset_index(drop=True)
        if len(self.rows) == 0:
            raise ValueError(f"No rows with split={split} in {pairs_csv}")
        self.root = Path(pairs_csv).parent
        self.cfg = cfg
        self.degrade = Degrade(cfg.get("degrade"))
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def _open(self, pair_dir: Path, name: str) -> np.ndarray:
        return np.asarray(Image.open(pair_dir / f"{name}.png").convert("RGB"))

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        row = self.rows.iloc[idx]
        pair_dir = Path(row["pair_dir"])
        if not pair_dir.is_absolute():
            pair_dir = self.root / pair_dir
        decoders = [d for d in row["decoders"].split(";") if d]

        c = self.cfg
        mode = sample_mode(rng, c["p_real"], c["p_full"], c["p_local"])
        real = self._open(pair_dir, "real")
        fake = self._open(pair_dir, decoders[rng.integers(len(decoders))]) if mode != "real" else None

        image, target = make_sample(
            real, fake, mode, rng,
            alpha_range=tuple(c.get("alpha_range", (0.3, 1.0))),
            area_range=tuple(c.get("area_range", (0.05, 0.6))),
        )
        pil, target = self.degrade(Image.fromarray(image), target, rng)
        image, target = random_crop(np.asarray(pil), target, c["crop_size"], rng)

        return {
            "image": to_tensor(image),
            "target": torch.from_numpy(np.array(target, copy=True))[None],  # [1, H, W]
            "label": torch.tensor(float(target.max())),
            "mode": mode,
        }


class ManifestDataset(Dataset):
    """Rows of a manifest. Returns the image, the binary mask (if any) and the label.

    fixed_size: center-crop / pad to a square of this size (batchable, for validation).
    None:       keep native resolution, shrunk to max_side and floored to the patch multiple
                (batch size 1, for testing).
    """

    def __init__(self, df: pd.DataFrame, multiple: int, fixed_size: int | None = None, max_side: int = 1024):
        self.df = df.reset_index(drop=True)
        self.multiple = multiple
        self.fixed_size = fixed_size
        self.max_side = max_side

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L") if row["mask_path"] else None

        if self.fixed_size:
            image, mask = center_square(image, mask, self.fixed_size)
        else:
            image = fit_to_multiple(image, self.max_side, self.multiple)
            if mask is not None:
                mask = mask.resize(image.size, Image.NEAREST)

        out = {
            "image": to_tensor(np.asarray(image)),
            "label": torch.tensor(float(row["label"])),
            "index": idx,
            "has_mask": mask is not None,
        }
        h, w = out["image"].shape[1:]
        out["mask"] = (
            torch.from_numpy((np.asarray(mask) > 127).astype(np.float32))[None]
            if mask is not None
            else torch.zeros(1, h, w)
        )
        return out


def center_square(image: Image.Image, mask: Image.Image | None, size: int):
    """Resize the shorter side to `size` and center-crop a square; keeps traces mostly intact."""
    w, h = image.size
    scale = size / min(w, h)
    new_size = (max(size, int(round(w * scale))), max(size, int(round(h * scale))))
    image = image.resize(new_size, Image.BICUBIC)
    left, top = (new_size[0] - size) // 2, (new_size[1] - size) // 2
    box = (left, top, left + size, top + size)
    image = image.crop(box)
    if mask is not None:
        mask = mask.resize(new_size, Image.NEAREST).crop(box)
    return image, mask
