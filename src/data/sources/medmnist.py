"""MedMNIST v2 / MedMNIST+ 2D subsets (Yang et al., 2023), converted from .npz to PNG.

Files come from the official Zenodo record 10519652 (MedMNIST+), which offers every 2D subset
at 28, 64, 128 and 224 px (square, center-cropped and resized by the MedMNIST authors). The HF
copy `albertvillanova/medmnist-v2` only has the 28 px version, so it is not used.

Default: size 224 and four subsets that span modalities without the multi-GB ones:
bloodmnist (blood-smear microscopy, 1.5 GB at 224), dermamnist (dermatoscope photos,
1.1 GB), retinamnist (fundus, 128 MB), breastmnist (ultrasound, 31 MB). Others: pathmnist
(12.6 GB at 224), octmnist, chestmnist, tissuemnist, organ{a,c,s}mnist, pneumoniamnist
(pneumoniamnist duplicates the chest_xray source). Each .npz is downloaded whole, so
`max_items` only limits the PNG export (split evenly across subsets, deterministic subset of
train+val+test); the .npz files are kept in npz/ and can be deleted afterwards.
A `.done` marker per subset skips re-export; delete the subset folder to export another cap.

Layout: raw_root/medmnist/npz/<subset>_<size>.npz, raw_root/medmnist/<subset>_<size>/<part>_<idx>.png

License: CC BY 4.0 (each subset also inherits its source dataset's terms). All images were
resampled by MedMNIST, so they carry interpolation traces rather than camera/scanner traces.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image

from ._real_utils import download_url, image_rows
from .base import Source, register, take

ZENODO = "https://zenodo.org/api/records/10519652/files/{}/content"
PARTS = ("train", "val", "test")


@register
class MedMnist(Source):
    name = "medmnist"
    license = "CC BY 4.0"
    url = "https://zenodo.org/records/10519652"
    description = "MedMNIST+ 2D subsets (28/64/128/224 px, default 224) from Zenodo, npz -> PNG"

    def __init__(
        self,
        split: str = "train",
        subsets: tuple[str, ...] = ("bloodmnist", "dermamnist", "retinamnist", "breastmnist"),
        size: int = 224,
    ):
        self.split = split
        self.subsets = subsets
        self.size = size

    def _export(self, npz_path: Path, out_dir: Path, quota: int | None) -> int:
        out_dir.mkdir(parents=True, exist_ok=True)
        with np.load(npz_path) as data:
            pairs = [(part, i) for part in PARTS for i in range(len(data[f"{part}_labels"]))]
            chosen = sorted(take(pairs, quota))
            for part in PARTS:
                indices = [i for p, i in chosen if p == part]
                if not indices:
                    continue
                images = data[f"{part}_images"]
                for i in indices:
                    target = out_dir / f"{part}_{i:06d}.png"
                    if not target.exists():
                        Image.fromarray(images[i]).save(target)
        return len(chosen)

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        root = Path(raw_root) / self.name
        quota = None if max_items is None else math.ceil(max_items / len(self.subsets))
        for subset in self.subsets:
            stem = subset if self.size == 28 else f"{subset}_{self.size}"
            out_dir = root / stem
            marker = out_dir / ".done"
            if marker.exists():
                continue
            npz_path = download_url(ZENODO.format(f"{stem}.npz"), root / "npz" / f"{stem}.npz")
            count = self._export(npz_path, out_dir, quota)
            marker.write_text(str(count))
            print(f"[{self.name}] {stem}: {count} PNGs")

    def rows(self, raw_root: Path) -> list[dict]:
        root = Path(raw_root) / self.name
        rows = []
        for folder in sorted(p for p in root.glob("*mnist*") if p.is_dir()):
            rows += image_rows(raw_root, folder, "biomed", self.name, self.split)
        return rows
