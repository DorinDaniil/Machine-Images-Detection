"""ROCOv2 radiology figures (Rückert et al., 2024): CT, MRI, X-ray, ultrasound, PET, angiography.

Images are figures cut from PubMed Central Open Access articles, i.e. exactly the kind of
radiology panel that appears in scientific papers (often with arrows, letters, scale bars).
HF dataset `eltorio/ROCOv2-radiology`: 79,789 images in 39 parquet shards (27 train,
6 validation, 6 test; ~2,200 images / 490 MB each, 18.6 GB total). Shards are fetched in a
fixed shuffled order, exported with their original bytes and deleted, until `max_items`
images passed `min_side` (default 512 px), so the cap is honoured to within one shard.

Resolution: mixed, from ~220 px to ~1600 px; in a 25-image probe 60% had a short side
>= 512 px (typical 500-800 px). JPEGs as published by PMC, often recompressed.

Layout: raw_root/rocov2/images/<shard>_<row>.jpg|png

License: CC BY-NC-SA 4.0 (per the HF card; the source articles carry CC BY or CC BY-NC
licenses). The PMC snapshot date was not verified (ROCOv2 was published in 2024), so a few
synthetic figures in recent articles cannot be ruled out. Figures were edited by the authors
(crops, arrows, contrast changes), which is ordinary editing, not generative AI.
"""

from pathlib import Path

from ._real_utils import fetch_hub_parquet, hub_files, image_rows
from .base import Source, register

REPO_ID = "eltorio/ROCOv2-radiology"


@register
class RocoV2(Source):
    name = "rocov2"
    license = "CC BY-NC-SA 4.0"
    url = "https://huggingface.co/datasets/eltorio/ROCOv2-radiology"
    description = "ROCOv2 radiology figures from PMC papers (mixed res), eltorio/ROCOv2-radiology"

    def __init__(self, split: str = "train", min_side: int = 512):
        self.split = split
        self.min_side = min_side

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        out_dir = Path(raw_root) / self.name / "images"
        files = hub_files(REPO_ID, ("data/",))
        fetch_hub_parquet(REPO_ID, files, out_dir, max_items, min_side=self.min_side)

    def rows(self, raw_root: Path) -> list[dict]:
        return image_rows(raw_root, Path(raw_root) / self.name / "images", "biomed", self.name, self.split)
