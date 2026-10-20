"""Pediatric chest X-rays (Kermany et al., Cell 2018) from `hf-vision/chest-xray-pneumonia`.

5,856 anterior-posterior radiographs (normal / pneumonia) from Guangzhou Women and Children's
Medical Center, exported by the authors as JPEG. HF stores them in 9 parquet shards (7 train,
validation, test; 1.2 GB total). Shards are fetched in a fixed shuffled order, exported with
their original JPEG bytes and deleted, until `max_items` images passed `min_side` (default
512 px), so the cap is honoured to within one shard. Shard sizes are uneven (3-450 MB).

Resolution: mostly 1000-2100 px on the long side (e.g. 1858x2090, 1422x1152); nearly all pass
the 512 px filter. Grayscale content stored as 8-bit JPEG.

An adult alternative is NIH ChestX-ray14 (`timm/nih-chest-xray-14`, 112k PNGs at 1024x1024,
21.6 GB, NIH terms); swap REPO_ID and the prefixes below to use it.

Layout: raw_root/chest_xray/images/<shard>_<row>.jpg

License: CC BY 4.0 (Mendeley Data rscbjbr9sj/2). Routine clinical radiographs with the usual
vendor post-processing (windowing, possibly edge enhancement), collected before 2018: no
generative AI.
"""

from pathlib import Path

from ._real_utils import fetch_hub_parquet, hub_files, image_rows
from .base import Source, register

REPO_ID = "hf-vision/chest-xray-pneumonia"


@register
class ChestXray(Source):
    name = "chest_xray"
    license = "CC BY 4.0"
    url = "https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia"
    description = "Kermany pediatric chest X-rays (5.9k JPEG, ~1-2k px), hf-vision/chest-xray-pneumonia"

    def __init__(self, split: str = "train", min_side: int = 512):
        self.split = split
        self.min_side = min_side

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        out_dir = Path(raw_root) / self.name / "images"
        files = hub_files(REPO_ID, ("data/train-", "data/validation-", "data/test-"))
        fetch_hub_parquet(REPO_ID, files, out_dir, max_items, min_side=self.min_side)

    def rows(self, raw_root: Path) -> list[dict]:
        return image_rows(raw_root, Path(raw_root) / self.name / "images", "biomed", self.name, self.split)
