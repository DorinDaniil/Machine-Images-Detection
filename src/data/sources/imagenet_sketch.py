"""ImageNet-Sketch drawings (50,889 images, 50 per ImageNet class) from `vaughankraska/imagenet_sketch`.

This HF copy is the original `songweig/imagenet_sketch` (a single 7.6 GB zip behind a loading
script that `datasets` no longer runs) shuffled with seed 42 and re-saved as 17 parquet shards
(13 train + 4 validation, ~3,000 images / 480 MB each). Because the rows are shuffled, any few
shards cover all classes. Shards are fetched in a fixed shuffled order, exported as the original
bytes and deleted, until `max_items` images passed `min_side` (default 512 px); the cap is
honoured to within one shard.

Resolution: very mixed, from ~150 px to ~2000 px (Google Image search results); with the 512 px
filter roughly half of the images remain.

Layout: raw_root/imagenet_sketch/images/<shard>_<row>.jpg|png

License: card says "unknown"; images are Google Image results collected in 2019 (pencil
sketches, line drawings, some digital drawings and scanned sketches), so no generative AI.
"""

from pathlib import Path

from ._real_utils import fetch_hub_parquet, hub_files, image_rows
from .base import Source, register

REPO_ID = "vaughankraska/imagenet_sketch"


@register
class ImageNetSketch(Source):
    name = "imagenet_sketch"
    license = "unknown (web images)"
    url = "https://github.com/HaohanWang/ImageNet-Sketch"
    description = "ImageNet-Sketch drawings (mixed res), parquet shards from vaughankraska/imagenet_sketch"

    def __init__(self, split: str = "train", min_side: int = 512):
        self.split = split
        self.min_side = min_side

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        out_dir = Path(raw_root) / self.name / "images"
        files = hub_files(REPO_ID, ("data/train-", "data/validation-"))
        fetch_hub_parquet(REPO_ID, files, out_dir, max_items, min_side=self.min_side)

    def rows(self, raw_root: Path) -> list[dict]:
        return image_rows(raw_root, Path(raw_root) / self.name / "images", "art", self.name, self.split)
