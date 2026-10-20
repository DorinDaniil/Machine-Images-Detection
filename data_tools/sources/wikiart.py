"""WikiArt paintings from the HF dataset `huggan/wikiart` (81,444 artworks, 27 styles).

Images are stored in 72 parquet shards (~1,130 images / 520 MB each) and were rescaled so the
short side is mostly 1382 px (some smaller). Shards are fetched in a fixed shuffled order,
exported as the original JPEG bytes and deleted, until `max_items` images passed `min_side`
(default 512 px); the cap is honoured to within one shard. Without a cap all 34 GB are fetched.

Layout: raw_root/wikiart/images/<shard>_<row>.jpg

License: the HF card says "unknown". WikiArt mixes public-domain works with copyrighted
modern art shown under fair use; use for research only. Content is photographs/scans of
physical artworks plus some digital art, collected around 2016 (before diffusion models), but
museum photos may carry ordinary camera processing and some scans are retouched.
"""

from pathlib import Path

from ._real_utils import fetch_hub_parquet, hub_files, image_rows
from .base import Source, register

REPO_ID = "huggan/wikiart"


@register
class WikiArt(Source):
    name = "wikiart"
    license = "unknown (fair use)"
    url = "https://huggingface.co/datasets/huggan/wikiart"
    description = "WikiArt paintings (short side ~1382 px), parquet shards from huggan/wikiart"

    def __init__(self, split: str = "train", min_side: int = 512):
        self.split = split
        self.min_side = min_side

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        out_dir = Path(raw_root) / self.name / "images"
        files = hub_files(REPO_ID, ("data/train-",))
        fetch_hub_parquet(REPO_ID, files, out_dir, max_items, min_side=self.min_side)

    def rows(self, raw_root: Path) -> list[dict]:
        return image_rows(raw_root, Path(raw_root) / self.name / "images", "art", self.name, self.split)
