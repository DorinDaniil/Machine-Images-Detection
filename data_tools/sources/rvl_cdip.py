"""RVL-CDIP scanned documents (Harley et al., 2015): letters, forms, memos, invoices, ...

The canonical HF repo `aharley/rvl_cdip` is a 38.8 GB tar.gz behind a loading script that
recent `datasets` versions no longer run, so this adapter uses the parquet copy
`chainyo/rvl-cdip` (same images and labels). Only its test split is used: 15 shards,
40,000 documents, 4.5 GB; shards are fetched in a fixed shuffled order, exported with their
original bytes (TIFF) and deleted, so `max_items` is honoured to within one shard (~2,700).

Resolution: grayscale scans resized so the largest side is at most 1000 px, typically
754x1000; nearly all pass the default 512 px `min_side` filter.

Layout: raw_root/rvl_cdip/images/<shard>_<row>.tif

License: "other": documents come from the IIT-CDIP / Legacy Tobacco Document Library (public
litigation records, 1950s-2000s); research use. Old office scans with fax/photocopy noise,
no AI processing; text is mostly typewritten or printed, some handwriting.
"""

from pathlib import Path

from ._real_utils import fetch_hub_parquet, hub_files, image_rows
from .base import Source, register

REPO_ID = "chainyo/rvl-cdip"


@register
class RvlCdip(Source):
    name = "rvl_cdip"
    license = "other (IIT-CDIP)"
    url = "https://huggingface.co/datasets/chainyo/rvl-cdip"
    description = "RVL-CDIP document scans (~754x1000 grayscale TIFF), test split of chainyo/rvl-cdip"

    def __init__(self, split: str = "train", min_side: int = 512):
        self.split = split
        self.min_side = min_side

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        out_dir = Path(raw_root) / self.name / "images"
        files = hub_files(REPO_ID, ("data/test-",))
        fetch_hub_parquet(REPO_ID, files, out_dir, max_items, min_side=self.min_side)

    def rows(self, raw_root: Path) -> list[dict]:
        return image_rows(raw_root, Path(raw_root) / self.name / "images", "text", self.name, self.split)
