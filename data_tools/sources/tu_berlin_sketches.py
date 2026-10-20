"""TU-Berlin sketch dataset (Eitz et al., SIGGRAPH 2012): 20,000 sketches, 250 categories.

Taken from the HF mirror `kmewhort/tu-berlin-png`: 3 parquet shards (two train shards of ~9,940
sketches / 290 MB and a small test shard), 590 MB in total. All images are 1111x1111 PNG
renderings of the original SVG strokes drawn by non-experts on a tablet/mouse: clean black
lines on white. Shards are fetched in a fixed shuffled order and deleted after export; with a
cap only as many shards as needed are fetched, so `max_items` is honoured to within one shard.

Layout: raw_root/tu_berlin_sketches/images/<shard>_<row>.png

The original project page (cybertron.cg.tu-berlin.de/eitz/projects/classifysketch) returned 404
in Oct 2026, so the HF mirror is the practical source.

License: CC BY 4.0. Crowd-sourced in 2011 (Amazon Mechanical Turk), no AI involvement; note
that PNGs are vector renderings, so they carry no camera or JPEG traces at all.
"""

from pathlib import Path

from ._real_utils import fetch_hub_parquet, hub_files, image_rows
from .base import Source, register

REPO_ID = "kmewhort/tu-berlin-png"


@register
class TuBerlinSketches(Source):
    name = "tu_berlin_sketches"
    license = "CC BY 4.0"
    url = "https://huggingface.co/datasets/kmewhort/tu-berlin-png"
    description = "TU-Berlin sketches (20k, 1111x1111 PNG), parquet from kmewhort/tu-berlin-png"

    def __init__(self, split: str = "train"):
        self.split = split

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        out_dir = Path(raw_root) / self.name / "images"
        files = hub_files(REPO_ID, ("data/train-", "data/test-"))
        fetch_hub_parquet(REPO_ID, files, out_dir, max_items)

    def rows(self, raw_root: Path) -> list[dict]:
        return image_rows(raw_root, Path(raw_root) / self.name / "images", "art", self.name, self.split)
