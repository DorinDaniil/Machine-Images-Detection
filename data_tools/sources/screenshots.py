"""Android UI screenshots from Rico (Deka et al., 2017) via `bevaya/RICO-Screen2Words`.

The HF dataset holds the 22,417 Screen2Words screens (a cleaned Rico subset) in 12 parquet
shards (8 train, 2 val, 2 test; ~2,000 screens / 220 MB each, 2.5 GB total). Each row also has
an app icon and a semantic-annotation rendering; only the `image` column (the real screenshot)
is exported. Shards are fetched in a fixed shuffled order, exported with their original JPEG
bytes and deleted, so `max_items` is honoured to within one shard.

Resolution: native phone screens, 1080x1920 or 1440x2560 (JPEG as stored by Rico).

Layout: raw_root/screenshots/images/<shard>_<row>.jpg

License: CC BY 4.0. Captured in 2017 from ~9,700 Play Store apps: rendered UI, text and
in-app photos/ads, no generative AI. Desktop/web UIs (e.g. biglab/webui-*) are not included;
they ship as multi-GB split zips.
"""

from pathlib import Path

from ._real_utils import fetch_hub_parquet, hub_files, image_rows
from .base import Source, register

REPO_ID = "bevaya/RICO-Screen2Words"


@register
class Screenshots(Source):
    name = "screenshots"
    license = "CC BY 4.0"
    url = "https://huggingface.co/datasets/bevaya/RICO-Screen2Words"
    description = "Rico Android screenshots (1080x1920 / 1440x2560), bevaya/RICO-Screen2Words"

    def __init__(self, split: str = "train", min_side: int = 512):
        self.split = split
        self.min_side = min_side

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        out_dir = Path(raw_root) / self.name / "images"
        files = hub_files(REPO_ID, ("data/",))
        fetch_hub_parquet(REPO_ID, files, out_dir, max_items, min_side=self.min_side)

    def rows(self, raw_root: Path) -> list[dict]:
        return image_rows(raw_root, Path(raw_root) / self.name / "images", "screenshot", self.name, self.split)
