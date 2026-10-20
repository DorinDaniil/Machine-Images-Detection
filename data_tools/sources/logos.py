"""LLD-logo, the Large Logo Dataset (Sage et al., 2017): 122,920 logos from Twitter profiles.

download() fetches the official "cleaned, single-file format" zip (LLD-logo_files.zip, 6.66 GB)
from ETH Zurich and extracts a deterministic subset of `max_items` images. The zip is one file,
so the cap cannot reduce the download itself; use `sample=True` to fetch the official 27 MB
sample zip instead (for smoke tests). The zip is kept so later calls only extract.

Resolution: files as served by Twitter (mostly JPEG or PNG), between 64 and 400 px, mostly
400x400; the 512 px preference cannot be met for logos.

Layout: raw_root/logos/LLD-logo_files.zip and raw_root/logos/images/<zip member path>

License: academic research only, copyright stays with the logo owners. Scraped in August 2017,
before image generators; logos are designed in vector tools and rasterised by Twitter.
"""

from pathlib import Path

import requests

from ._real_utils import download_url, extract_images, image_rows
from .base import Source, register

BASE_URL = "https://data.vision.ee.ethz.ch/sagea/lld/data/"
FULL_ZIP = "LLD-logo_files.zip"
SAMPLE_ZIP = "LLD-logo_sample.zip"


@register
class Logos(Source):
    name = "logos"
    license = "academic research only"
    url = "https://data.vision.ee.ethz.ch/sagea/lld/"
    description = "LLD-logo Twitter logos (64-400 px), official 6.7 GB zip (or 27 MB sample)"

    def __init__(self, split: str = "train", sample: bool = False):
        self.split = split
        self.sample = sample

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        root = Path(raw_root) / self.name
        zip_name = SAMPLE_ZIP if self.sample else FULL_ZIP
        try:
            zip_path = download_url(BASE_URL + zip_name, root / zip_name)
        except requests.RequestException as err:
            print(
                f"[{self.name}] download failed ({err}).\n"
                f"  Manual: fetch {BASE_URL + zip_name} into {root}/ and rerun, or unpack any\n"
                f"  logo images into {root}/images/."
            )
            return
        count = extract_images(zip_path, root / "images", max_items)
        print(f"[{self.name}] {count} logos extracted from {zip_name}")

    def rows(self, raw_root: Path) -> list[dict]:
        return image_rows(raw_root, Path(raw_root) / self.name / "images", "graphics", self.name, self.split)
