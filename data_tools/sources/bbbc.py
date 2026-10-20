"""Broad Bioimage Benchmark Collection microscopy: BBBC038 (nuclei) and BBBC041 (malaria smears).

- BBBC038v1 (Kaggle Data Science Bowl 2018), stage1_train.zip + stage1_test.zip: 735 images
  of nuclei across fluorescence, brightfield and H&E histology, many labs and magnifications.
  PNG (often RGBA), from 256x256 up to 1040x1388; only <id>/images/*.png is kept, masks are
  skipped. License CC0.
- BBBC041v1, malaria.zip: 1,364 Giemsa-stained blood smears from three labs, PNG/JPEG,
  around 1600x1200 (resolution not verified). License CC BY-NC-SA 3.0.
Not used on purpose: BBBC005 (simulated images) and BBBC039 (16-bit TIFF, Pillow cannot
convert it to RGB without rescaling).

Each set is one zip from data.broadinstitute.org, so `max_items` cannot reduce the download;
it only caps extraction (split evenly over the selected sets, deterministic subset). Zips are
kept in zips/ so reruns only extract. The download links come from the official set pages
(bbbc.broadinstitute.org/BBBC038, /BBBC041); the file host itself timed out from the
development machine, so file sizes are unverified (BBBC038 stage1 is believed to be ~90 MB).
If the download fails, save each zip as raw_root/bbbc/zips/<set name>.zip by hand and rerun.

Layout: raw_root/bbbc/zips/<set>.zip, raw_root/bbbc/<set>/<zip member path>

Microscope camera images from research labs (2010s), no AI enhancement.
"""

import math
from pathlib import Path

import requests

from ._real_utils import download_url, extract_images, image_rows
from .base import Source, register

BASE_URL = "https://data.broadinstitute.org/bbbc/"
SETS = {
    "BBBC038_stage1_train": BASE_URL + "BBBC038/stage1_train.zip",
    "BBBC038_stage1_test": BASE_URL + "BBBC038/stage1_test.zip",
    "BBBC041_malaria": BASE_URL + "BBBC041/malaria.zip",
}


@register
class Bbbc(Source):
    name = "bbbc"
    license = "CC0 / CC BY-NC-SA 3.0"
    url = "https://bbbc.broadinstitute.org/"
    description = "BBBC038 nuclei (fluo/brightfield/H&E) + BBBC041 malaria smears, official zips"

    def __init__(self, split: str = "train", sets: tuple[str, ...] = tuple(SETS)):
        self.split = split
        self.sets = sets

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        root = Path(raw_root) / self.name
        quota = None if max_items is None else math.ceil(max_items / len(self.sets))
        for set_name in self.sets:
            try:
                zip_path = download_url(SETS[set_name], root / "zips" / f"{set_name}.zip")
            except requests.RequestException as err:
                print(
                    f"[{self.name}] {set_name}: download failed ({err}).\n"
                    f"  Manual: save {SETS[set_name]} as {root / 'zips' / (set_name + '.zip')} and rerun."
                )
                continue
            count = extract_images(zip_path, root / set_name, quota, keep=lambda n: "/masks/" not in n)
            print(f"[{self.name}] {set_name}: {count} images")

    def rows(self, raw_root: Path) -> list[dict]:
        root = Path(raw_root) / self.name
        rows = []
        for set_name in SETS:
            rows += image_rows(raw_root, root / set_name, "biomed", self.name, self.split)
        return rows
