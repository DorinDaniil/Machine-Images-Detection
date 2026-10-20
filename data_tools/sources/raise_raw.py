"""RAISE camera-original photos (8,156 shots, Nikon D40/D90/D7000, 2011-2014), semi-manual.

RAISE hands out its file list only through a web form, so the first step is manual:
  1. open http://loki.disi.unitn.it/RAISE/download.html, pick a package (RAISE-1k is enough),
     fill in the form and save the CSV it returns into raw_root/raise_raw/ (any *.csv name);
  2. run download() again: it reads the CSV's `TIFF` column (columns are File, NEF, TIFF, ...)
     and fetches a deterministic subset of `max_items` TIFFs with requests.
Any other camera-original set (e.g. Dresden Image Database JPEGs) can be dropped manually into
raw_root/raise_raw/images/ in any sub-folder layout; rows() lists everything there.

Resolution: 3008x2000 (D40), 4288x2848 (D90), 4928x3264 (D7000); the TIFFs are uncompressed
RGB developed from the NEF raws, roughly 30-50 MB each, so always pass a cap
(all 8,156 would be ~300 GB). Bit depth of the TIFFs was not verified; Pillow reads 8-bit RGB.

Layout: raw_root/raise_raw/*.csv and raw_root/raise_raw/images/**/<file>.tif|jpg|png

License: free for research, cite Dang-Nguyen et al., ACM MMSys 2015. DSLR shots, never
JPEG-compressed, no phone ISP or AI enhancement: the cleanest "real" reference in this set.
"""

from pathlib import Path

import pandas as pd
import requests

from ._real_utils import download_url, image_rows
from .base import Source, register, take


@register
class RaiseRaw(Source):
    name = "raise_raw"
    license = "research only"
    url = "http://loki.disi.unitn.it/RAISE/"
    description = "RAISE camera-original TIFFs (3008-4928 px); form-gated CSV, then auto download"

    def __init__(self, split: str = "train"):
        self.split = split

    def _tiff_urls(self, root: Path) -> list[str]:
        urls = []
        for csv_path in sorted(root.glob("*.csv")):
            table = pd.read_csv(csv_path, dtype=str)
            if "TIFF" in table.columns:
                urls += [u.strip() for u in table["TIFF"].dropna() if u.strip()]
        return sorted(set(urls))

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        root = Path(raw_root) / self.name
        root.mkdir(parents=True, exist_ok=True)
        urls = self._tiff_urls(root)
        if not urls:
            print(
                f"[{self.name}] manual step: get the RAISE CSV from {self.url}download.html (web form)\n"
                f"  and save it as {root}/RAISE.csv, then rerun; or put camera-original images\n"
                f"  (any layout, jpg/png/tif) under {root}/images/."
            )
            return
        if max_items is None:
            print(f"[{self.name}] no cap given: downloading all {len(urls)} TIFFs (~40 MB each)")
        for url in take(urls, max_items):
            name = url.rsplit("/", 1)[-1]
            try:
                download_url(url, root / "images" / name)
            except requests.RequestException as err:
                print(f"[{self.name}] failed {url}: {err}")

    def rows(self, raw_root: Path) -> list[dict]:
        return image_rows(raw_root, Path(raw_root) / self.name / "images", "photo", self.name, self.split)
