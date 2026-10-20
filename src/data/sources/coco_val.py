"""MS COCO val2017: 5,000 Flickr photos, downloaded one by one from the official image host.

The image list (file name, URL, size, license id) comes from the 1 MB validation parquet of the HF
mirror `phiyodr/coco2017`; the JPEGs themselves come from http://images.cocodataset.org/val2017/,
so `max_items` is honoured exactly (a deterministic subset of the 5,000 ids).

Resolution: at most 640 px on the long side, typically 640x480 / 640x427, so the 512 px
short-side preference cannot be met; `min_side` defaults to 0 (set e.g. 480 to drop small ones).

Layout: raw_root/coco_val/images/<file_name>.jpg

License: annotations CC BY 4.0; images keep their Flickr licenses (ids 1-8 in COCO's `licenses`
list: BY-NC-SA, BY-NC, BY-NC-ND, BY, BY-SA, BY-ND, no known restrictions, US gov work).
Photos are from 2004-2014 consumer cameras and phones: JPEG recompressed by Flickr, no AI ISP.
"""

from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import hf_hub_download

from ._real_utils import image_rows
from .base import Source, register, take

INDEX_REPO = "phiyodr/coco2017"
INDEX_FILE = "data/validation-00000-of-00001-e3c37e369512a3aa.parquet"


@register
class CocoVal(Source):
    name = "coco_val"
    license = "Flickr CC (mixed)"
    url = "https://cocodataset.org/#download"
    description = "MS COCO val2017 photos (5k, <=640 px), per-image download"

    def __init__(self, split: str = "train", min_side: int = 0):
        self.split = split
        self.min_side = min_side

    def _index(self, root: Path) -> pd.DataFrame:
        path = hf_hub_download(INDEX_REPO, INDEX_FILE, repo_type="dataset", local_dir=root / "_index")
        meta = pd.read_parquet(path, columns=["file_name", "coco_url", "height", "width"])
        meta = meta[meta[["height", "width"]].min(axis=1) >= self.min_side]
        return meta.sort_values("file_name").reset_index(drop=True)

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        root = Path(raw_root) / self.name
        out_dir = root / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = self._index(root)
        urls = take(list(meta["coco_url"]), max_items)
        failed = 0
        with requests.Session() as session:
            for url in urls:
                target = out_dir / url.rsplit("/", 1)[-1]
                if target.exists():
                    continue
                try:
                    response = session.get(url, timeout=60)
                    response.raise_for_status()
                except requests.RequestException:
                    failed += 1
                    continue
                target.write_bytes(response.content)
        if failed:
            print(f"[{self.name}] {failed} of {len(urls)} images failed to download; rerun to retry")

    def rows(self, raw_root: Path) -> list[dict]:
        return image_rows(raw_root, Path(raw_root) / self.name / "images", "photo", self.name, self.split)
