"""PatchCamelyon (Veeling et al., 2018): H&E lymph-node histopathology patches from Camelyon16.

HF dataset `1aurent/PatchCamelyon`: 327,680 patches in 17 parquet shards (13 train shards of
~20,000 patches / 470 MB, 2 valid + 2 test shards of ~16,400 patches / 375 MB; 7.6 GB total).
Only the valid + test shards (65,536 patches, 1.5 GB) are used, which is plenty for a
real-image pool. Shards are fetched in a fixed shuffled order, exported with their original
bytes and deleted, so `max_items` is honoured to within one shard (~16k patches).

Resolution: 96x96 px at 10x magnification (downsampled from 40x whole-slide scans); far
below the 512 px preference, so the detector must handle tiny inputs for this source.

Layout: raw_root/patchcamelyon/images/<shard>_<row>.png|jpg

License: CC0 1.0. Whole-slide scans from the Camelyon16 challenge (Radboud UMC, UMC Utrecht),
no AI enhancement.
"""

from pathlib import Path

from ._real_utils import fetch_hub_parquet, hub_files, image_rows
from .base import Source, register

REPO_ID = "1aurent/PatchCamelyon"


@register
class PatchCamelyon(Source):
    name = "patchcamelyon"
    license = "CC0 1.0"
    url = "https://huggingface.co/datasets/1aurent/PatchCamelyon"
    description = "PatchCamelyon H&E histology patches (96x96), valid+test of 1aurent/PatchCamelyon"

    def __init__(self, split: str = "train"):
        self.split = split

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        out_dir = Path(raw_root) / self.name / "images"
        files = hub_files(REPO_ID, ("data/valid-", "data/test-"))
        fetch_hub_parquet(REPO_ID, files, out_dir, max_items)

    def rows(self, raw_root: Path) -> list[dict]:
        return image_rows(raw_root, Path(raw_root) / self.name / "images", "biomed", self.name, self.split)
