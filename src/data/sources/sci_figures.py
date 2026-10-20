"""Scientific figures from arXiv papers via ArxivCap (Li et al., 2024, `MMInstruction/ArxivCap`).

ArxivCap holds 6.4M figure images from 572k arXiv papers in 6,544 parquet shards named
arXiv_src_<YYMM>_<NNN> (one per arXiv source tarball, 100-160 MB each, 755 GB in total).
Every row is a paper; figures sit in caption_images[].cil_pairs[].image. Only shards from
papers submitted before 2022 (1992-2021) are used, so generative-AI illustrations are very
unlikely. Shards are fetched in a fixed shuffled order, their figures exported with the
original bytes and the shard deleted, until `max_items` figures passed `min_side` (default
512 px); without a cap only the first 20 shards are fetched (the 4,509 pre-2022 shards total
572 GB).

Resolution: mixed (card example 2016x1063); the ArxivCap pipeline converted the authors'
PDF/EPS/PNG figure files to raster images (the card shows .jpg), so vector plots carry
rasteriser and JPEG traces rather than camera traces.

Content: mostly charts, plots and diagrams, plus photos, micrographs and renderings, all
marked `graphics` (no reliable per-figure biomed flag). Not used: DocFigure (not on the
Hub) and SciCap (`CrowdAILab/scicap`, a 27 GB multi-part zip, CC BY-NC-SA 4.0).

Layout: raw_root/sci_figures/images/<shard>_<row>.jpg|png

License: dataset card CC BY 4.0; each figure keeps its paper's arXiv license (often the
non-exclusive arXiv distribution license), so treat as research use.
"""

from pathlib import Path

import pyarrow.parquet as pq

from ._real_utils import fetch_hub_parquet, hub_files, image_rows, save_image_bytes
from .base import Source, register

REPO_ID = "MMInstruction/ArxivCap"
CUTOFF_YEAR = 2022
DEFAULT_SHARDS = 20


def submitted_before_cutoff(filename: str) -> bool:
    """arXiv_src_2112_060.parquet -> 2021 < CUTOFF_YEAR."""
    year = int(Path(filename).stem.split("_")[2][:2])
    year += 1900 if year >= 90 else 2000
    return year < CUTOFF_YEAR


@register
class SciFigures(Source):
    name = "sci_figures"
    license = "CC BY 4.0 / arXiv"
    url = "https://huggingface.co/datasets/MMInstruction/ArxivCap"
    description = "arXiv paper figures 1992-2021 (mixed res) from MMInstruction/ArxivCap"

    def __init__(self, split: str = "train", min_side: int = 512):
        self.split = split
        self.min_side = min_side

    def _export(self, parquet: Path, out_dir: Path) -> int:
        count = 0
        index = 0
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=8, columns=["caption_images"]):
            for paper in batch.column(0).to_pylist():
                for figure in paper or []:
                    for pair in figure.get("cil_pairs") or []:
                        data = (pair.get("image") or {}).get("bytes")
                        stem = f"{parquet.stem}_{index:06d}"
                        if data and save_image_bytes(data, out_dir, stem, self.min_side):
                            count += 1
                        index += 1
        return count

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        out_dir = Path(raw_root) / self.name / "images"
        files = [f for f in hub_files(REPO_ID, ("data/arXiv_src_",)) if submitted_before_cutoff(f)]
        max_shards = DEFAULT_SHARDS if max_items is None else None
        fetch_hub_parquet(REPO_ID, files, out_dir, max_items, max_shards=max_shards, export=self._export)

    def rows(self, raw_root: Path) -> list[dict]:
        return image_rows(raw_root, Path(raw_root) / self.name / "images", "graphics", self.name, self.split)
