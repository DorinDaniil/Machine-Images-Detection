"""Community Forensics (CVPR 2025, arXiv 2411.04125): capped evaluation subset.

HF repo `OwensLab/CommunityForensics` (CC BY 4.0 on the card; the paired real images keep the
licenses of their source datasets). The whole repo is ~1.1 TB (2.7M fakes from 4,803 generators);
this adapter only uses its two test-side parts:

- data/publicEval/*.parquet  6 shards, 27.7 GB, ~51.8k images, real (label 0) and fake, e.g.
  FLUX-dev, FLUX-schnell, IdeogramV1, Firefly_Image3 (split column = "test");
- data/commercial/*.parquet 2 shards, 16.6 GB, ~14.9k fakes only from commercial models
  (MidjourneyV5_2 / V6_1, Dalle2 / Dalle3, Imagen3, IdeogramV1 / V2, Firefly ...).

Columns used: image_data (raw bytes), model_name, label (0/1). `generator` = slug of model_name;
for real rows model_name names the paired generator and is dropped. The paper's full
"comprehensive" eval set lives in `OwensLab/CommunityForensics-Eval` (206 GB, CC BY-NC-SA 4.0),
not used here.

Shards are single 3-6 GB row groups, so a shard is always downloaded whole (and deleted once its
images are exported). With `max_items`, one shard per ~4,000 requested images is used per part
(at least one each) and a random spread of rows is kept, so the picks cover all generators in it.
Layout: raw_root/community_forensics/images/<part>/<shard>_<n>.<ext> + index.csv.
"""

import math
from pathlib import Path

from ._fake_files import append_index, index_rows, rel, save_image_bytes, slug
from ._fake_hub import hub_files, iter_parquet, process_hub_shards, spread
from .base import Source, register

REPO_ID = "OwensLab/CommunityForensics"
PARTS = ("data/publicEval/", "data/commercial/")
IMAGES_PER_SHARD = 4000


@register
class CommunityForensics(Source):
    name = "community_forensics"
    license = "CC BY 4.0"
    url = "https://huggingface.co/datasets/OwensLab/CommunityForensics"
    description = "Community Forensics PublicEval + Commercial parquet (capped), real + many generators"

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        raw_root = Path(raw_root)
        for part in PARTS:
            files = hub_files(REPO_ID, part)
            part_cap = None if max_items is None else math.ceil(max_items / len(PARTS))
            if part_cap is not None:
                files = spread(files, math.ceil(part_cap / IMAGES_PER_SHARD))
            out_dir = raw_root / self.name / "images" / Path(part).name
            process_hub_shards(
                REPO_ID, files, raw_root / self.name, part_cap,
                lambda local, quota: self._export_shard(local, quota, raw_root, out_dir),
            )

    def _export_shard(self, local: Path, quota: int | None, raw_root: Path, out_dir: Path) -> int:
        records = []
        for i, row in enumerate(iter_parquet(local, ["image_data", "model_name", "label"], quota)):
            path = save_image_bytes(row["image_data"], out_dir / f"{local.stem}_{i:06d}")
            if path is None:
                continue
            label = int(row["label"])
            records.append({
                "path": rel(path, raw_root),
                "label": label,
                "generator": slug(row["model_name"]) if label else "",
                "content_type": "photo",
                "mask_path": "",
            })
        append_index(raw_root / self.name, records)
        return len(records)

    def rows(self, raw_root: Path) -> list[dict]:
        return index_rows(raw_root, self.name)
