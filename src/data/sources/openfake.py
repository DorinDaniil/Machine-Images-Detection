"""OpenFake v2 (ComplexDataLab, arXiv 2509.09495): capped subset of the two test sets.

HF repo `ComplexDataLab/OpenFake`, CC BY-NC 4.0 (proprietary-generator subsets carry provider
non-compete terms). NOTE: the content is deliberately political / socially salient: public
figures' faces, news-like events (disasters, protests) and meme-style images with text overlays,
so expect identifiable people and misinformation-style imagery.

Used parts (the 2.3M-row, 3.4 TB train split is not touched):
- core/test-*.parquet   13 shards, ~5 GB each, 91,397 rows: held-out OOD generators (no overlap with
  training generators; one shard holds e.g. z-image-turbo, flux.2-klein-9b, illustrious,
  gpt-image-1.5, gpt-image-2, midjourney-7, nano-banana-pro, seedream-v5.0, recraft-v3 and frames of
  veo-3 / sora-2 / wan-video-2.5) paired with DOCCI / ImageNet reals;
- reddit/test-*.parquet  5 shards, ~4.9 GB each, 36,087 rows: in-the-wild images from AI-art and
  photography subreddits (model is "reddit" for both classes, so fakes get generator "unknown").

Columns used: image {bytes}, label ("real"/"fake"), model (already a lowercase slug). Everything
is split="test"; content_type is "art" for anime-style fine-tunes (model name contains one of
ART_HINTS) and "photo" otherwise (memes with text are not flagged in the metadata).

Rows are shuffled within shards. With `max_items` (half per part) one shard per ~5,000 requested
images is downloaded, a random spread of rows exported, then the shard deleted, so even a small cap
downloads at least 2 x ~5 GB. Layout: raw_root/openfake/images/{core,reddit}/... + index.csv.
"""

import math
from pathlib import Path

from ._fake_files import append_index, index_rows, rel, save_image_bytes, slug
from ._fake_hub import hub_files, iter_parquet, process_hub_shards, spread
from .base import Source, register

REPO_ID = "ComplexDataLab/OpenFake"
PARTS = {"core": "core/test-", "reddit": "reddit/test-"}
IMAGES_PER_SHARD = 5000
ART_HINTS = ("illustrious", "anime", "pony")


@register
class OpenFake(Source):
    name = "openfake"
    license = "CC BY-NC 4.0"
    url = "https://huggingface.co/datasets/ComplexDataLab/OpenFake"
    description = "OpenFake v2 core OOD test + Reddit in-the-wild test (political content), capped"

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        raw_root = Path(raw_root)
        for part, prefix in PARTS.items():
            files = hub_files(REPO_ID, prefix)
            part_cap = None if max_items is None else math.ceil(max_items / len(PARTS))
            if part_cap is not None:
                files = spread(files, math.ceil(part_cap / IMAGES_PER_SHARD))
            out_dir = raw_root / self.name / "images" / part
            process_hub_shards(
                REPO_ID, files, raw_root / self.name, part_cap,
                lambda local, quota: self._export_shard(local, quota, raw_root, out_dir),
            )

    def _export_shard(self, local: Path, quota: int | None, raw_root: Path, out_dir: Path) -> int:
        records = []
        for i, row in enumerate(iter_parquet(local, ["image", "label", "model"], quota)):
            image = row["image"] or {}
            path = save_image_bytes(image.get("bytes") or b"", out_dir / f"{local.stem}_{i:06d}")
            if path is None:
                continue
            label = 0 if row["label"] == "real" else 1
            model = row["model"] or "unknown"
            records.append({
                "path": rel(path, raw_root),
                "label": label,
                "generator": "" if label == 0 else ("unknown" if model == "reddit" else slug(model)),
                "content_type": "art" if any(h in model.lower() for h in ART_HINTS) else "photo",
                "mask_path": "",
            })
        append_index(raw_root / self.name, records)
        return len(records)

    def rows(self, raw_root: Path) -> list[dict]:
        return index_rows(raw_root, self.name)
