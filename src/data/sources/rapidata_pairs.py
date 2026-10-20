"""Rapidata human-preference / benchmark sets: generated images of many models on the same prompts.

Each parquet row compares two images for one prompt (image1/model1 vs image2/model2), and the same
(prompt, model) image recurs in many rows. Images are deduplicated on (prompt, model) and max_items
is split evenly over the models found in the first shard. Shards are visited evenly spaced over the
repo; per row group only the prompt/model columns are read first, the image columns only when the
row group holds at least `min_new` still-needed images (HTTP range reads through HfFileSystem,
only the chosen images are written). Stops at max_items or after `patience` shards that add nothing
(models with fewer images than their share). All images are generated: label 1.

    kind: custom
    adapter: rapidata_pairs
    args:
      repo: Rapidata/text-to-image-human-feedback-benchmark
      models: {"GPT Image 2.0": gpt-image-2}    # optional renames, default slug of the raw name
      max_shards: 64
      content_type: other
"""

import math
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from ._fake_files import append_index, index_rows, rel, save_image_bytes, slug
from ._fake_hub import hub_files, spread
from .base import Source, register

PAIRS = (("image1", "model1"), ("image2", "model2"))


@register
class RapidataPairs(Source):
    name = "rapidata_pairs"
    license = "see dataset card"

    def __init__(self, repo: str = "", models: dict | None = None, max_shards: int = 64, min_new: int = 4,
                 patience: int = 5, prefix: str = "data/", content_type: str = "other", manifest_split: str = "test"):
        self.repo, self.models, self.prefix = repo, dict(models or {}), prefix
        self.max_shards, self.min_new, self.patience = max_shards, min_new, patience
        self.content_type, self.manifest_split = content_type, manifest_split
        self.url, self.description = f"https://huggingface.co/datasets/{repo}", f"{repo} (deduplicated pairs)"

    def _slug(self, model: str) -> str:
        return self.models.get(model, slug(str(model).split("/")[-1]))

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        root = Path(raw_root) / self.name
        if (root / "index.csv").exists():
            return
        fs, seen, count, records, quota, stale = HfFileSystem(), set(), Counter(), [], None, 0
        for shard, filename in enumerate(spread(hub_files(self.repo, self.prefix), self.max_shards)):
            before = len(records)
            with fs.open(f"datasets/{self.repo}/{filename}", "rb", block_size=1 << 20) as f:
                pf = pq.ParquetFile(f)
                metas = [pf.read_row_group(g, columns=["prompt", "model1", "model2"]).to_pylist()
                         for g in range(pf.num_row_groups)]
                if quota is None and max_items is not None:
                    models = {r[m] for meta in metas for r in meta for _, m in PAIRS}
                    quota = math.ceil(max_items / max(len(models), 1))
                for group, meta in enumerate(metas):
                    wanted, keys, pending = [], set(), Counter()
                    for i, r in enumerate(meta):
                        for image_col, model_col in PAIRS:
                            key, model = (r["prompt"], r[model_col]), r[model_col]
                            if key in seen or key in keys or (quota and count[model] + pending[model] >= quota):
                                continue
                            keys.add(key)
                            pending[model] += 1
                            wanted.append((i, image_col, model))
                    if len(wanted) < self.min_new:
                        continue
                    table = pf.read_row_group(group, columns=sorted({c for _, c, _ in wanted}))
                    for i, image_col, model in wanted:
                        cell = table.column(image_col)[i].as_py() or {}
                        stem = root / self._slug(model) / f"{shard:04d}_{group}_{i}_{image_col[-1]}"
                        path = save_image_bytes(cell.get("bytes") or b"", stem)
                        if path is None:
                            continue
                        seen.add((meta[i]["prompt"], model))
                        count[model] += 1
                        records.append({"path": rel(path, Path(raw_root)), "label": 1, "generator": self._slug(model),
                                        "content_type": self.content_type, "mask_path": ""})
            print(f"   {self.repo}: {filename}: {len(records)} images, {len(count)} models")
            stale = stale + 1 if len(records) == before else 0
            if max_items is not None and (len(records) >= max_items or stale >= self.patience):
                break
        append_index(root, records)

    def rows(self, raw_root: Path) -> list[dict]:
        return index_rows(Path(raw_root), self.name, self.manifest_split)
