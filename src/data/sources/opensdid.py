"""OpenSDID test set from OpenSDI (CVPR 2025, arXiv 2503.19653, github.com/iamwangyabin/OpenSDI).

HF repo `nebula/OpenSDI_test` (CC BY-SA 4.0, academic use; real photos from megalith-10m),
52 parquet shards, 18.3 GB, 10k real + 10k fake per generator: SD1.5, SD2.1, SDXL, SD3, Flux.1 (shards are
named <gen>-NNNNN-of-NNNNN.parquet). Columns: key, image, mask, label. The key tells the kind:

    entire/<gen>/fake/*   global synthesis (no mask)
    partial/<gen>/fake/*  local edit / inpainting, grey-level mask (white = edited)
    entire|partial/<gen>/real/*     (a few real rows only have a bare file name as key)

Partial-edit masks are re-saved as binary PNG (255 = edited) next to the image as <stem>.mask.png.
All rows split="test", content_type "photo". The SD1.5 training set (`nebula/OpenSDI_train`) and
the larger `nebula/OpenSDIDplus` are not used.

Within a generator the shards are roughly ordered by kind (entire/fake first, partial/real last,
neighbouring kinds share shards), so with `max_items` at least 4 evenly spaced shards per
generator (~0.35 GB each, more for large caps) are downloaded, a random spread of rows is
exported, and the shard is deleted.
Layout: raw_root/opensdid/images/<gen>/<entire|partial|bare>_<real|fake>/<name> + index.csv.
"""

import math
from pathlib import Path

from ._fake_files import append_index, index_rows, mask_path_for, rel, save_binary_mask, save_image_bytes
from ._fake_hub import hub_files, iter_parquet, process_hub_shards, spread
from .base import Source, register

REPO_ID = "nebula/OpenSDI_test"
GENERATORS = {"sd15": "sd-1.5", "sd2": "sd-2.1", "sdxl": "sdxl", "sd3": "sd3", "flux": "flux.1"}
ROWS_PER_SHARD = 1000
MIN_SHARDS = 4


@register
class OpenSDID(Source):
    name = "opensdid"
    license = "CC BY-SA 4.0"
    url = "https://huggingface.co/datasets/nebula/OpenSDI_test"
    description = "OpenSDID test: SD1.5/2.1/XL/3 + Flux.1 global fakes and local edits with masks"

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        raw_root = Path(raw_root)
        all_files = hub_files(REPO_ID, "data/")
        for gen in GENERATORS:
            files = [f for f in all_files if Path(f).name.startswith(gen + "-")]
            gen_cap = None if max_items is None else math.ceil(max_items / len(GENERATORS))
            if gen_cap is not None:
                files = spread(files, max(MIN_SHARDS, math.ceil(gen_cap / ROWS_PER_SHARD)))
            process_hub_shards(
                REPO_ID, files, raw_root / self.name, gen_cap,
                lambda local, quota: self._export_shard(local, quota, raw_root),
            )

    def _export_shard(self, local: Path, quota: int | None, raw_root: Path) -> int:
        records = []
        for row in iter_parquet(local, ["key", "image", "mask", "label"], quota):
            parts = row["key"].split("/")
            kind = parts[-4] if len(parts) >= 4 else "bare"
            gen = local.name.split("-")[0]
            label = int(row["label"])
            folder = f"{kind}_{'fake' if label else 'real'}"
            stem = raw_root / self.name / "images" / gen / folder / Path(parts[-1]).stem
            path = save_image_bytes((row["image"] or {}).get("bytes") or b"", stem)
            if path is None:
                continue
            mask = row["mask"] or {}
            mask_path = ""
            if label == 1 and kind == "partial" and mask.get("bytes"):
                mask_path = rel(save_binary_mask(mask["bytes"], mask_path_for(path)), raw_root)
            records.append({
                "path": rel(path, raw_root),
                "label": label,
                "generator": GENERATORS.get(gen, gen) if label else "",
                "content_type": "photo",
                "mask_path": mask_path,
            })
        append_index(raw_root / self.name, records)
        return len(records)

    def rows(self, raw_root: Path) -> list[dict]:
        return index_rows(raw_root, self.name)
