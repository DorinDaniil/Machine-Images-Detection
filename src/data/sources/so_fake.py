"""So-Fake (arXiv 2505.18660): So-Fake-OOD v3 + the So-Fake-Set validation split, with masks.

- `saberzl/So-Fake-OOD` (CC BY-NC 4.0, 135 GB, 46 parquet shards "test_image-*"): 100.6k rows,
  REAL from Reddit / Tumblr / Bluesky (X and Instagram reals are released only as links in
  manifests/*.json and are skipped), FULL_SYNTHETIC from 15 held-out commercial generators
  (GPT-image-2, nano_banana_2, FLUX_2, Seedream 4.5, Imagen 4, ...), TAMPERED local edits by the
  OpenAI and Ideogram editors with pixel masks.
- `saberzl/So-Fake-Set` (card text says CC BY 4.0, no license tag; 1.28 TB): only the validation
  split (232 shards, ~236k rows: real, full_synthetic from ~30 open generators, tampered by
  flux.1-dev / sdxl / latent_diffusion / Imagic with masks) is used; the 1.17 TB train split is not.

Everything becomes split="test". Tampered rows get mask_path: native masks come as L/RGB/RGBA PNGs,
some soft, and are re-saved as binary PNG (255 = edited) next to the image as <stem>.mask.png.
Edits by OpenAI / Ideogram get the generator "openai-edit" / "ideogram-edit".

Shard rows are shuffled, so with `max_items` (half per repo) one shard per ~1,000 requested
images is downloaded (OOD shards ~2.9 GB, Set shards ~0.5 GB), a random spread of rows is
exported and the shard is deleted. Layout: raw_root/so_fake/images/{ood,set}/<shard>_<n>.<ext> + index.csv.
"""

import math
from pathlib import Path

from ._fake_files import (
    append_index, index_rows, mask_path_for, rel, save_binary_mask, save_image_bytes, slug,
)
from ._fake_hub import hub_files, iter_parquet, process_hub_shards, spread
from .base import Source, register

COLUMNS = ["image", "mask", "label", "generator"]
PARTS = {  # part -> (repo, shard prefix, columns to read)
    "ood": ("saberzl/So-Fake-OOD", "data/test_image-", COLUMNS + ["editor"]),
    "set": ("saberzl/So-Fake-Set", "data/validation-", COLUMNS),
}
IMAGES_PER_SHARD = 1000
OOD_GENERATORS = {
    "FLUX_2": "flux.2", "Flux.1_pro": "flux.1-pro", "GPT-image-1.5": "gpt-image-1.5", "GPT-image-2": "gpt-image-2",
    "GPT4o": "gpt-4o", "Hidream": "hidream", "Ideogram2": "ideogram-2", "Ideogram3": "ideogram-3",
    "Imagen4": "imagen-4", "imagen3": "imagen-3", "Recraftv3": "recraft-v3", "Seedream3.0": "seedream-3.0",
    "seedream4.5": "seedream-4.5", "nano_banana": "nano-banana", "nano_banana_2": "nano-banana-2",
}


def generator_of(row: dict, label: str) -> str:
    if label == "real":
        return ""
    if row.get("editor"):  # OOD tampered rows name the editing service
        return slug(row["editor"]) + "-edit"
    name = row.get("generator") or "unknown"
    return OOD_GENERATORS.get(name, slug(name))


@register
class SoFake(Source):
    name = "so_fake"
    license = "CC BY-NC 4.0 / CC BY"
    url = "https://huggingface.co/datasets/saberzl/So-Fake-OOD"
    description = "So-Fake-OOD v3 + So-Fake-Set val: social-media real / synthetic / tampered (masks)"

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        raw_root = Path(raw_root)
        for part, (repo_id, prefix, columns) in PARTS.items():
            files = hub_files(repo_id, prefix)
            part_cap = None if max_items is None else math.ceil(max_items / len(PARTS))
            if part_cap is not None:
                files = spread(files, math.ceil(part_cap / IMAGES_PER_SHARD))
            out_dir = raw_root / self.name / "images" / part
            process_hub_shards(
                repo_id, files, raw_root / self.name, part_cap,
                lambda local, quota: self._export_shard(local, quota, columns, raw_root, out_dir),
            )

    def _export_shard(
        self, local: Path, quota: int | None, columns: list[str], raw_root: Path, out_dir: Path
    ) -> int:
        records = []
        for i, row in enumerate(iter_parquet(local, columns, quota)):
            image = row["image"] or {}
            path = save_image_bytes(image.get("bytes") or b"", out_dir / f"{local.stem}_{i:06d}")
            if path is None:
                continue
            label = str(row["label"]).lower()
            mask_path = ""
            mask = row.get("mask") or {}
            if label == "tampered" and mask.get("bytes"):
                mask_path = rel(save_binary_mask(mask["bytes"], mask_path_for(path)), raw_root)
            records.append({
                "path": rel(path, raw_root),
                "label": 0 if label == "real" else 1,
                "generator": generator_of(row, label),
                "content_type": "photo",
                "mask_path": mask_path,
            })
        append_index(raw_root / self.name, records)
        return len(records)

    def rows(self, raw_root: Path) -> list[dict]:
        return index_rows(raw_root, self.name)
