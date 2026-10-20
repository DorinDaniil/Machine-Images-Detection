"""GenImage (NeurIPS 2023 D&B, github.com/GenImage-Dataset/GenImage): capped subset of the test split.

The official release is on Google Drive / Baidu; this adapter uses the HF Arrow repack
`nebula/GenImage-arrow` (CC BY-NC-SA 4.0, by the OpenSDI author; the stored encoded bytes are
written out as they are). Test split only (data/test/<generator>/data-*.arrow, ~25 GB, 6,000 real
+ 6,000 fake per generator, real = ImageNet val photos): ADM, BigGAN, GLIDE, Midjourney (v5),
SD v1.4, SD v1.5, VQDM, Wukong; the 630 GB train split is not used. Columns: image_path, image (bytes),
label (0 real / 1 fake). All rows split="test", content_type "photo".

Known bias: GenImage fakes are mostly PNG while ImageNet reals are JPEG, and sizes differ by
generator, so format / resolution shortcuts work on it; treat scores on it with care.

Inside a generator the shards hold reals first and fakes last, so with `max_items` at least two
evenly spaced shards per generator (first and last, ~0.5 GB each) are downloaded, a random spread
of rows is exported and the shard deleted. Layout: raw_root/genimage/images/<generator>/<name> + index.csv.
"""

import math
from pathlib import Path

import pyarrow as pa

from ._fake_files import append_index, index_rows, rel, save_image_bytes
from ._fake_hub import hub_files, pick_indices, process_hub_shards, spread
from .base import Source, register

REPO_ID = "nebula/GenImage-arrow"
GENERATORS = {
    "ADM": "adm", "BigGAN": "biggan", "glide": "glide", "Midjourney": "midjourney-v5",
    "stable_diffusion_v_1_4": "sd-1.4", "stable_diffusion_v_1_5": "sd-1.5", "VQDM": "vqdm", "wukong": "wukong",
}
ROWS_PER_SHARD = 1000
MIN_SHARDS = 2


@register
class GenImage(Source):
    name = "genimage"
    license = "CC BY-NC-SA 4.0"
    url = "https://huggingface.co/datasets/nebula/GenImage-arrow"
    description = "GenImage test (capped): ADM, BigGAN, GLIDE, MJ v5, SD1.4/1.5, VQDM, Wukong vs ImageNet"

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        raw_root = Path(raw_root)
        for folder, generator in GENERATORS.items():
            files = hub_files(REPO_ID, f"data/test/{folder}/", suffix=".arrow")
            gen_cap = None if max_items is None else math.ceil(max_items / len(GENERATORS))
            if gen_cap is not None:
                files = spread(files, max(MIN_SHARDS, math.ceil(gen_cap / ROWS_PER_SHARD)))
            process_hub_shards(
                REPO_ID, files, raw_root / self.name, gen_cap,
                lambda local, quota: self._export_shard(local, quota, raw_root, generator),
            )

    def _export_shard(self, local: Path, quota: int | None, raw_root: Path, generator: str) -> int:
        with pa.OSFile(str(local)) as source:
            table = pa.ipc.open_stream(source).read_all()
        records = []
        for index in pick_indices(table.num_rows, quota):
            row = table.slice(index, 1).to_pylist()[0]
            stem = raw_root / self.name / "images" / generator / Path(row["image_path"]).stem
            path = save_image_bytes(row["image"], stem)
            if path is None:
                continue
            label = int(row["label"])
            records.append({
                "path": rel(path, raw_root),
                "label": label,
                "generator": generator if label else "",
                "content_type": "photo",
                "mask_path": "",
            })
        append_index(raw_root / self.name, records)
        return len(records)

    def rows(self, raw_root: Path) -> list[dict]:
        return index_rows(raw_root, self.name)
