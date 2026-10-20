"""MLLM-generated image benchmark: GPT Image 2 & Nano Banana 2 (arXiv 2608.01258, SAP-DSP).

HF repo `zr-zhang/MLLM-Generated-Image-Detection-Dataset` (linked from github.com/xbrainnet/SAP-DSP),
Apache-2.0 on the card plus a use restriction: research on AIGC detection / forensics only, no use
for fraud, forgery or evading detectors. ~45k files, ~50 GB, mostly PNG:

    images/<generator>/<regime> Images/*.png|jpg
    generator: GPT Image 2 | Nano Banana 2 | FLUX.2 | Qwen Image 2.1 | Real
    regime:    Texture | Structure | Hybrid  (which artifact type dominates)

The repo holds ~3k images per (generator, regime) cell; besides the two MLLMs named in the paper
title it also has FLUX.2 and Qwen Image 2.1 folders. Real images are label 0; all rows split="test".

`max_items` is honoured exactly: an even share per (generator, regime) folder is fetched file by
file; without a cap the whole images/ tree is fetched with snapshot_download.
"""

import math
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from ._fake_files import IMAGE_EXTS, list_images, rel
from ._fake_hub import hub_files
from .base import Source, make_row, register, take

REPO_ID = "zr-zhang/MLLM-Generated-Image-Detection-Dataset"
GENERATORS = {
    "GPT Image 2": "gpt-image-2",
    "Nano Banana 2": "nano-banana-2",
    "FLUX.2": "flux.2",
    "Qwen Image 2.1": "qwen-image-2.1",
    "Real": "",
}


@register
class MllmBench(Source):
    name = "mllm_bench"
    license = "Apache-2.0 + use terms"
    url = "https://huggingface.co/datasets/zr-zhang/MLLM-Generated-Image-Detection-Dataset"
    description = "MLLM-gen benchmark: GPT Image 2, Nano Banana 2 (+FLUX.2, Qwen Image 2.1) vs real"

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        root = Path(raw_root) / self.name
        if max_items is None:
            snapshot_download(REPO_ID, repo_type="dataset", allow_patterns=["images/**"], local_dir=root)
            return
        folders = defaultdict(list)
        for f in hub_files(REPO_ID, "images/", suffix=""):
            if Path(f).suffix.lower() in IMAGE_EXTS:
                folders[str(Path(f).parent)].append(f)
        quota = math.ceil(max_items / len(folders))
        for files in folders.values():
            for filename in take(files, quota):
                if not (root / filename).exists():
                    hf_hub_download(REPO_ID, filename, repo_type="dataset", local_dir=root)

    def rows(self, raw_root: Path) -> list[dict]:
        root = Path(raw_root) / self.name / "images"
        rows = []
        for path in list_images(root):
            folder = path.relative_to(root).parts[0]
            if folder not in GENERATORS:
                continue
            generator = GENERATORS[folder]
            label = 1 if generator else 0
            rows.append(make_row(rel(path, raw_root), label, "photo", self.name, "test", generator=generator))
        return rows
