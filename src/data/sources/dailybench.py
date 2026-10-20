"""DailyBench (arXiv 2607.24016, https://dailybench.github.io): FakeBench + ManipulationBench.

HF repo `WhiteJiangzz/DailyBench` (Apache-2.0; `Misaka426/DailyBench` is an identical mirror,
ModelScope `WhiteJiang/DailyBench` another). 14 zips, ~20 GB in total:

- FakeBench/<subset>.zip: full synthesis by SD3.5-Large, FLUX.1, FLUX.2, Qwen-Image-2512,
  Z-Image, Nano Banana 2, GPT-Image 2, each paired with real photos
  (<subset>/0_real/NNN/*.jpg, <subset>/1_fake/NNN/*.jpg; GPT-Image 2 and Nano Banana 2 have only
  ~250 fakes, the open models ~12-20k).
- ManipulationBench/<subset>.zip: object-level Remove / Replace edits of real photos by FLUX-Fill
  (object and random masks), FLUX.2-klein, Qwen-Image-Edit-2511, Step1X-Edit-v1p2, Nano Banana 2,
  GPT-Image 2 (fake file names end in _Remove / _Replace). No masks are released, so mask_path
  stays empty; generic generators get an "-edit" suffix to keep them apart from FakeBench.

Real images repeat across subsets under the same relative names; they are listed once per subset
(the benchmark evaluates each subset as its own real/fake pair). `Qwen-Image_train.zip` (training
data) is skipped. All rows are split="test".

`max_items` is spread evenly over the 14 zips; with a cap only the chosen members are read through
HTTP range requests, without a cap every zip is downloaded, extracted and deleted.
"""

import math
from pathlib import Path

from ._fake_files import real_fake_rows
from ._fake_hub import extract_hub_zip
from .base import Source, register

REPO_ID = "WhiteJiangzz/DailyBench"

# zip in the repo -> generator slug of its fakes
SUBSETS = {
    "FakeBench/SD35.zip": "sd3.5-large",
    "FakeBench/FLUX.zip": "flux.1",
    "FakeBench/FLUX2.zip": "flux.2",
    "FakeBench/Qwen-Image-2512.zip": "qwen-image-2512",
    "FakeBench/ZImage.zip": "z-image",
    "FakeBench/NanoBanana2.zip": "nano-banana-2",
    "FakeBench/GPT-Image2.zip": "gpt-image-2",
    "ManipulationBench/FLUX_Object.zip": "flux.1-fill",
    "ManipulationBench/FLUX_Random.zip": "flux.1-fill",
    "ManipulationBench/FLUX2-klein.zip": "flux.2-klein-edit",
    "ManipulationBench/Qwen-Image-Edit-2511.zip": "qwen-image-edit-2511",
    "ManipulationBench/Step1X-Edit-v1p2.zip": "step1x-edit-v1.2",
    "ManipulationBench/NanoBanana2.zip": "nano-banana-2-edit",
    "ManipulationBench/GPT-Image2.zip": "gpt-image-2-edit",
}


@register
class DailyBench(Source):
    name = "dailybench"
    license = "Apache-2.0"
    url = "https://huggingface.co/datasets/WhiteJiangzz/DailyBench"
    description = "DailyBench FakeBench (7 modern generators) + ManipulationBench (7 editors), no masks"

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        root = Path(raw_root) / self.name
        quota = None if max_items is None else math.ceil(max_items / len(SUBSETS))
        for filename in SUBSETS:
            bench = filename.split("/")[0]
            count = extract_hub_zip(REPO_ID, filename, root / bench, quota)
            print(f"   {filename}: {count} images")

    def rows(self, raw_root: Path) -> list[dict]:
        root = Path(raw_root) / self.name
        rows = []
        for filename, generator in SUBSETS.items():
            bench, subset = filename[: -len(".zip")].split("/")
            rows += real_fake_rows(Path(raw_root), root / bench / subset, self.name, generator)
        return rows
