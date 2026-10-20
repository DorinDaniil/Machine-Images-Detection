"""AIGIBench (NeurIPS 2025, arXiv 2505.12335, github.com/HorizonTEL/AIGIBench): the 25 test subsets.

HF repo `HorizonTEL/AIGIBench` (CC BY-NC-SA 4.0; also on Baidu Netdisk). test/<subset>.zip, 176 GB
in total, each unpacking to <subset>/0_real/* and <subset>/1_fake/* (mostly PNG, real and fake
counts equal per subset). The 25 subsets cover the paper's five groups: GAN noise-to-image
(ProGAN, R3GAN, StyleGAN3, StyleGAN-XL, WFIR), text-to-image diffusion (SDXL, SD3, FLUX1-dev,
DALLE-3, Imagen3, Midjourney-V6, GLIDE), GAN deepfakes / face swaps (BlendFace, E4S, FaceSwap,
InSwap, SimSwap, StyleSwim), personalized diffusion (BLIP(-Diffusion), InstantID, Infinite-ID,
IP-Adapter, PhotoMaker) and crawls of AI communities / social media (CommunityAI, SocialRF;
generator "unknown"). Group membership of WFIR, E4S and StyleSwim is inferred from the names.
The train/ and val/ zips (ProGAN, SD-v1.4) are not used. All rows split="test", content_type "photo".

`max_items` is spread over the 25 zips and only the chosen members are read through HTTP range
requests; without a cap every zip is downloaded (up to 24 GB each), extracted and deleted.
"""

import math
from pathlib import Path

from ._fake_files import real_fake_rows, slug
from ._fake_hub import extract_hub_zip, hub_files
from .base import Source, register

REPO_ID = "HorizonTEL/AIGIBench"
GENERATORS = {
    "ProGAN": "progan", "R3GAN": "r3gan", "StyleGAN3": "stylegan3", "StyleGAN-XL": "stylegan-xl",
    "WFIR": "wfir", "SDXL": "sdxl", "SD3": "sd3", "FLUX1-dev": "flux.1-dev", "DALLE-3": "dalle-3",
    "Imagen3": "imagen-3", "Midjourney": "midjourney-v6", "GLIDE": "glide", "BLIP": "blip-diffusion",
    "BlendFace": "blendface", "E4S": "e4s", "FaceSwap": "faceswap", "InSwap": "inswap",
    "SimSwap": "simswap", "StyleSwim": "styleswim", "InstantID": "instantid", "Infinite-ID": "infinite-id",
    "IP-Adapter": "ip-adapter", "PhotoMaker": "photomaker", "CommunityAI": "unknown", "SocialRF": "unknown",
}


@register
class AIGIBench(Source):
    name = "aigibench"
    license = "CC BY-NC-SA 4.0"
    url = "https://huggingface.co/datasets/HorizonTEL/AIGIBench"
    description = "AIGIBench test: 25 subsets (GANs, T2I, face swaps, personalization, in-the-wild)"

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        root = Path(raw_root) / self.name / "test"
        files = hub_files(REPO_ID, "test/", suffix=".zip")
        quota = None if max_items is None else math.ceil(max_items / len(files))
        for filename in files:
            count = extract_hub_zip(REPO_ID, filename, root, quota)
            print(f"   {filename}: {count} images")

    def rows(self, raw_root: Path) -> list[dict]:
        root = Path(raw_root) / self.name / "test"
        if not root.is_dir():
            return []
        rows = []
        for subset in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            generator = GENERATORS.get(subset.name, slug(subset.name))
            rows += real_fake_rows(Path(raw_root), subset, self.name, generator)
        return rows
