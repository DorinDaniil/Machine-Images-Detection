"""Training fakes without generators: a real photo passed through a generator's VAE.

decode(encode(x)) keeps the content and the size; the only difference from x is the trace of
the decoder. Every latent-diffusion model ends with such a decoder.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

VAES = {
    "sd15": ("stabilityai/sd-vae-ft-mse", None),          # Stable Diffusion 1.x / 2.x
    "sdxl": ("madebyollin/sdxl-vae-fp16-fix", None),      # SDXL
    "flux": ("black-forest-labs/FLUX.1-schnell", "vae"),  # FLUX.1 / SD3 family; gated, accept the license
}


class VAE:
    def __init__(self, name: str, device: str = "cuda"):
        from diffusers import AutoencoderKL

        repo, subfolder = VAES[name]
        self.vae = AutoencoderKL.from_pretrained(repo, subfolder=subfolder).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> Image.Image:
        x = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1
        h, w = x.shape[:2]
        x = np.pad(x, ((0, -h % 8), (0, -w % 8), (0, 0)), mode="reflect")  # VAE needs sides divisible by 8
        x = torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device)
        y = self.vae.decode(self.vae.encode(x).latent_dist.mode()).sample[0, :, :h, :w]
        y = ((y.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        return Image.fromarray(y.permute(1, 2, 0).cpu().numpy())


def shrink(image: Image.Image, max_side: int) -> Image.Image:
    scale = max_side / max(image.size)
    return image if scale >= 1 else image.resize((round(image.width * scale), round(image.height * scale)), Image.BICUBIC)


def make_pairs(images: list[Path], vae_names: list[str], out_dir, device: str = "cuda", max_side: int = 1024) -> list[Path]:
    """For every image writes <out_dir>/<name>/real.png and <vae>.png for each VAE. Returns the folders."""
    out_dir = Path(out_dir)
    todo = [p for p in images if not all((out_dir / Path(p).stem / f"{v}.png").exists() for v in vae_names)]
    vaes = {name: VAE(name, device) for name in vae_names} if todo else {}
    for path in tqdm(todo, desc=f"re-decoding with {', '.join(vae_names)}"):
        pair = out_dir / Path(path).stem
        pair.mkdir(parents=True, exist_ok=True)
        real = shrink(Image.open(path).convert("RGB"), max_side)
        real.save(pair / "real.png")
        for name, vae in vaes.items():
            vae(real).save(pair / f"{name}.png")
    return [out_dir / Path(p).stem for p in images]
