"""Bank of generator decoders (VAEs) used to re-decode real images.

decode(encode(x)) keeps the content and stamps the decoder's trace on every pixel.
Products change monthly, decoder families change yearly, so the bank is small and stable.
"""

import numpy as np
import torch
from PIL import Image

DECODER_BANK = {
    "sd15": {
        "repo": "stabilityai/sd-vae-ft-mse",
        "subfolder": None,
        "note": "Stable Diffusion 1.x / 2.x family, 4-channel latent, f8. MIT.",
    },
    "sdxl": {
        "repo": "madebyollin/sdxl-vae-fp16-fix",
        "subfolder": None,
        "note": "SDXL family, 4-channel latent, f8. MIT.",
    },
    "flux": {
        "repo": "black-forest-labs/FLUX.1-schnell",
        "subfolder": "vae",
        "note": "FLUX.1 / SD3 family, 16-channel latent, f8. Apache-2.0, gated with auto approval.",
    },
}


def _to_multiple(x: int, m: int) -> int:
    return int(np.ceil(x / m) * m)


class DecoderBank:
    """Lazily loads VAEs from `DECODER_BANK` and re-decodes PIL images."""

    def __init__(self, names: list[str], device: str = "cuda", dtype: torch.dtype = torch.float32):
        unknown = set(names) - set(DECODER_BANK)
        if unknown:
            raise KeyError(f"Unknown decoders {unknown}. Available: {list(DECODER_BANK)}")
        self.names = list(names)
        self.device = device
        self.dtype = dtype
        self._models: dict[str, torch.nn.Module] = {}

    def _load(self, name: str):
        if name not in self._models:
            from diffusers import AutoencoderKL

            spec = DECODER_BANK[name]
            vae = AutoencoderKL.from_pretrained(spec["repo"], subfolder=spec["subfolder"], torch_dtype=self.dtype)
            vae.enable_tiling()
            self._models[name] = vae.to(self.device).eval()
        return self._models[name]

    @torch.no_grad()
    def redecode(self, name: str, image: Image.Image) -> Image.Image:
        """Return decode(encode(image)) with exactly the input size."""
        vae = self._load(name)
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
        h, w = arr.shape[:2]
        pad_h, pad_w = _to_multiple(h, 8) - h, _to_multiple(w, 8) - w
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        x = torch.from_numpy(arr).permute(2, 0, 1)[None].to(self.device, self.dtype)
        latents = vae.encode(x).latent_dist.mode()
        y = vae.decode(latents).sample[0, :, :h, :w]

        out = ((y.float().clamp(-1, 1) + 1) * 127.5 + 0.5).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(out.astype(np.uint8))
