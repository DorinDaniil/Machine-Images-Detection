"""Training fakes without generators: a real image passed through a generator's decoder.

decode(encode(x)) keeps the content and the size; the only difference from x is the trace of the
decoder. Every listed autoencoder is on the Hub without an access request.

Families matter for honest tests: hold a whole family out, not just one checkpoint.
"""

import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, JpegImagePlugin
from tqdm import tqdm

#        name: (repo, subfolder, diffusers class, downsampling factor, family)
DECODERS = {
    "sd":    ("stabilityai/sd-vae-ft-mse", None, "AutoencoderKL", 8, "kl-f8-c4"),          # SD 1.x / 2.x
    "sdxl":  ("madebyollin/sdxl-vae-fp16-fix", None, "AutoencoderKL", 8, "kl-f8-c4"),      # SDXL, PixArt-Sigma
    "flux1": ("Tongyi-MAI/Z-Image-Turbo", "vae", "AutoencoderKL", 8, "kl-f8-c16"),         # FLUX.1, Z-Image, HiDream, BAGEL
    "flux2": ("black-forest-labs/FLUX.2-klein-4B", "vae", "AutoencoderKLFlux2", 8, "kl-f8-c32"),  # FLUX.2
    "qwen":  ("Qwen/Qwen-Image", "vae", "AutoencoderKLQwenImage", 8, "wan-f8-c16"),        # Qwen-Image, Wan 2.1
    "wan22": ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "vae", "AutoencoderKLWan", 16, "wan-f16-c48"),
    "dcae":  ("mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers", None, "AutoencoderDC", 32, "dc-f32-c32"),  # Sana
    "movq":  ("kandinsky-community/kandinsky-2-2-decoder", "movq", "VQModel", 8, "vq-f8"),  # Kandinsky 2.2
}
VIDEO = {"AutoencoderKLQwenImage", "AutoencoderKLWan"}  # these take [B, C, T, H, W]


def family(name: str) -> str:
    return DECODERS[name][4]


class Decoder:
    """Callable: PIL image -> the same image re-decoded, same size, 8-bit."""

    def __init__(self, name: str, device: str = "cuda", dtype=torch.float32):
        import diffusers

        repo, subfolder, cls, self.factor, _ = DECODERS[name]
        self.cls, self.device, self.dtype = cls, device, dtype
        self.model = getattr(diffusers, cls).from_pretrained(repo, subfolder=subfolder, torch_dtype=dtype).to(device).eval()

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> Image.Image:
        x = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1
        h, w = x.shape[:2]
        f = self.factor
        x = np.pad(x, ((0, -h % f), (0, -w % f), (0, 0)), mode="reflect")
        x = torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device, self.dtype)
        if self.cls in VIDEO:
            x = x[:, :, None]
        out = self.model.encode(x)
        if self.cls == "AutoencoderDC":
            z = out.latent
        elif self.cls == "VQModel":
            z = out.latents  # decode() quantizes
        else:
            z = out.latent_dist.mode()
        y = self.model.decode(z).sample
        if self.cls in VIDEO:
            y = y[:, :, 0]
        y = ((y[0, :, :h, :w].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        return Image.fromarray(y.permute(1, 2, 0).cpu().numpy())


def save_like(image: Image.Image, src: Image.Image, stem: Path) -> Path:
    """Save with the compression history of `src`: same JPEG tables and subsampling, or PNG if src is lossless.

    Without this the detector learns "JPEG grid = real, no grid = fake".
    """
    if src.format == "JPEG":
        sampling = JpegImagePlugin.get_sampling(src)
        extra = {"subsampling": sampling} if sampling != -1 else {}
        path = stem.with_suffix(".jpg")
        image.save(path, format="JPEG", qtables=src.quantization, **extra)
    else:
        path = stem.with_suffix(".png")
        image.save(path)
    return path


def find(pair: Path, name: str) -> Path:
    for ext in (".jpg", ".png"):
        if (pair / f"{name}{ext}").exists():
            return pair / f"{name}{ext}"
    raise FileNotFoundError(pair / name)


def make_pairs(images: list, names: list[str], out_dir, device: str = "cuda", max_side: int = 1024) -> list[Path]:
    """For every image: <out_dir>/<stem>/real.* (the original file, or shrunk to max_side) and <name>.* per decoder,
    all with the same compression history. Returns the pair folders."""
    out_dir = Path(out_dir)

    def done(p):
        return all(any((out_dir / Path(p).stem / f"{n}{e}").exists() for e in (".jpg", ".png")) for n in names + ["real"])

    todo = [p for p in images if not done(p)]
    decoders = {n: Decoder(n, device) for n in names} if todo else {}
    for path in tqdm(todo, desc=f"re-decoding with {', '.join(names)}"):
        pair = out_dir / Path(path).stem
        pair.mkdir(parents=True, exist_ok=True)
        src = Image.open(path)
        src.load()
        if max(src.size) <= max_side:
            real = src.convert("RGB")
            shutil.copyfile(path, (pair / "real").with_suffix(".jpg" if src.format == "JPEG" else ".png"))
        else:  # shrinking erases the old history; both get one fresh compression with the same tables
            scale = max_side / max(src.size)
            real = src.convert("RGB").resize((round(src.width * scale), round(src.height * scale)), Image.BICUBIC)
            save_like(real, src, pair / "real")
        for name, decoder in decoders.items():
            save_like(decoder(real), src, pair / name)
    return [out_dir / Path(p).stem for p in images]
