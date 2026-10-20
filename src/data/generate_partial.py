"""Make AI-edited images with exact ground-truth masks from real images (diffusers inpainting).

For every real row (label=0) of a manifest writes <out>/<pipeline>/<id>/image.png and mask.png
(255 = generated region), and one row in <out>/partial_fakes.csv in the manifest schema
(label=1, generator=<pipeline>, source=generated_<pipeline>, content_type of the input row).

    inpaint     a random region (5-40 % of the image) is repainted; only the masked pixels of the
                output are pasted back, so every pixel outside the mask is the original
    regenerate  img2img over the whole image with strength 0.5-0.9; the mask is all ones

Images are shrunk to --max-side (never enlarged), then the right/bottom edges are cropped to a
multiple of 8 (16 for FLUX), so no extra resampling. CSV paths start with --out.

    python -m src.data.generate_partial --manifest data/manifests/test_real.csv \
        --input-split test --out data/partial_fakes --pipeline sdxl-inpaint --mode inpaint --limit 500
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .manifest import COLUMNS, SPLITS, filter_manifest, read_manifest, write_manifest
from .synth import random_region_mask

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
MIN_SIDE = 64  # smaller images are skipped

# stabilityai/stable-diffusion-2-inpainting was removed from the Hub; sd2-community hosts the same weights.
# FLUX.1-Fill-dev is gated (accept the licence on the Hub, then `huggingface-cli login`), non-commercial.
PIPELINES = {
    "sd2-inpaint": dict(repo="sd2-community/stable-diffusion-2-inpainting", steps=50, guidance=7.5,
                        strength=1.0, multiple=8, cuda_dtype="float16"),
    "sdxl-inpaint": dict(repo="diffusers/stable-diffusion-xl-1.0-inpainting-0.1", steps=30, guidance=8.0,
                         strength=0.99, multiple=8, cuda_dtype="float16"),  # model card: keep strength < 1
    "flux-fill": dict(repo="black-forest-labs/FLUX.1-Fill-dev", steps=50, guidance=30.0,
                      strength=1.0, multiple=16, cuda_dtype="bfloat16"),  # FLUX text encoders overflow in fp16
}

DEFAULT_PROMPTS = [
    "a realistic photo, high detail", "a natural photograph, sharp focus",
    "high quality photo, natural lighting", "a detailed photo of the scene",
    "realistic textures, natural colors", "a clean, well-exposed photograph",
    "photorealistic, fine details", "a candid everyday photo",
    "an ordinary scene, realistic photo", "high resolution photograph, realistic background",
]


def load_pipeline(name: str, device: str, dtype: torch.dtype, offload: bool):
    """Load an inpainting pipeline. diffusers is imported here so the module imports without it."""
    from diffusers import FluxFillPipeline, StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline

    cls = {"sd2-inpaint": StableDiffusionInpaintPipeline, "sdxl-inpaint": StableDiffusionXLInpaintPipeline,
           "flux-fill": FluxFillPipeline}[name]
    pipe = cls.from_pretrained(PIPELINES[name]["repo"], torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    if offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    return pipe


def prepare(image: Image.Image, max_side: int, multiple: int) -> Image.Image:
    """Shrink so the longer side is <= max_side (keep aspect), crop right/bottom to a multiple."""
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        image = image.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
        w, h = image.size
    return image.crop((0, 0, w - w % multiple, h - h % multiple))


def edit(pipe, spec: dict, image: Image.Image, mode: str, prompt: str, rng: np.random.Generator, args):
    """Run the pipeline once. Returns (edited RGB image, uint8 mask with 255 = generated)."""
    w, h = image.size
    if mode == "inpaint":
        mask = random_region_mask(h, w, rng, area_range=(0.05, 0.40)) > 0.5
        strength = spec["strength"]
    else:
        mask = np.ones((h, w), dtype=bool)
        strength = float(rng.uniform(0.5, 0.9))
    mask_u8 = mask.astype(np.uint8) * 255
    out = pipe(
        prompt=prompt, image=image, mask_image=Image.fromarray(mask_u8), height=h, width=w, strength=strength,
        num_inference_steps=args.steps or spec["steps"],
        guidance_scale=spec["guidance"] if args.guidance is None else args.guidance,
        generator=torch.Generator("cpu").manual_seed(int(rng.integers(2**31))),
    ).images[0].convert("RGB")
    if out.size != image.size:
        out = out.resize(image.size, Image.BICUBIC)
    if mode == "inpaint":  # keep original pixels outside the mask bit-exact
        out = Image.fromarray(np.where(mask[..., None], np.asarray(out), np.asarray(image)))
    return out, mask_u8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="manifest CSV; only label=0 rows are used")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--input-split", default=None, choices=list(SPLITS), help="use only real rows of this split (default: all)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pipeline", default="sdxl-inpaint", choices=list(PIPELINES))
    parser.add_argument("--mode", default="inpaint", choices=["inpaint", "regenerate"])
    parser.add_argument("--prompt-file", default=None, help="one prompt per line; default: built-in neutral prompts")
    parser.add_argument("--steps", type=int, default=None, help="default per pipeline: 50 / 30 / 50")
    parser.add_argument("--guidance", type=float, default=None, help="default per pipeline: 7.5 / 8.0 / 30")
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default="test", choices=list(SPLITS), help="split written to the output rows")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default=None, choices=list(DTYPES), help="default: fp16 (bf16 for FLUX) on cuda, else fp32")
    parser.add_argument("--offload", action="store_true", help="enable_model_cpu_offload (needed for FLUX on <40 GB GPUs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    spec = PIPELINES[args.pipeline]
    dtype = args.dtype or (spec["cuda_dtype"] if args.device.startswith("cuda") else "float32")
    prompts = DEFAULT_PROMPTS
    if args.prompt_file:
        prompts = [line.strip() for line in open(args.prompt_file, encoding="utf-8") if line.strip()]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "partial_fakes.csv"
    existing = pd.read_csv(csv_path, dtype=str, keep_default_na=False) if csv_path.exists() else pd.DataFrame(columns=COLUMNS)

    df = filter_manifest(read_manifest(args.manifest, args.data_root), label=0, split=args.input_split)
    if args.limit:
        df = df.head(args.limit)
    pipe = None  # loaded on the first image that needs it

    rows, skipped = [], 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{args.pipeline}/{args.mode}"):
        digest = hashlib.sha1(row["path"].encode("utf-8")).hexdigest()[:12]
        item_dir = out / args.pipeline / f"{args.mode}_{digest}"
        if not args.overwrite and (item_dir / "image.png").exists() and (item_dir / "mask.png").exists():
            rows.append(_row(item_dir, row, args))
            continue

        try:
            image = prepare(Image.open(row["path"]).convert("RGB"), args.max_side, spec["multiple"])
        except Exception as e:  # unreadable file
            print(f"skip {row['path']}: {e}")
            skipped += 1
            continue
        if min(image.size) < MIN_SIDE:
            skipped += 1
            continue

        if pipe is None:
            pipe = load_pipeline(args.pipeline, args.device, DTYPES[dtype], args.offload)
        rng = np.random.default_rng([args.seed, int(digest, 16)])  # per image: reruns and skips stay reproducible
        edited, mask = edit(pipe, spec, image, args.mode, str(rng.choice(prompts)), rng, args)
        item_dir.mkdir(parents=True, exist_ok=True)
        edited.save(item_dir / "image.png")
        Image.fromarray(mask).save(item_dir / "mask.png")
        rows.append(_row(item_dir, row, args))

    new = pd.DataFrame(rows, columns=COLUMNS)
    merged = pd.concat([existing, new]).drop_duplicates("path", keep="last")
    write_manifest(merged, csv_path)
    print(f"partial fakes: {len(merged)} total, {len(new)} from this run, {skipped} skipped -> {csv_path}")


def _row(item_dir: Path, row, args) -> dict:
    return {
        "path": str(item_dir / "image.png"),
        "label": 1,
        "mask_path": str(item_dir / "mask.png"),
        "generator": args.pipeline,
        "content_type": row["content_type"],
        "source": f"generated_{args.pipeline}",
        "split": args.split,
    }


if __name__ == "__main__":
    main()
