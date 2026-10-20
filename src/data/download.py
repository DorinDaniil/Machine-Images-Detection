"""Two open datasets from Hugging Face.

Real photos:      detection-datasets/coco, COCO val2017 (Flickr photos, about 640 px).
Generated images: Qwen/Qwen-Image-Bench (Apache-2.0), 1000 images per model, the same
                  prompts for every model: gpt-image-2, nano-banana-pro, FLUX.2-pro,
                  Seedream-4.5, Imagen-4.0, Qwen-Image, HunyuanImage-3.0, kling_v2_1 and others.
"""

from pathlib import Path

from tqdm import tqdm

COCO = "detection-datasets/coco"
FAKES = "Qwen/Qwen-Image-Bench"


def download_coco(out_dir, n: int, min_side: int = 384) -> list[Path]:
    """First n COCO val2017 photos whose shorter side is at least min_side, saved as the original JPEG."""
    from datasets import Image, load_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    have = sorted(out_dir.glob("*.jpg"))
    if len(have) >= n:
        return have[:n]

    ds = load_dataset(COCO, split="val", streaming=True).cast_column("image", Image(decode=False))
    paths = []
    with tqdm(total=n, desc="COCO") as bar:
        for row in ds:
            if min(row["width"], row["height"]) < min_side:
                continue
            path = out_dir / f"{row['image_id']:012d}.jpg"
            path.write_bytes(row["image"]["bytes"])
            paths.append(path)
            bar.update()
            if len(paths) == n:
                break
    return paths


def download_fakes(out_dir, generators: list[str], n_per_generator: int) -> dict[str, list[Path]]:
    """First n images of each generator folder of Qwen-Image-Bench (same prompt ids for every model)."""
    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi().list_repo_files(FAKES, repo_type="dataset")
    result = {}
    for gen in generators:
        names = sorted(f for f in files if f.startswith(f"images/{gen}/"))[:n_per_generator]
        if not names:
            raise ValueError(f"no folder images/{gen} in {FAKES}")
        result[gen] = [Path(hf_hub_download(FAKES, f, repo_type="dataset", local_dir=out_dir))
                       for f in tqdm(names, desc=gen)]
    return result
