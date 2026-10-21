"""Open datasets from Hugging Face, streamed: only the kept images are written to disk.

OpenFake (ComplexDataLab/OpenFake, CC BY-NC 4.0): real photos and images of many generators, with the
    generator name for every image. core/train: reals from LAION and Pexels; fakes from SD 1.x to FLUX.2,
    GPT-Image-1, Imagen 4, Midjourney 6 and ~70 others. core/test: newer models (GPT-Image-1.5 and 2,
    Nano Banana Pro, Seedream 5, Midjourney 7, FLUX.2-klein-9B, Z-Image-Turbo, ...) and other real
    sources (DOCCI, ImageNet).
SID_Set (saberzl/SID_Set, CC BY 4.0): real (0), fully synthetic (1) and locally edited (2) images,
    with masks of the edited region.

Every image is stored the same way, whatever its label: long side at most 1024, JPEG q95. Images whose
short side is below `min_side` are skipped, also whatever the label. So file format and size carry no
information about the label.
"""

import csv
import io
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

COLUMNS = ["path", "label", "generator", "mask", "source"]


def save_rows(rows: list[dict], path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [{**r, "label": int(r["label"])} for r in csv.DictReader(f)]


def _store(data: bytes, path: Path, max_side: int, min_side: int):
    """Decode, skip if too small, shrink to max_side, save JPEG q95. Returns the stored size or None."""
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
    if min(im.size) < min_side:
        return None
    scale = max_side / max(im.size)
    if scale < 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    im.save(path, format="JPEG", quality=95)
    return im.size


def _stream(repo: str, config: str, split: str, columns: list[str]):
    from datasets import Image as HFImage
    from datasets import load_dataset

    ds = load_dataset(repo, config, split=split, streaming=True)
    for col in columns:
        ds = ds.cast_column(col, HFImage(decode=False))
    return ds


def openfake(split: str, out_dir, n_real: int, n_per_model: int, max_rows: int,
             real_sources=None, max_side: int = 1024, min_side: int = 512) -> list[dict]:
    """Up to n_real reals (optionally only from real_sources) and n_per_model images of every generator,
    from the first max_rows rows of OpenFake core/<split>. Cached in <out_dir>.csv."""
    out_dir = Path(out_dir)
    cache = out_dir.with_suffix(".csv")
    if cache.exists():
        return load_rows(cache)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, counts = [], Counter()
    for i, row in enumerate(tqdm(_stream("ComplexDataLab/OpenFake", "core", split, ["image"]), total=max_rows, desc=f"OpenFake {split}")):
        if i >= max_rows:
            break
        label, model = int(row["label"] == "fake"), row["model"]
        if label == 0 and (counts["real"] >= n_real or (real_sources and model not in real_sources)):
            continue
        if label == 1 and counts[model] >= n_per_model:
            continue
        path = out_dir / f"{i:07d}.jpg"
        if _store(row["image"]["bytes"], path, max_side, min_side):
            counts["real" if label == 0 else model] += 1
            rows.append({"path": str(path), "label": label, "generator": "" if label == 0 else model,
                         "mask": "", "source": f"openfake/{model}" if label == 0 else "openfake"})
    save_rows(rows, cache)
    print(f"OpenFake {split}: {counts['real']} reals, {len(rows) - counts['real']} fakes from {len(counts) - 1} generators")
    return rows


def sid_set(split: str, out_dir, n_per_class: int, max_rows: int, max_side: int = 1024, min_side: int = 512) -> list[dict]:
    """SID_Set: n_per_class reals, fully synthetic and locally edited images (with masks). Cached in <out_dir>.csv."""
    out_dir = Path(out_dir)
    cache = out_dir.with_suffix(".csv")
    if cache.exists():
        return load_rows(cache)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = {0: "real", 1: "full", 2: "edit"}
    rows, counts = [], Counter()
    for i, row in enumerate(tqdm(_stream("saberzl/SID_Set", "default", split, ["image", "mask"]), total=max_rows, desc=f"SID_Set {split}")):
        if i >= max_rows or min(counts[k] for k in names.values()) >= n_per_class:
            break
        kind = names[int(row["label"])]
        if counts[kind] >= n_per_class:
            continue
        path = out_dir / f"{i:07d}.jpg"
        size = _store(row["image"]["bytes"], path, max_side, min_side)
        if not size:
            continue
        mask = ""
        if kind == "edit" and row["mask"] and row["mask"]["bytes"]:
            mask = str(out_dir / f"{i:07d}_mask.png")
            with Image.open(io.BytesIO(row["mask"]["bytes"])) as m:
                m.convert("L").resize(size, Image.NEAREST).point(lambda v: 255 if v > 127 else 0).save(mask)
        counts[kind] += 1
        rows.append({"path": str(path), "label": int(kind != "real"), "generator": "" if kind == "real" else f"sid-{kind}",
                     "mask": mask, "source": f"sid/{kind}"})
    save_rows(rows, cache)
    print(f"SID_Set {split}: {dict(counts)}")
    return rows
