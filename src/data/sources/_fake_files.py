"""File helpers shared by the fake / benchmark adapters (registers no source).

Conventions used by those adapters:
- every source keeps its files under raw_root/<name>/;
- images are written with their original bytes (never re-encoded);
- converted masks are binary PNGs (255 = edited region) saved next to the image as <stem>.mask.png;
- sources exported from parquet/arrow shards write raw_root/<name>/index.csv (path relative to
  raw_root, label, generator, content_type, mask_path) and their rows() read it back.
"""

import csv
import io
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .base import make_row

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
FORMAT_EXT = {
    "JPEG": ".jpg", "MPO": ".jpg", "PNG": ".png", "TIFF": ".tif", "BMP": ".bmp", "WEBP": ".webp", "GIF": ".gif",
}
MASK_SUFFIX = ".mask.png"
INDEX_COLUMNS = ["path", "label", "generator", "content_type", "mask_path"]


def slug(name: str) -> str:
    """Short lowercase generator id: 'GPT Image 2' -> 'gpt-image-2', 'nano_banana_2' -> 'nano-banana-2'."""
    text = re.sub(r"[^a-z0-9.]+", "-", str(name).strip().lower()).strip("-.")
    return text or "unknown"


def rel(path: Path, raw_root: Path) -> str:
    return Path(path).relative_to(raw_root).as_posix()


def list_images(folder: Path) -> list[Path]:
    """Image files under `folder` (sorted), without the converted *.mask.png files."""
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS and not p.name.endswith(MASK_SUFFIX)
    )


def real_fake_rows(
    raw_root: Path, folder: Path, source: str, generator: str, content_type: str = "photo", split: str = "test"
) -> list[dict]:
    """Rows for a tree with `0_real` / `1_fake` folders anywhere below `folder`."""
    rows = []
    for path in list_images(folder):
        parts = path.relative_to(folder).parts[:-1]
        if "0_real" in parts:
            label = 0
        elif "1_fake" in parts:
            label = 1
        else:
            continue
        rows.append(make_row(rel(path, raw_root), label, content_type, source, split, generator=generator))
    return rows


def save_image_bytes(data: bytes, stem_path: Path) -> Path | None:
    """Write encoded bytes unchanged as <stem_path>.<ext of the real format>; None if unreadable."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            ext = FORMAT_EXT.get(im.format)
    except Exception:
        return None
    if ext is None:
        return None
    path = stem_path.with_name(stem_path.name + ext)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def save_binary_mask(src: bytes | Path, dst: Path, threshold: int = 127) -> Path:
    """Convert a mask (L/RGB/RGBA, soft or 0/1 valued) to a binary PNG with 255 = edited region."""
    with Image.open(io.BytesIO(src) if isinstance(src, bytes) else src) as im:
        gray = np.asarray(im.convert("L"))
        if im.mode in ("RGBA", "LA") and gray.max() == 0:
            gray = np.asarray(im.getchannel("A"))
    if gray.max() <= 1:
        gray = gray * 255
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(gray > threshold, 255, 0).astype(np.uint8)).save(dst)
    return dst


def mask_path_for(image_path: Path) -> Path:
    return image_path.with_name(image_path.stem + MASK_SUFFIX)


def append_index(root: Path, records: list[dict]) -> None:
    path = root / "index.csv"
    is_new = not path.exists()
    root.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerows(records)


def index_rows(raw_root: Path, source: str, split: str = "test") -> list[dict]:
    """Manifest rows from raw_root/<source>/index.csv (written by the shard exporters)."""
    path = Path(raw_root) / source / "index.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    return [
        make_row(r["path"], int(r["label"]), r["content_type"], source, split, r["mask_path"], r["generator"])
        for r in records
    ]


def manual_notice(name: str, root: Path, steps: str) -> None:
    print(f"[{name}] manual download required, nothing was fetched.\n{steps.rstrip()}\n  expected root: {root}")
