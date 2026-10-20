"""Download and listing helpers shared by the fake / benchmark adapters (registers no source).

Conventions used by those adapters:
- every source keeps its files under raw_root/<name>/;
- images are written with their original bytes (never re-encoded);
- converted masks are binary PNGs (255 = edited region) saved next to the image as <stem>.mask.png;
- sources exported from parquet/arrow shards write raw_root/<name>/index.csv (path relative to
  raw_root, label, generator, content_type, mask_path) and their rows() read it back;
- finished shards / zips leave a marker in raw_root/<name>/_done/, so download() is safe to repeat.
"""

import csv
import io
import math
import re
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from PIL import Image

from .base import make_row, take

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
FORMAT_EXT = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tif", "BMP": ".bmp", "WEBP": ".webp", "GIF": ".gif"}
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


def hub_files(repo_id: str, prefix: str, suffix: str = ".parquet") -> list[str]:
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(suffix))


def spread(items: list, n: int) -> list:
    """`n` items evenly spaced over `items` (all of them when n >= len(items))."""
    if n >= len(items):
        return list(items)
    step = len(items) / n
    return [items[int(i * step + step / 2)] for i in range(n)]


def iter_parquet(path: Path, columns: list[str], quota: int | None) -> Iterator[dict]:
    """Rows of a local parquet file; with `quota`, every k-th row so the picks span the whole file."""
    pf = pq.ParquetFile(path, buffer_size=1 << 24)
    stride = 1 if quota is None else max(1, pf.metadata.num_rows // max(quota, 1))
    index = kept = 0
    for batch in pf.iter_batches(batch_size=16, columns=columns):
        for row in batch.to_pylist():
            if index % stride == 0 and (quota is None or kept < quota):
                kept += 1
                yield row
            index += 1
        if quota is not None and kept >= quota:
            return


def process_hub_shards(
    repo_id: str, files: list[str], root: Path, max_items: int | None, export: Callable[[Path, int | None], int]
) -> None:
    """Download shards one at a time, call export(local_shard, per_shard_quota), delete the shard."""
    quota = None if max_items is None else math.ceil(max_items / max(len(files), 1))
    for filename in files:
        marker = root / "_done" / (filename.replace("/", "__") + ".done")
        if marker.exists():
            continue
        print(f"   {repo_id}: {filename}")
        local = Path(hf_hub_download(repo_id, filename, repo_type="dataset", local_dir=root / "_shards"))
        count = export(local, quota)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(count))
        local.unlink()


def extract_hub_zip(repo_id: str, filename: str, dest: Path, max_items: int | None) -> int:
    """Extract the images of a zip stored in a HF dataset repo into `dest`.

    Without a cap the zip is downloaded, extracted and deleted. With a cap only a deterministic
    random subset of members is fetched through HTTP range requests (the zip never hits the disk).
    """
    marker = dest / "_done" / (filename.replace("/", "__") + ".done")
    if marker.exists():
        return int(marker.read_text())
    if max_items is None:
        local = Path(hf_hub_download(repo_id, filename, repo_type="dataset", local_dir=dest / "_zips"))
        with zipfile.ZipFile(local) as zf:
            count = _extract(zf, _zip_images(zf), dest)
        local.unlink()
    else:
        with HfFileSystem().open(f"datasets/{repo_id}/{filename}", "rb", block_size=1 << 20) as f:
            with zipfile.ZipFile(f) as zf:
                count = _extract(zf, take(_zip_images(zf), max_items), dest)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(count))
    return count


def _zip_images(zf: zipfile.ZipFile) -> list[str]:
    return sorted(n for n in zf.namelist() if Path(n).suffix.lower() in IMAGE_EXTS and ".." not in Path(n).parts)


def _extract(zf: zipfile.ZipFile, names: list[str], dest: Path) -> int:
    for name in names:
        target = dest / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))
    return len(names)


def manual_notice(name: str, root: Path, steps: str) -> None:
    print(f"[{name}] manual download required, nothing was fetched.\n{steps.rstrip()}\n  expected root: {root}")
