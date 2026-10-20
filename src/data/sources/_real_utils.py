"""Download and listing helpers shared by the real-image adapters (registers no source).

Images are always written with their original bytes (no re-encoding), so JPEG/PNG traces of the
source stay intact. Parquet shards are deleted right after their images are exported, which keeps
the disk footprint close to the size of the exported images.
"""

import io
import random
import zipfile
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from .base import make_row, take

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
FORMAT_EXT = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tif", "BMP": ".bmp", "WEBP": ".webp"}


def list_images(folder: Path) -> list[Path]:
    """Image files under `folder` in a stable (sorted) order."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def image_rows(raw_root: Path, folder: Path, content_type: str, source: str, split: str) -> list[dict]:
    """One label-0 manifest row per image under `folder`, paths relative to raw_root."""
    raw_root = Path(raw_root)
    return [
        make_row(p.relative_to(raw_root).as_posix(), 0, content_type, source, split)
        for p in list_images(folder)
    ]


def save_image_bytes(data: bytes, out_dir: Path, stem: str, min_side: int = 0) -> bool:
    """Write encoded image bytes unchanged. Skips unreadable images and ones below `min_side`."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            size, fmt = im.size, im.format
    except Exception:
        return False
    if min(size) < min_side or fmt not in FORMAT_EXT:
        return False
    (out_dir / f"{stem}{FORMAT_EXT[fmt]}").write_bytes(data)
    return True


def hub_files(repo_id: str, prefixes: tuple[str, ...], suffix: str = ".parquet") -> list[str]:
    """Sorted file names of a HF dataset repo that start with one of `prefixes`."""
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    return sorted(f for f in files if f.startswith(prefixes) and f.endswith(suffix))


def export_parquet_images(parquet: Path, out_dir: Path, min_side: int = 0) -> int:
    """Save every image of a shard's `image` column; a cell is {bytes, path} or raw bytes."""
    count = 0
    index = 0
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=64, columns=["image"]):
        for cell in batch.column(0).to_pylist():
            data = cell.get("bytes") if isinstance(cell, dict) else cell
            if data and save_image_bytes(data, out_dir, f"{parquet.stem}_{index:06d}", min_side):
                count += 1
            index += 1
    return count


def fetch_hub_parquet(
    repo_id: str,
    files: list[str],
    out_dir: Path,
    max_items: int | None,
    min_side: int = 0,
    max_shards: int | None = None,
    export: Callable[[Path, Path], int] | None = None,
) -> None:
    """Download shards in a fixed shuffled order until `max_items` images are exported.

    A `.<shard>.done` marker (holding the image count) is written per shard, so a second call
    skips finished shards and a larger `max_items` continues where the last call stopped.
    `export(parquet, out_dir)` replaces the default single-column export when given.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir.parent / "_shards"
    order = sorted(files)
    random.Random(0).shuffle(order)
    if max_shards is not None:
        order = order[:max_shards]
    for filename in order:
        markers = list(out_dir.glob(".*.done"))
        if max_items is not None and sum(int(m.read_text()) for m in markers) >= max_items:
            break
        marker = out_dir / f".{Path(filename).stem}.done"
        if marker.exists():
            continue
        print(f"   {repo_id}: {filename}")
        local = Path(hf_hub_download(repo_id, filename, repo_type="dataset", local_dir=tmp_dir))
        if export is None:
            count = export_parquet_images(local, out_dir, min_side)
        else:
            count = export(local, out_dir)
        marker.write_text(str(count))
        local.unlink()


def download_url(url: str, dest: Path) -> Path:
    """Stream `url` to `dest` unless it already exists (a .part file guards partial downloads)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(part, "wb") as f:
            for block in response.iter_content(1 << 20):
                f.write(block)
    part.replace(dest)
    return dest


def extract_images(
    zip_path: Path, out_dir: Path, max_items: int | None = None, keep: Callable[[str], bool] | None = None
) -> int:
    """Extract image members of a zip (a deterministic subset when `max_items` is set)."""
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(
            n for n in zf.namelist()
            if Path(n).suffix.lower() in IMAGE_EXTS and ".." not in Path(n).parts and (keep is None or keep(n))
        )
        chosen = take(names, max_items)
        for name in chosen:
            target = out_dir / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(name))
    return len(chosen)
