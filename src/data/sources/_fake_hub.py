"""Hugging Face download helpers shared by the fake / benchmark adapters (registers no source).

Finished shards / zips leave a marker in <root>/_done/, so download() is safe to call twice.
"""

import math
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

from ._fake_files import IMAGE_EXTS
from .base import take


def hub_files(repo_id: str, prefix: str, suffix: str = ".parquet") -> list[str]:
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(suffix))


def spread(items: list, n: int) -> list:
    """`n` items evenly spaced over `items` (all of them when n >= len(items))."""
    if n >= len(items):
        return list(items)
    step = len(items) / n
    return [items[int(i * step + step / 2)] for i in range(n)]


def pick_indices(num_rows: int, quota: int | None) -> list[int]:
    """All row indices, or a sorted deterministic random subset of `quota` of them."""
    return sorted(take(list(range(num_rows)), quota))


def iter_parquet(path: Path, columns: list[str], quota: int | None) -> Iterator[dict]:
    """Rows of a local parquet file; with `quota`, a random subset spread over the whole file."""
    with pq.ParquetFile(path, buffer_size=1 << 24) as pf:
        wanted = set(pick_indices(pf.metadata.num_rows, quota))
        last = max(wanted, default=-1)
        index = 0
        for batch in pf.iter_batches(batch_size=16, columns=columns):
            for row in batch.to_pylist():
                if index in wanted:
                    yield row
                index += 1
            if index > last:
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

    Without a cap the zip is downloaded, extracted and deleted. With a cap only `max_items` members,
    evenly spaced over the sorted names (so every folder gets its share), are fetched through HTTP
    range requests and the zip never hits the disk.
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
                count = _extract(zf, spread(_zip_images(zf), max_items), dest)
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
