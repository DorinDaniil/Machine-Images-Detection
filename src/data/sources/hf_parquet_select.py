"""Hugging Face parquet shards read with pyarrow range requests: only the row groups that hold wanted rows.

For real-image sets where the generic `hf_parquet` kind (rows streamed in file order) falls short: rows
sorted by class or department (shards visited in a seeded random order, at most `per_file` images each),
metadata filters (`where` values, `max_year` on a date column), one image repeated over many QA rows
(sha1 dedup, always on), images in a binary, list or several columns. Filter columns are read first
(row groups in parallel), then the image column only for row groups with a wanted row. Original bytes
are written; nothing else touches the disk.

    kind: custom
    adapter: hf_parquet_select
    args:
      repo: Alejandro98/biomedica_microscopy_imagining_subset_parquet
      files: ["data/*.parquet"]                     # fnmatch patterns on the repo file names
      image_column: image                           # or a list of columns, each one is exported
      where: {image_secondary_label: [immunoblot]}  # value in list (any element of a list cell)
      year_column: article_date                     # year, date string or unix seconds
      max_year: 2021
      per_file: 100
      content_type: biomed
"""

import fnmatch
import hashlib
import io
import random
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from PIL import Image

from .base import Source, make_row, register

EXT = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tif", "BMP": ".bmp", "WEBP": ".webp", "GIF": ".gif"}


def year_of(value) -> int | None:
    if value is None or value == "":
        return None
    if hasattr(value, "year"):
        return value.year
    if isinstance(value, (int, float)) or str(value).strip().isdigit():
        v = int(value)
        return v if v < 10000 else datetime.fromtimestamp(v, tz=timezone.utc).year
    found = re.search(r"(1[5-9]\d\d|20\d\d)", str(value))
    return int(found.group(1)) if found else None


def image_bytes(cell) -> list[bytes]:
    """Encoded images in a cell: {bytes, path} struct, raw bytes, or a list of those."""
    if isinstance(cell, (bytes, bytearray)):
        return [bytes(cell)]
    if isinstance(cell, dict):
        return [cell["bytes"]] if cell.get("bytes") else []
    return [b for c in cell for b in image_bytes(c)] if isinstance(cell, list) else []


@register
class HFParquetSelect(Source):
    name = "hf_parquet_select"
    license = "see dataset card"

    def __init__(self, repo: str, content_type: str, files: list[str] | None = None,
                 image_column: str | list[str] = "image", where: dict | None = None, year_column: str | None = None,
                 max_year: int | None = None, per_file: int | None = None, min_side: int = 0, seed: int = 0,
                 manifest_split: str = "train"):
        self.repo, self.files, self.seed = repo, files or ["*.parquet"], seed
        self.columns = [image_column] if isinstance(image_column, str) else list(image_column)
        self.where = {k: {str(x) for x in v} for k, v in (where or {}).items()}
        self.year_column, self.max_year, self.per_file = year_column, max_year, per_file
        self.content_type, self.min_side, self.manifest_split = content_type, min_side, manifest_split
        self.url, self.description = f"https://huggingface.co/datasets/{repo}", f"{repo} ({', '.join(self.files)})"

    def shard_names(self) -> list[str]:
        names = HfApi().list_repo_files(self.repo, repo_type="dataset")
        names = sorted(n for n in names if n.endswith(".parquet") and any(fnmatch.fnmatch(n, p) for p in self.files))
        random.Random(self.seed).shuffle(names)
        return names

    def wanted_rows(self, pf: pq.ParquetFile, opener=None) -> dict[int, list[int]]:
        """{row group: [row offsets]} passing `where` and `max_year`; `opener()` gives extra handles for parallel reads."""
        keep = [True] * pf.metadata.num_rows
        filters = list(self.where) + ([self.year_column] if self.year_column and self.max_year else [])
        if filters:
            def part(g):
                with opener() as f:
                    return pq.ParquetFile(f, metadata=pf.metadata).read_row_group(g, columns=filters)
            if opener is None:
                table = pf.read(columns=filters)
            else:
                with ThreadPoolExecutor(8) as pool:
                    table = pa.concat_tables(pool.map(part, range(pf.metadata.num_row_groups)))
            for col, allowed in self.where.items():
                cells = table.column(col).to_pylist()
                keep = [k and any(str(x) in allowed for x in (c if isinstance(c, list) else [c])) for k, c in zip(keep, cells)]
            if self.year_column and self.max_year:
                years = [year_of(c) for c in table.column(self.year_column).to_pylist()]
                keep = [k and y is not None and y <= self.max_year for k, y in zip(keep, years)]
        groups, start = {}, 0
        for g in range(pf.metadata.num_row_groups):
            n = pf.metadata.row_group(g).num_rows
            if rows := [i for i in range(n) if keep[start + i]]:
                groups[g] = rows
            start += n
        return groups

    def export(self, pf: pq.ParquetFile, out: Path, stem: str, quota: int, seen: set, opener=None) -> list[str]:
        """Writes up to `quota` new images of one parquet file, row groups in a seeded random order."""
        groups, written = self.wanted_rows(pf, opener), []
        order = sorted(groups)
        random.Random(f"{self.seed}{stem}").shuffle(order)
        for g in order:
            table = pf.read_row_group(g, columns=self.columns)
            for r in groups[g]:
                for j, c in enumerate(self.columns):
                    for k, data in enumerate(image_bytes(table.column(c)[r].as_py())):
                        if len(written) >= quota:
                            return written
                        digest = hashlib.sha1(data).hexdigest()
                        if digest in seen:
                            continue
                        try:
                            with Image.open(io.BytesIO(data)) as im:
                                size, fmt = im.size, im.format
                        except Exception:
                            continue
                        if min(size) >= self.min_side:
                            seen.add(digest)
                            written.append(f"{stem}_{g:03d}_{r:05d}_{j}{k}{EXT.get(fmt, '.png')}")
                            (out / written[-1]).write_bytes(data)
        return written

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        out = Path(raw_root) / self.name
        if (out / "index.csv").exists():
            return
        out.mkdir(parents=True, exist_ok=True)
        fs, seen, names = HfFileSystem(), set(), []
        for i, shard in enumerate(self.shard_names()):
            left = 10**9 if max_items is None else max_items - len(names)
            if left <= 0:
                break
            print(f"   {self.repo}: {shard} ({len(names)} so far)")
            path = f"datasets/{self.repo}/{shard}"
            opener = lambda: fs.open(path, "rb", block_size=1 << 16)  # noqa: E731
            with fs.open(path, "rb", block_size=1 << 18) as f:
                names += self.export(pq.ParquetFile(f), out, f"{i:04d}", min(left, self.per_file or left), seen, opener)
        (out / "index.csv").write_text("file\n" + "".join(f"{n}\n" for n in names), encoding="utf-8")

    def rows(self, raw_root: Path) -> list[dict]:
        index = Path(raw_root) / self.name / "index.csv"
        if not index.exists():
            return []
        return [make_row(Path(self.name) / f, 0, self.content_type, self.name, self.manifest_split)
                for f in index.read_text(encoding="utf-8").splitlines()[1:] if f]
