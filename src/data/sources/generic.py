"""Generic sources driven by configs/datasets.yaml. Special layouts get their own module instead.

hf_parquet   a Hugging Face dataset streamed row by row; stops at max_items, so nothing
             extra is downloaded. Images are saved with their original bytes.
hf_files     a Hugging Face repo of image files; label / generator read from folder names.
"""

import io
from pathlib import Path

from PIL import Image

from .base import Source, make_row

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _ext_from_bytes(data: bytes) -> str:
    try:
        return "." + (Image.open(io.BytesIO(data)).format or "png").lower().replace("jpeg", "jpg")
    except Exception:
        return ".png"


class HFParquetSource(Source):
    """
    datasets.yaml entry:
      kind: hf_parquet
      repo: huggan/wikiart          # dataset id
      split: train                  # HF split (default train)
      config: null                  # HF config name, if the dataset has several
      image_column: image
      label: 0                      # fixed, or label_column + label_map: {real: 0, fake: 1}
      generator: ""                 # fixed, or generator_column
      content_type: art
      manifest_split: train
      min_side: 512                 # skip smaller images (0 = keep all)
      max_items: 5000
    """

    def __init__(self, name: str, repo: str, content_type: str, split: str = "train", config: str | None = None,
                 image_column: str = "image", label: int | None = None, label_column: str | None = None,
                 label_map: dict | None = None, generator: str = "", generator_column: str | None = None,
                 manifest_split: str = "train", min_side: int = 0, mask_column: str | None = None):
        self.name, self.repo, self.split, self.config = name, repo, split, config
        self.image_column, self.mask_column = image_column, mask_column
        self.label, self.label_column, self.label_map = label, label_column, label_map or {}
        self.generator, self.generator_column = generator, generator_column
        self.content_type, self.manifest_split, self.min_side = content_type, manifest_split, min_side
        self.license, self.url, self.description = "see dataset card", f"https://huggingface.co/datasets/{repo}", f"{repo} ({split})"

    def _label(self, row: dict) -> int:
        if self.label_column is None:
            return int(self.label)
        value = row[self.label_column]
        return int(self.label_map.get(str(value), value))

    def _generator(self, row: dict) -> str:
        return str(row[self.generator_column]).lower() if self.generator_column else self.generator

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        from datasets import Image as HFImage
        from datasets import load_dataset

        out = Path(raw_root) / self.name
        index = out / "index.csv"
        if index.exists():
            return
        out.mkdir(parents=True, exist_ok=True)

        ds = load_dataset(self.repo, self.config, split=self.split, streaming=True)
        ds = ds.cast_column(self.image_column, HFImage(decode=False))
        if self.mask_column:
            ds = ds.cast_column(self.mask_column, HFImage(decode=False))

        lines, kept = ["file,mask,label,generator"], 0
        for i, row in enumerate(ds):
            if max_items is not None and kept >= max_items:
                break
            data = row[self.image_column]["bytes"]
            if data is None:
                continue
            if self.min_side and min(Image.open(io.BytesIO(data)).size) < self.min_side:
                continue
            file = f"{i:08d}{_ext_from_bytes(data)}"
            (out / file).write_bytes(data)
            mask = ""
            if self.mask_column and row[self.mask_column] and row[self.mask_column]["bytes"]:
                mask = f"{i:08d}_mask.png"
                Image.open(io.BytesIO(row[self.mask_column]["bytes"])).convert("L").save(out / mask)
            lines.append(f"{file},{mask},{self._label(row)},{self._generator(row)}")
            kept += 1
        index.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def rows(self, raw_root: Path) -> list[dict]:
        index = Path(raw_root) / self.name / "index.csv"
        if not index.exists():
            return []
        rows = []
        for line in index.read_text(encoding="utf-8").splitlines()[1:]:
            file, mask, label, generator = line.split(",", 3)
            rows.append(make_row(Path(self.name) / file, int(label), self.content_type, self.name, self.manifest_split,
                                 mask_path=Path(self.name) / mask if mask else "", generator=generator))
        return rows


class HFFilesSource(Source):
    """
    datasets.yaml entry:
      kind: hf_files
      repo: OwensLab/CommunityForensics
      repo_type: dataset
      allow_patterns: ["images/**"]        # what to download
      label: 1                             # fixed, or folder_labels: {real: 0, fake: 1} matched on path parts
      generator: ""                        # fixed, or generator_from: 1 (index of the path part under the root)
      content_type: photo
      manifest_split: test
    """

    def __init__(self, name: str, repo: str, content_type: str, repo_type: str = "dataset",
                 allow_patterns: list[str] | None = None, label: int | None = None, folder_labels: dict | None = None,
                 generator: str = "", generator_from: int | None = None, manifest_split: str = "test", min_side: int = 0):
        self.name, self.repo, self.repo_type, self.allow_patterns = name, repo, repo_type, allow_patterns
        self.label, self.folder_labels = label, folder_labels or {}
        self.generator, self.generator_from = generator, generator_from
        self.content_type, self.manifest_split, self.min_side = content_type, manifest_split, min_side
        self.license, self.url, self.description = "see dataset card", f"https://huggingface.co/datasets/{repo}", f"{repo} files"

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        from huggingface_hub import snapshot_download

        out = Path(raw_root) / self.name
        if out.exists() and any(out.rglob("*")):
            return
        snapshot_download(self.repo, repo_type=self.repo_type, allow_patterns=self.allow_patterns, local_dir=out)

    def rows(self, raw_root: Path) -> list[dict]:
        root = Path(raw_root) / self.name
        if not root.is_dir():
            return []
        rows = []
        for f in sorted(root.rglob("*")):
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            parts = f.relative_to(root).parts
            label = self.label
            for part in parts:
                if part in self.folder_labels:
                    label = int(self.folder_labels[part])
            if label is None:
                continue
            generator = parts[self.generator_from].lower() if self.generator_from is not None and len(parts) > self.generator_from else self.generator
            if self.min_side and min(Image.open(f).size) < self.min_side:
                continue
            rows.append(make_row(f.relative_to(raw_root), label, self.content_type, self.name, self.manifest_split, generator=generator))
        return rows
