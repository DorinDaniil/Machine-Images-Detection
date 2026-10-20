"""Hand-collected images, e.g. the company's own false-positive cases (logos, text screenshots).

Layout under raw_root/local_folder/:
    <content_type>/real/[<anything>/]*.jpg|jpeg|png|webp            label 0
    <content_type>/fake/[<generator>/]*.jpg|jpeg|png|webp           label 1, generator = subfolder
    <content_type>/masks/<same stem>.png                            optional mask for a fake image

<content_type> must be one of manifest.CONTENT_TYPES. Every row gets split `split` ("test").
"""

from pathlib import Path

from ..manifest import CONTENT_TYPES

from .base import Source, make_row, register

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@register
class LocalFolder(Source):
    name = "local_folder"
    license = "internal"
    url = ""
    description = "local hand-collected images: <content_type>/<real|fake>/[<generator>/]*, optional masks/"

    def __init__(self, split: str = "test"):
        self.split = split

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        root = Path(raw_root) / self.name
        if root.is_dir() and any(root.iterdir()):
            return
        print(
            f"[{self.name}] nothing to download: put images under {root}/<content_type>/<real|fake>/\n"
            f"  (content_type in {CONTENT_TYPES}); fakes may sit in a <generator>/ subfolder,\n"
            f"  masks go to {root}/<content_type>/masks/<image stem>.png."
        )

    def rows(self, raw_root: Path) -> list[dict]:
        root = Path(raw_root) / self.name
        if not root.is_dir():
            return []
        rows = []
        for content_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if content_dir.name not in CONTENT_TYPES:
                raise ValueError(f"{content_dir}: folder name must be a content type from {CONTENT_TYPES}")
            for label, kind in ((0, "real"), (1, "fake")):
                kind_dir = content_dir / kind
                if not kind_dir.is_dir():
                    continue
                for f in sorted(kind_dir.rglob("*")):
                    if f.suffix.lower() not in IMAGE_EXTS:
                        continue
                    sub = f.relative_to(kind_dir).parts
                    generator = sub[0].lower() if len(sub) > 1 else "unknown"
                    mask = content_dir / "masks" / f"{f.stem}.png"
                    rows.append(
                        make_row(
                            f.relative_to(raw_root),
                            label,
                            content_dir.name,
                            self.name,
                            self.split,
                            mask_path=mask.relative_to(raw_root) if label and mask.is_file() else "",
                            generator=generator,
                        )
                    )
        return rows
