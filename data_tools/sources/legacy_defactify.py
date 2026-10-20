"""Defactify 4 image track as used by the previous project (local copy, no download).

Layout: raw_root/legacy_defactify/Train/<class_name>/*.jpg|png. Class "coco_image" is real,
every other class is a generator (generator = class name lowercased).

Split: files sorted by name, first 90 % "train", last 10 % "test". By default this is done per
class, so every generator appears in test. The old dataset.py split the concatenation of all
classes in os.listdir order, so its test set was the tail class(es) only and depended on the
filesystem; `per_class=False` does that global split over classes in sorted order.
"""

from pathlib import Path

from .base import Source, make_row, register

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@register
class LegacyDefactify(Source):
    name = "legacy_defactify"
    license = "research use (Defactify 4 shared task)"
    url = "https://huggingface.co/datasets/NasrinImp/Defactify4_Train"
    description = "Defactify 4 (previous project): COCO real + SD2.1, SDXL, SD3, DALL-E 3, Midjourney"

    def __init__(self, train_fraction: float = 0.9, per_class: bool = True):
        self.train_fraction = train_fraction
        self.per_class = per_class

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        root = Path(raw_root) / self.name / "Train"
        if root.is_dir() and any(root.iterdir()):
            return
        print(
            f"[{self.name}] manual download: get the Defactify 4 image training set and place it so that\n"
            f"  {root}/coco_image/*.jpg and {root}/<generator>/*.jpg exist."
        )

    def rows(self, raw_root: Path) -> list[dict]:
        root = Path(raw_root) / self.name / "Train"
        if not root.is_dir():
            return []
        groups = []
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            files = sorted((f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS), key=lambda f: f.name)
            groups.append([(f, class_dir.name) for f in files])
        if not self.per_class:
            groups = [[item for group in groups for item in group]]

        rows = []
        for group in groups:
            n_train = int(self.train_fraction * len(group))
            for i, (f, class_name) in enumerate(group):
                label = int(class_name != "coco_image")
                split = "train" if i < n_train else "test"
                rows.append(
                    make_row(
                        f.relative_to(raw_root), label, "photo", self.name, split, generator=class_name.lower()
                    )
                )
        return rows
