"""ArtiFact as used by the previous project (local copy, no download).

Layout: raw_root/legacy_artifact/<generator_folder>/metadata.csv with columns image_path
(relative to the folder) and target (0 = real, anything else = fake).

Split per folder: the first `split_per_folder = (n_real, n_fake)` real and fake rows go to
"test", the rest to "train". The old generate_metadata.py took them in metadata.csv order
(its random.shuffle ran after the selection, so it did not change membership); keep
`shuffle_seed=None` to reproduce that split, or set a seed to shuffle each folder first.
"""

from pathlib import Path

import pandas as pd

from .base import Source, make_row, register


@register
class LegacyArtifact(Source):
    name = "legacy_artifact"
    license = "mixed, per source dataset"
    url = "https://www.kaggle.com/datasets/awsaf49/artifact-dataset"
    description = "ArtiFact (previous project): 8 real + 25 generator folders, metadata.csv per folder"

    def __init__(self, split_per_folder: tuple[int, int] = (300, 200), shuffle_seed: int | None = None):
        self.split_per_folder = split_per_folder
        self.shuffle_seed = shuffle_seed

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        root = Path(raw_root) / self.name
        if any(root.glob("*/metadata.csv")):
            return
        print(
            f"[{self.name}] manual download: `kaggle datasets download -d awsaf49/artifact-dataset`,\n"
            f"  unzip so that {root}/<folder>/metadata.csv exists for every generator folder."
        )

    def rows(self, raw_root: Path) -> list[dict]:
        root = Path(raw_root) / self.name
        if not root.is_dir():
            return []
        rows = []
        for folder in sorted(p for p in root.iterdir() if (p / "metadata.csv").is_file()):
            meta = pd.read_csv(folder / "metadata.csv", usecols=["image_path", "target"])
            if self.shuffle_seed is not None:
                meta = meta.sample(frac=1.0, random_state=self.shuffle_seed)
            test_left = list(self.split_per_folder)  # [real, fake]
            for image_path, target in zip(meta["image_path"], meta["target"]):
                label = int(target != 0)
                split = "test" if test_left[label] > 0 else "train"
                test_left[label] -= 1
                rows.append(
                    make_row(
                        Path(self.name) / folder.name / image_path,
                        label,
                        "photo",
                        self.name,
                        split,
                        generator=folder.name.lower(),
                    )
                )
        return rows
