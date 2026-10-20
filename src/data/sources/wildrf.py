"""WildRF (LaDeDa paper, arXiv 2406.09398, https://vision.huji.ac.il/ladeda/).

Real and AI-generated images collected by hand (keywords / hashtags) from Reddit, X (Twitter) and
Facebook, with platform compression and resizing. Generators are unknown. No dataset license is
stated; the code repo ships a non-commercial "Software Research License".

Not downloadable automatically: a single zip on Google Drive, linked from
https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld . After unzipping:

    raw_root/wildrf/WildRF/train/{0_real,1_fake}            (Reddit, for training)
    raw_root/wildrf/WildRF/val/{0_real,1_fake}
    raw_root/wildrf/WildRF/test/{reddit,twitter,facebook}/{0_real,1_fake}

(the top WildRF/ folder is optional). Only the held-out test/ part is listed, as split="test",
generator "unknown", content_type "photo"; the train/val folders are left alone.
"""

from pathlib import Path

from ._fake_files import manual_notice, real_fake_rows
from .base import Source, register

DRIVE_URL = "https://drive.google.com/file/d/1A0xoL44Yg68ixd-FuIJn2VC4vdZ6M2gn/view?usp=sharing"
INSTRUCTIONS = f"""\
  1. Download the WildRF zip from Google Drive: {DRIVE_URL}
  2. Unzip it into <raw_root>/wildrf/ so that wildrf/WildRF/test/<platform>/0_real|1_fake exist."""


@register
class WildRF(Source):
    name = "wildrf"
    license = "unstated, research"
    url = "https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld"
    description = "WildRF test: Reddit / X / Facebook real vs fake, manual Google Drive download"

    def _test_dir(self, raw_root: Path) -> Path:
        root = Path(raw_root) / self.name
        nested = root / "WildRF" / "test"
        return nested if nested.is_dir() else root / "test"

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        if not self._test_dir(raw_root).is_dir():
            manual_notice(self.name, Path(raw_root) / self.name, INSTRUCTIONS)

    def rows(self, raw_root: Path) -> list[dict]:
        return real_fake_rows(Path(raw_root), self._test_dir(raw_root), self.name, "unknown")
