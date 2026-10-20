"""ITW-SM: In The Wild - Social Media (arXiv 2507.10236, https://mever-team.github.io/itw-sm).

10,000 images (5,000 real from verified accounts, 5,000 AI-generated posted by users) collected
from Facebook, Instagram, LinkedIn and X at native resolution and platform compression; ~3.6 GB.
Generators are unknown. HF repo `dkarageo/itw-sm`:

    0_real/<Platform>_real_<n>.jpg|png   1_fake/<platform>_<n>.jpg   metadata.csv

The repo is gated ("manual" approval): accept the ITW-SM Terms of Access on the HF page (research
only, no redistribution, no re-identification) and log in (`hf auth login` or HF_TOKEN)
before download() can fetch anything; until then it prints these instructions.

`max_items` is honoured exactly (half real, half fake, fetched file by file); without a cap the
whole repo is fetched with snapshot_download. Rows: split="test", generator "unknown", "photo".
"""

import math
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError

from ._fake_files import manual_notice, real_fake_rows
from ._fake_hub import hub_files
from .base import Source, register, take

REPO_ID = "dkarageo/itw-sm"
INSTRUCTIONS = """\
  1. Open https://huggingface.co/datasets/dkarageo/itw-sm and request access (accept the terms).
  2. After approval, log in with `hf auth login` (or set HF_TOKEN) and rerun download()."""


@register
class ItwSm(Source):
    name = "itw_sm"
    license = "research only, gated"
    url = "https://huggingface.co/datasets/dkarageo/itw-sm"
    description = "ITW-SM: real vs AI images from Facebook/Instagram/LinkedIn/X (gated HF)"

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        root = Path(raw_root) / self.name
        try:
            if max_items is None:
                snapshot_download(
                    REPO_ID, repo_type="dataset", local_dir=root,
                    allow_patterns=["metadata.csv", "0_real/*", "1_fake/*"],
                )
                return
            for folder in ("0_real/", "1_fake/"):
                for filename in take(hub_files(REPO_ID, folder, suffix=""), math.ceil(max_items / 2)):
                    if not (root / filename).exists():
                        hf_hub_download(REPO_ID, filename, repo_type="dataset", local_dir=root)
        except GatedRepoError:
            manual_notice(self.name, root, INSTRUCTIONS)

    def rows(self, raw_root: Path) -> list[dict]:
        return real_fake_rows(Path(raw_root), Path(raw_root) / self.name, self.name, "unknown")
