"""Chameleon test set from "A Sanity Check for AI-generated Image Detection" (AIDE, arXiv 2406.19435).

~26k images: AI-generated images collected from AI-art communities (picked to fool humans, mostly
photorealistic: people, animals, objects, scenes; generators not recorded) and real photos from
Unsplash-like photographer sites. License: academic research only, commercial use prohibited.

Not downloadable automatically: the authors send the data on request by e-mail
(tattoo.ysl@gmail.com, see https://github.com/shilinyan99/AIDE#-chameleon). There is no official
HF / Google Drive mirror. Unpack what you receive so that it looks like

    raw_root/chameleon/test/0_real/*.jpg|png
    raw_root/chameleon/test/1_fake/*.jpg|png

(any depth works: every image below a `0_real` or `1_fake` folder is listed). Rows: split="test",
generator "unknown", content_type "photo".
"""

from pathlib import Path

from ._fake_files import list_images, manual_notice, real_fake_rows
from .base import Source, register

INSTRUCTIONS = """\
  1. E-mail tattoo.ysl@gmail.com asking for the Chameleon test set (academic, non-commercial use).
  2. Unpack it to <raw_root>/chameleon/ so the 0_real/ and 1_fake/ folders sit under test/:
       chameleon/test/0_real/...   chameleon/test/1_fake/..."""


@register
class Chameleon(Source):
    name = "chameleon"
    license = "academic only, e-mail"
    url = "https://github.com/shilinyan99/AIDE"
    description = "Chameleon (AIDE): human-fooling AI images vs real photos, manual download"

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        root = Path(raw_root) / self.name
        if not list_images(root):
            manual_notice(self.name, root, INSTRUCTIONS)

    def rows(self, raw_root: Path) -> list[dict]:
        return real_fake_rows(Path(raw_root), Path(raw_root) / self.name, self.name, "unknown")
