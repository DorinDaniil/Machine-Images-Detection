"""Impostor (arXiv 2606.04545): agent-curated AI-edited images for manipulation localization.

~100k manipulated images made by the CraftAgent pipeline from LVIS source photos with seven
editors: FLUX.1 Fill, FLUX.1 Kontext, PowerPaint, Step1X-Edit, Qwen-Image-Edit, OmniEraser and
Nano Banana Pro; three manipulation types (addition, removal, inpainting); each sample has the
source image, the edited image and a tampering mask; per-generator 7:1 train/test split.

STUB: as of 2026-10-20 no public download was found (no link in the paper, no HF / GitHub release;
`Impostor/Pixel` on HF is an unrelated empty repo). download() only prints this note. If you obtain
the data, arrange its test part as

    raw_root/impostor/real/*.jpg|png                       LVIS source images (label 0)
    raw_root/impostor/fake/<editor>/<stem>.jpg|png         edited images (label 1)
    raw_root/impostor/mask/<editor>/<stem>.png             mask with the same stem as the fake

where <editor> is one of the EDITORS keys (other names become the slug of the folder name).
download() then converts every mask to a binary PNG (255 = edited) next to its fake as
<stem>.mask.png, and rows() lists everything as split="test", content_type="photo".
"""

from pathlib import Path

from ._fake_files import list_images, manual_notice, mask_path_for, rel, save_binary_mask, slug
from .base import Source, make_row, register

EDITORS = {
    "flux_fill": "flux.1-fill", "flux_kontext": "flux.1-kontext", "powerpaint": "powerpaint",
    "step1x_edit": "step1x-edit", "qwen_image_edit": "qwen-image-edit", "omnieraser": "omnieraser",
    "nano_banana_pro": "nano-banana-pro-edit",
}
INSTRUCTIONS = """\
  Impostor has no public release yet (checked 2026-10-20: arXiv 2606.04545 links no data).
  Ask the authors (Southeast University / Xiaohongshu) and arrange the files as described in
  src/data/sources/impostor.py: impostor/real/, impostor/fake/<editor>/, impostor/mask/<editor>/."""


@register
class Impostor(Source):
    name = "impostor"
    license = "unreleased (unknown)"
    url = "https://arxiv.org/abs/2606.04545"
    description = "Impostor: AI-edited images by 7 editors with masks (stub, not released yet)"

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        root = Path(raw_root) / self.name
        if not (root / "fake").is_dir():
            manual_notice(self.name, root, INSTRUCTIONS)
            return
        for mask in list_images(root / "mask"):
            fakes = list((root / "fake" / mask.parent.name).glob(mask.stem + ".*"))
            for fake in fakes:
                if not fake.name.endswith(".mask.png") and not mask_path_for(fake).exists():
                    save_binary_mask(mask, mask_path_for(fake))

    def rows(self, raw_root: Path) -> list[dict]:
        raw_root = Path(raw_root)
        root = raw_root / self.name
        rows = [make_row(rel(p, raw_root), 0, "photo", self.name, "test") for p in list_images(root / "real")]
        for fake in list_images(root / "fake"):
            folder = fake.parent.name
            mask = mask_path_for(fake)
            rows.append(make_row(
                rel(fake, raw_root), 1, "photo", self.name, "test",
                mask_path=rel(mask, raw_root) if mask.exists() else "",
                generator=EDITORS.get(folder.lower(), slug(folder)),
            ))
        return rows
