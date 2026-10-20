"""INP-X / Inpainting Exchange (arXiv 2602.00192, github.com/emirhanbilgic/INP-X): test split.

Kaggle dataset `emirhanbilgic/inpainting-exchange` (MIT, ~10.7 GB, 120k files; extends Semi-Truths).
Real images from CelebA-HQ, CityScapes, OpenImages and SUN-RGBD are inpainted by Kandinsky 2.2,
OpenJourney and Stable Diffusion v1.4; every edit exists twice:
- standard_inpainting/: the usual pipeline output (VAE round trip changes the whole image);
- inpainting_exchange/: INP-X, original pixels restored outside the mask, so only the generated
  region differs (generator gets an "-inpx" suffix, e.g. "kandinsky-2.2-inpx").

Not downloadable automatically (Kaggle needs an account / API token). Expected layout:

    raw_root/inpx/inpainting_exchange/test-data/data/originals/<DS>/*.jpg
    raw_root/inpx/inpainting_exchange/test-data/data/standard_inpainting/<DS>/<id>_<DS>_<Model>.jpg
    raw_root/inpx/inpainting_exchange/test-data/data/inpainting_exchange/<DS>/<id>_<DS>_<Model>_simple.jpg
    raw_root/inpx/inpainting_exchange/test-data/masks/<DS>_masks/<id>.jpg

Masks are JPEGs (white = inpainted, some sit in another dataset's *_masks folder, so they are
matched by <id> across all of masks/). Once the files are there, download() writes binary PNG masks
next to each fake as <stem>.mask.png and rows() lists them. Only test-data/ is used (split="test").
"""

from pathlib import Path

from ._fake_files import list_images, manual_notice, mask_path_for, rel, save_binary_mask
from .base import Source, make_row, register

MODELS = {"Kandinsky_2_2": "kandinsky-2.2", "OpenJourney": "openjourney", "StableDiffusion_v4": "sd-1.4"}
FAKE_DIRS = {"standard_inpainting": "", "inpainting_exchange": "-inpx"}
INSTRUCTIONS = """\
  1. Download https://www.kaggle.com/datasets/emirhanbilgic/inpainting-exchange (Kaggle login), e.g.
       kaggle datasets download -d emirhanbilgic/inpainting-exchange -p <raw_root>/inpx --unzip
  2. Check that <raw_root>/inpx/inpainting_exchange/test-data/data/originals/ exists, rerun download()."""


def split_stem(stem: str, dataset: str) -> tuple[str, str]:
    """'10052_hair_CelebAHQ_OpenJourney_simple' -> ('10052_hair', 'OpenJourney')."""
    mask_id, _, model = stem.partition(f"_{dataset}_")
    return mask_id, model.removesuffix("_simple")


@register
class InpX(Source):
    name = "inpx"
    license = "MIT"
    url = "https://www.kaggle.com/datasets/emirhanbilgic/inpainting-exchange"
    description = "INP-X: SD1.4/Kandinsky/OpenJourney inpaints, standard + exchanged, with masks"

    def _test_dir(self, raw_root: Path) -> Path:
        return Path(raw_root) / self.name / "inpainting_exchange" / "test-data"

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        test_dir = self._test_dir(raw_root)
        if not (test_dir / "data" / "originals").is_dir():
            manual_notice(self.name, Path(raw_root) / self.name, INSTRUCTIONS)
            return
        masks = {p.stem: p for p in list_images(test_dir / "masks")}
        for fake_dir in FAKE_DIRS:
            for image in list_images(test_dir / "data" / fake_dir):
                mask = masks.get(split_stem(image.stem, image.parent.name)[0])
                if mask is not None and not mask_path_for(image).exists():
                    save_binary_mask(mask, mask_path_for(image))

    def rows(self, raw_root: Path) -> list[dict]:
        raw_root = Path(raw_root)
        data_dir = self._test_dir(raw_root) / "data"
        originals = list_images(data_dir / "originals")
        rows = [make_row(rel(p, raw_root), 0, "photo", self.name, "test") for p in originals]
        for fake_dir, suffix in FAKE_DIRS.items():
            for image in list_images(data_dir / fake_dir):
                model = split_stem(image.stem, image.parent.name)[1]
                mask = mask_path_for(image)
                rows.append(make_row(
                    rel(image, raw_root), 1, "photo", self.name, "test",
                    mask_path=rel(mask, raw_root) if mask.exists() else "",
                    generator=MODELS.get(model, "unknown") + suffix,
                ))
        return rows
