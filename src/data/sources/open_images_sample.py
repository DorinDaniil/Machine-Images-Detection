"""Open Images V7 photos, sampled from the validation split of the HF mirror `finedet/openimages`.

The mirror stores the CVDF-hosted images (the copies in the official open-images-dataset S3 bucket,
mostly rescaled to 1024 px on the long side, e.g. 1024x768 / 1024x683) inside parquet shards of
about 2,500 images / 780 MB. The validation split has 15 shards (37,306 images). Shards are
fetched in a fixed shuffled order, exported as the original JPEG bytes and deleted, until
`max_items` images passed the `min_side` filter (default 512 px), so the cap is honoured to
within one shard.

Layout: raw_root/open_images_sample/images/<shard>_<row>.jpg

License: the mirror is CC BY 4.0 (annotations); every image is listed as CC BY 2.0 by Google,
which does not warrant the license status of each image. Flickr photos uploaded up to ~2016:
consumer cameras and early smartphones, some edited/HDR, essentially no generative AI.
"""

from pathlib import Path

from ._real_utils import fetch_hub_parquet, hub_files, image_rows
from .base import Source, register

REPO_ID = "finedet/openimages"


@register
class OpenImagesSample(Source):
    name = "open_images_sample"
    license = "CC BY 2.0 (images)"
    url = "https://huggingface.co/datasets/finedet/openimages"
    description = "Open Images V7 validation photos (~1024 px), parquet shards from finedet/openimages"

    def __init__(self, split: str = "train", min_side: int = 512):
        self.split = split
        self.min_side = min_side

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        out_dir = Path(raw_root) / self.name / "images"
        files = hub_files(REPO_ID, ("data/validation-",))
        fetch_hub_parquet(REPO_ID, files, out_dir, max_items, min_side=self.min_side)

    def rows(self, raw_root: Path) -> list[dict]:
        return image_rows(raw_root, Path(raw_root) / self.name / "images", "photo", self.name, self.split)
