"""Icons: Microsoft Fluent UI emoji plus iOS app icons, both from small HF parquet datasets.

- `Norod78/microsoft-fluentui-emoji-768` (MIT, from github.com/microsoft/fluentui-emoji):
  7,564 emoji in the Color, Flat and High Contrast styles, SVGs rasterised to 768x768 PNG;
  2 shards, 700 MB.
- `ppierzc/ios-app-icons`: 1,919 app icons collected from the iOS Icon Gallery, 512x512;
  2 shards, 340 MB (the captions in that set are BLIP-generated, the images are not).

Shards are fetched in a fixed shuffled order (per repo) and deleted after export; with a cap,
each repo contributes up to half of `max_items`, honoured to within one shard. No Icons-50
(Hendrycks & Dietterich) mirror was found on the Hub; drop any extra icon folder into
raw_root/icons/<name>/ by hand and rows() picks it up.

Layout: raw_root/icons/<fluentui_emoji|ios_app_icons>/<shard>_<row>.png

License: Fluent emoji MIT. App icons are copyrighted by their app owners; the card's
"openrail" tag does not cover them, so research use only. App icons were collected in
Dec 2023 and a few recent icons might have been made with generative tools.
"""

from pathlib import Path

from ._real_utils import fetch_hub_parquet, hub_files, image_rows
from .base import Source, register

REPOS = {
    "fluentui_emoji": "Norod78/microsoft-fluentui-emoji-768",
    "ios_app_icons": "ppierzc/ios-app-icons",
}


@register
class Icons(Source):
    name = "icons"
    license = "MIT + research only"
    url = "https://huggingface.co/datasets/Norod78/microsoft-fluentui-emoji-768"
    description = "Fluent UI emoji (768 px) + iOS app icons (512 px) from HF parquet"

    def __init__(self, split: str = "train"):
        self.split = split

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        per_repo = None if max_items is None else -(-max_items // len(REPOS))
        for folder, repo_id in REPOS.items():
            out_dir = Path(raw_root) / self.name / folder
            files = hub_files(repo_id, ("data/",))
            fetch_hub_parquet(repo_id, files, out_dir, per_repo)

    def rows(self, raw_root: Path) -> list[dict]:
        return image_rows(raw_root, Path(raw_root) / self.name, "graphics", self.name, self.split)
