"""TextFake (arXiv 2606.01050): AI-generated text-rich images (screenshots, documents, news pages).

20,000 images: 10,000 real (8,002 screenshots + 1,998 photos of paper) and 10,000 fakes made by
Qwen-Image-2, Nano-Banana-2, Seedream-5-Lite and GPT-Image-2; 28 languages, 4 topics (politics &
military, society, tech & finance, culture & entertainment), screen / paper scenes 80:20, all
re-encoded as JPEG. content_type="text" for every row, split="test".

HF repo `Yuning0123/TextFake` (named in the paper's appendix), a single 22 GB TextFake.zip, gated
with manual approval: e-mail ynzhang2022@gmail.com (template on the HF page; academic,
non-commercial, no redistribution), then log in (`hf auth login` or HF_TOKEN). Until access is
granted download() prints these instructions.

With `max_items`, a random subset of zip members is read through HTTP range requests; without a
cap the zip is downloaded, extracted and deleted. The folder layout inside the zip could not be
inspected (gated), so rows() infers labels from path words: real / 0_real / human -> 0,
fake / 1_fake / ai / generated / synthetic -> 1, and the generator from words such as qwen, nano,
seedream, gpt. Check the counts after the first download and adjust REAL_WORDS / FAKE_WORDS.
"""

from pathlib import Path

from huggingface_hub.errors import GatedRepoError

from ._fake_files import list_images, manual_notice, rel
from ._fake_hub import extract_hub_zip
from .base import Source, make_row, register

REPO_ID = "Yuning0123/TextFake"
REAL_WORDS = {"real", "0_real", "human", "authentic"}
FAKE_WORDS = {"fake", "1_fake", "ai", "generated", "synthetic", "aigc"}
GENERATORS = {"qwen": "qwen-image-2", "nano": "nano-banana-2", "banana": "nano-banana-2",
              "seedream": "seedream-5-lite", "gpt": "gpt-image-2"}
INSTRUCTIONS = """\
  1. Follow https://huggingface.co/datasets/Yuning0123/TextFake : e-mail ynzhang2022@gmail.com with
     the request template, then request access on the HF page.
  2. After approval, log in with `hf auth login` (or set HF_TOKEN) and rerun download()."""


def label_and_generator(parts: tuple[str, ...]) -> tuple[int | None, str]:
    words = [p.lower() for p in parts]
    generator = next((g for w in words for key, g in GENERATORS.items() if key in w), "unknown")
    if any(w in REAL_WORDS for w in words):
        return 0, ""
    if any(w in FAKE_WORDS for w in words) or generator != "unknown":
        return 1, generator
    return None, ""


@register
class TextFake(Source):
    name = "textfake"
    license = "research only, gated"
    url = "https://huggingface.co/datasets/Yuning0123/TextFake"
    description = "TextFake: text-rich real vs AI images (Qwen-Image-2, NB2, Seedream, GPT-Image-2)"

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        root = Path(raw_root) / self.name
        try:
            count = extract_hub_zip(REPO_ID, "TextFake.zip", root, max_items)
            print(f"   TextFake.zip: {count} images")
        except GatedRepoError:
            manual_notice(self.name, root, INSTRUCTIONS)

    def rows(self, raw_root: Path) -> list[dict]:
        raw_root = Path(raw_root)
        root = raw_root / self.name
        rows = []
        for path in list_images(root):
            parts = path.relative_to(root).parts[:-1]
            if parts and parts[0] in ("_zips", "_done"):
                continue
            label, generator = label_and_generator(parts)
            if label is not None:
                rows.append(make_row(rel(path, raw_root), label, "text", self.name, "test", generator=generator))
        return rows
