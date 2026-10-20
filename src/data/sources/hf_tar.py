"""Images from .tar / .tar.gz archives (e.g. WebDataset shards) of a Hugging Face dataset repo.

A tar.gz has no index, so each archive is streamed from its start through HfFileSystem and
abandoned as soon as its share of max_items has been read: a cap of a few hundred images per
archive transfers only the first few hundred MB, never the whole multi-GB file. One archive per
generator; the share is max_items / number of archives. Without a cap every archive is read fully.

    kind: custom
    adapter: hf_tar
    args:
      repo: ruojiruoli/Co-Spy-Bench
      files:                                    # generator slug -> archive path in the repo
        flux.1-dev: mscoco/black-forest-labs@FLUX.1-dev/images_with_metadata.tar.gz
      label: 1
      content_type: photo
"""

import math
import tarfile
from pathlib import Path

from huggingface_hub import HfFileSystem

from ._fake_files import IMAGE_EXTS, append_index, index_rows, rel
from .base import Source, register


@register
class HFTar(Source):
    name = "hf_tar"
    license = "see dataset card"

    def __init__(self, repo: str = "", files: dict | None = None, label: int = 1, content_type: str = "photo",
                 manifest_split: str = "test"):
        self.repo, self.files, self.label = repo, dict(files or {}), int(label)
        self.content_type, self.manifest_split = content_type, manifest_split
        self.url, self.description = f"https://huggingface.co/datasets/{repo}", f"{repo} tar members"

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        root = Path(raw_root) / self.name
        if (root / "index.csv").exists():
            return
        quota = None if max_items is None else math.ceil(max_items / max(len(self.files), 1))
        fs, records = HfFileSystem(), []
        for generator, filename in sorted(self.files.items()):
            count = 0
            with fs.open(f"datasets/{self.repo}/{filename}", "rb", block_size=1 << 22) as f:
                with tarfile.open(fileobj=f, mode="r|*") as tf:
                    for member in tf:
                        if quota is not None and count >= quota:
                            break
                        name = member.name
                        if not member.isfile() or Path(name).suffix.lower() not in IMAGE_EXTS or ".." in name:
                            continue
                        target = root / generator / Path(name).name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(tf.extractfile(member).read())
                        records.append({"path": rel(target, Path(raw_root)), "label": self.label, "generator": generator,
                                        "content_type": self.content_type, "mask_path": ""})
                        count += 1
            print(f"   {self.repo}: {generator}: {count} images")
        append_index(root, records)

    def rows(self, raw_root: Path) -> list[dict]:
        return index_rows(Path(raw_root), self.name, self.manifest_split)
