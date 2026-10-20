"""Images inside zip archives of a Hugging Face dataset repo, read member by member.

Only the zip central directories and the chosen members are fetched (HTTP range requests through
HfFileSystem), so a cap of a few thousand images never downloads the multi-GB archives. Members are
picked with a regex; its named groups give the generator (`generator`), the label (`label`, mapped
through `labels`) and optionally the content type (`content`, mapped through `contents`).
max_items is split evenly over the (generator, label) groups and spread over the sorted names.

    kind: custom
    adapter: hf_zip
    args:
      repo: Yanran21/ARForensics
      files: [ARForensics_updated_test_part001.zip]
      pattern: '^ARForensics/test/(?P<generator>[^/]+)/(?P<label>0_real|1_fake)/[^/]+$'
      labels: {0_real: 0, 1_fake: 1}          # or label: 1 for all members
      generators: {Janus_Pro: janus-pro}      # optional renames, default slug(name)
      content_type: photo                     # or contents: {A: art, P: photo} for the `content` group
"""

import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfFileSystem

from ._fake_files import IMAGE_EXTS, append_index, index_rows, rel, slug
from ._fake_hub import spread
from .base import Source, register


@register
class HFZip(Source):
    name = "hf_zip"
    license = "see dataset card"

    def __init__(self, repo: str = "", files: list[str] | None = None, pattern: str = ".*", label: int | None = None,
                 labels: dict | None = None, generators: dict | None = None, contents: dict | None = None,
                 content_type: str = "photo", manifest_split: str = "test"):
        self.repo, self.files, self.regex = repo, list(files or []), re.compile(pattern)
        self.label, self.labels = label, {str(k): int(v) for k, v in (labels or {}).items()}
        self.generators, self.contents = dict(generators or {}), dict(contents or {})
        self.content_type, self.manifest_split = content_type, manifest_split
        self.url, self.description = f"https://huggingface.co/datasets/{repo}", f"{repo} zip members"

    def _record(self, match: re.Match) -> tuple[str, int, str] | None:
        groups = match.groupdict()
        label = self.labels.get(groups["label"]) if "label" in groups else self.label
        if label is None:
            return None
        raw = groups.get("generator") or ""
        generator = self.generators.get(raw, slug(raw)) if raw else ""
        content = self.contents.get(groups.get("content") or "", self.content_type)
        return generator, int(label), content

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        root = Path(raw_root) / self.name
        if (root / "index.csv").exists():
            return
        fs, archives, groups = HfFileSystem(), {}, defaultdict(list)
        for filename in self.files:
            handle = fs.open(f"datasets/{self.repo}/{filename}", "rb", block_size=1 << 20)
            archives[filename] = (handle, zipfile.ZipFile(handle))
            for member in archives[filename][1].namelist():
                match = self.regex.match(member)
                if match is None or Path(member).suffix.lower() not in IMAGE_EXTS or ".." in member:
                    continue
                info = self._record(match)
                if info is not None:
                    groups[info[:2]].append((member, filename, info))
        quota = None if max_items is None else math.ceil(max_items / max(len(groups), 1))
        records = []
        for key in sorted(groups):
            members = sorted(groups[key])
            for member, filename, (generator, label, content) in members if quota is None else spread(members, quota):
                target = root / Path(filename).stem / member
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archives[filename][1].read(member))
                records.append({"path": rel(target, Path(raw_root)), "label": label, "generator": generator,
                                "content_type": content, "mask_path": ""})
            print(f"   {self.repo}: {key[0] or '-'} label={key[1]}: {min(len(members), quota or len(members))} images")
        for handle, zf in archives.values():
            zf.close()
            handle.close()
        append_index(root, records)

    def rows(self, raw_root: Path) -> list[dict]:
        return index_rows(Path(raw_root), self.name, self.manifest_split)
