"""`hf_files` with a cap: picks max_items files from the repo listing before downloading anything.

Like generic.HFFilesSource (labels from folder names, generator from a path part), but download()
honours max_items: the cap is split evenly over the groups (the path part at `generator_from`, else
the top folder) and spread over the sorted names, then only those files are fetched one by one.
`generator_map` renames the lowercased folder names to proper generator slugs.

    kind: custom
    adapter: hf_files_capped
    args:
      repo: Qwen/Qwen-Image-Bench
      allow_patterns: ["images/*"]
      label: 1                                 # or folder_labels: {original: 0, Flux: 1}
      generator_from: 1
      generator_map: {flux.2_max: flux.2-max}
      content_type: other
"""

import fnmatch
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from ._fake_hub import spread
from .base import register
from .generic import IMAGE_EXTS, HFFilesSource


@register
class HFFilesCapped(HFFilesSource):
    name = "hf_files_capped"

    def __init__(self, generator_map: dict | None = None, workers: int = 8, **kwargs):
        super().__init__(name=self.name, **kwargs)
        self.generator_map = {str(k).lower(): v for k, v in (generator_map or {}).items()}
        self.workers = workers

    def _group(self, path: str) -> str:
        parts = path.split("/")
        if self.generator_from is not None and len(parts) > self.generator_from + 1:
            return parts[self.generator_from]
        return parts[0] if len(parts) > 1 else ""

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        out = Path(raw_root) / self.name
        done = out / "_done"
        if done.exists():
            return
        files = HfApi().list_repo_files(self.repo, repo_type=self.repo_type)
        files = [f for f in files if Path(f).suffix.lower() in IMAGE_EXTS]
        if self.allow_patterns:
            files = [f for f in files if any(fnmatch.fnmatch(f, p) for p in self.allow_patterns)]
        groups = defaultdict(list)
        for f in sorted(files):
            groups[self._group(f)].append(f)
        quota = None if max_items is None else math.ceil(max_items / max(len(groups), 1))
        chosen = [f for g in sorted(groups) for f in (groups[g] if quota is None else spread(groups[g], quota))]
        print(f"   {self.repo}: {len(chosen)} of {len(files)} files in {len(groups)} groups")

        def fetch(filename: str) -> None:
            if not (out / filename).exists():
                hf_hub_download(self.repo, filename, repo_type=self.repo_type, local_dir=out)

        with ThreadPoolExecutor(self.workers) as pool:
            list(pool.map(fetch, chosen))
        out.mkdir(parents=True, exist_ok=True)
        done.write_text(str(len(chosen)))

    def rows(self, raw_root: Path) -> list[dict]:
        rows = super().rows(raw_root)
        for row in rows:
            row["generator"] = self.generator_map.get(row["generator"], row["generator"])
        return rows
