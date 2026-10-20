"""Interface every dataset adapter implements.

A source knows how to fetch its files into `raw_root/<name>/` and how to list them
as manifest rows. Adapters stay small: download, walk files, emit rows.

    @register
    class MySource(Source):
        name = "my_source"
        license = "CC BY 4.0"
        url = "https://..."

        def download(self, raw_root, max_items=None): ...
        def rows(self, raw_root): return [make_row(...), ...]
"""

from pathlib import Path

from ..manifest import COLUMNS, CONTENT_TYPES, SPLITS

REGISTRY: dict[str, type["Source"]] = {}


class Source:
    name: str = ""
    license: str = ""
    url: str = ""
    description: str = ""

    def download(self, raw_root: Path, max_items: int | None = None) -> None:
        """Fetch files into raw_root / self.name. Must be safe to call twice."""
        raise NotImplementedError

    def rows(self, raw_root: Path) -> list[dict]:
        """List manifest rows. Paths are absolute or relative to raw_root."""
        raise NotImplementedError

    @property
    def root(self) -> str:
        return self.name


def register(cls: type[Source]) -> type[Source]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} needs a `name`")
    REGISTRY[cls.name] = cls
    return cls


def get_source(name: str) -> Source:
    if name not in REGISTRY:
        raise KeyError(f"Unknown source '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]()


def make_row(
    path: str | Path,
    label: int,
    content_type: str,
    source: str,
    split: str,
    mask_path: str | Path = "",
    generator: str = "",
) -> dict:
    if content_type not in CONTENT_TYPES:
        raise ValueError(f"content_type {content_type!r} not in {CONTENT_TYPES}")
    if split not in SPLITS:
        raise ValueError(f"split {split!r} not in {SPLITS}")
    row = {
        "path": str(path),
        "label": int(label),
        "mask_path": str(mask_path) if mask_path else "",
        "generator": generator if label else "",
        "content_type": content_type,
        "source": source,
        "split": split,
    }
    return {c: row[c] for c in COLUMNS}


def take(items: list, max_items: int | None, seed: int = 0) -> list:
    """Deterministic random subset when `max_items` is set."""
    if max_items is None or len(items) <= max_items:
        return list(items)
    import random

    rng = random.Random(seed)
    return rng.sample(list(items), max_items)
