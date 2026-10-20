"""Build Source objects from configs/datasets.yaml.

    raw_root: data/raw
    datasets:
      wikiart:                      # name used in the manifest `source` column
        kind: hf_parquet            # hf_parquet | hf_files | custom
        repo: huggan/wikiart
        content_type: art
        max_items: 5000             # cap for download and rows
      dailybench:
        kind: custom
        adapter: dailybench         # class registered in src/data/sources
        args: {}                    # constructor arguments of that class
        max_items: 2000
"""

from pathlib import Path

import yaml

from .sources import REGISTRY
from .sources.generic import HFFilesSource, HFParquetSource

GENERIC = {"hf_parquet": HFParquetSource, "hf_files": HFFilesSource}


def load_registry(path: str | Path) -> dict:
    """Reads the YAML and merges every `datasets` block from the sibling folder <stem>.d/*.yaml."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("raw_root", "data/raw")
    cfg.setdefault("datasets", {})
    for extra in sorted((path.parent / f"{path.stem}.d").glob("*.yaml")):
        with open(extra, encoding="utf-8") as f:
            for name, entry in (yaml.safe_load(f) or {}).get("datasets", {}).items():
                if name in cfg["datasets"]:
                    raise ValueError(f"dataset '{name}' is defined twice ({extra})")
                cfg["datasets"][name] = entry
    return cfg


def build_source(name: str, entry: dict):
    """Returns (source, max_items) for one registry entry."""
    entry = dict(entry)
    kind = entry.pop("kind")
    max_items = entry.pop("max_items", None)
    entry.pop("notes", None)

    if kind == "custom":
        adapter = entry.pop("adapter", name)
        if adapter not in REGISTRY:
            raise KeyError(f"{name}: no adapter '{adapter}' in src/data/sources. Known: {sorted(REGISTRY)}")
        source = REGISTRY[adapter](**entry.pop("args", {}))
        source.name = name
    elif kind in GENERIC:
        source = GENERIC[kind](name=name, **entry)
    else:
        raise ValueError(f"{name}: unknown kind '{kind}', use one of {list(GENERIC) + ['custom']}")
    return source, max_items
