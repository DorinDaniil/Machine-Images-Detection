"""Download the datasets listed in a registry YAML and merge them into one manifest CSV.

    python -m src.data.build_manifest --list
    python -m src.data.build_manifest --datasets wikiart,coco_val --out data/manifests/train_real.csv
    python -m src.data.build_manifest --datasets all --role test --out data/manifests/test.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from .manifest import COLUMNS, summarize_manifest, write_manifest
from .registry import build_source, load_registry
from .sources import take


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", default="configs/datasets.yaml")
    parser.add_argument("--datasets", default="all", help="comma-separated names or 'all'")
    parser.add_argument("--role", default=None, help="only entries whose `role` matches (e.g. train_real, test)")
    parser.add_argument("--out", default="data/manifests/all.csv")
    parser.add_argument("--max-items", type=int, default=None, help="override every entry's max_items")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    reg = load_registry(args.registry)
    entries = reg["datasets"]
    if args.role:
        entries = {k: v for k, v in entries.items() if v.get("role") == args.role}
    if args.datasets != "all":
        wanted = [n.strip() for n in args.datasets.split(",") if n.strip()]
        entries = {k: entries[k] for k in wanted}

    if args.list:
        for name, e in entries.items():
            print(f"{name:26s} {e['kind']:10s} {e.get('role', ''):11s} {e.get('repo', e.get('adapter', '')):40s} {e.get('notes', '')}")
        return

    raw_root = Path(reg["raw_root"])
    raw_root.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, entry in entries.items():
        entry = {k: v for k, v in entry.items() if k != "role"}
        source, max_items = build_source(name, entry)
        if args.max_items is not None:
            max_items = args.max_items
        print(f"== {name}")
        if not args.no_download:
            source.download(raw_root, max_items)
        rows = take(source.rows(raw_root), max_items)
        print(f"   {len(rows)} rows")
        frames.append(pd.DataFrame(rows, columns=COLUMNS))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    for col in ("path", "mask_path"):
        df[col] = [p if (not p or Path(p).is_absolute()) else str(raw_root / p) for p in df[col]]
    write_manifest(df, args.out)
    print(f"\nwrote {len(df)} rows -> {args.out}")
    if len(df):
        print(summarize_manifest(df).to_string(index=False))


if __name__ == "__main__":
    main()
