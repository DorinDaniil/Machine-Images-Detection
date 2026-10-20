"""Download sources and merge them into one manifest CSV.

    python -m data_tools.build_manifest --list
    python -m data_tools.build_manifest --sources coco_val,dailybench --max-per-source 2000 \
        --raw-root data/raw --out data/manifests/test.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from data_tools.sources import REGISTRY, get_source
from trace_detector.data.manifest import COLUMNS, summarize_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", default="all", help="comma-separated source names or 'all'")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--out", default="data/manifests/all.csv")
    parser.add_argument("--max-per-source", type=int, default=None, help="cap images per source (download + rows)")
    parser.add_argument("--no-download", action="store_true", help="only list files already on disk")
    parser.add_argument("--list", action="store_true", help="print registered sources and exit")
    args = parser.parse_args()

    if args.list:
        for name in sorted(REGISTRY):
            s = REGISTRY[name]
            print(f"{name:28s} {s.license:22s} {s.description}")
        return

    names = sorted(REGISTRY) if args.sources == "all" else [n.strip() for n in args.sources.split(",") if n.strip()]
    raw_root = Path(args.raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)

    frames = []
    for name in names:
        source = get_source(name)
        print(f"== {name}")
        if not args.no_download:
            source.download(raw_root, args.max_per_source)
        rows = source.rows(raw_root)
        if args.max_per_source and len(rows) > args.max_per_source:
            from data_tools.sources import take

            rows = take(rows, args.max_per_source)
        print(f"   {len(rows)} rows")
        frames.append(pd.DataFrame(rows, columns=COLUMNS))

    df = pd.concat(frames, ignore_index=True)
    df["path"] = [p if Path(p).is_absolute() else str(raw_root / p) for p in df["path"]]
    df["mask_path"] = [p if (not p or Path(p).is_absolute()) else str(raw_root / p) for p in df["mask_path"]]
    write_manifest(df, args.out)
    print(f"\nwrote {len(df)} rows -> {args.out}")
    print(summarize_manifest(df).to_string(index=False))


if __name__ == "__main__":
    main()
