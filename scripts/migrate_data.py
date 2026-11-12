"""python scripts/migrate_data.py --config configs/train.yaml

One-off, for data downloaded by the old code: brings them to the layout the downloader writes now, without
downloading anything.
    data/openfake_train/0000_00012.png  ->  data/fake/<generator>/train_0000_00012.png
    data/openfake_train/0000_00013.jpg  ->  data/real/<source>/train_0000_00013.jpg
The new paths are written to data/openfake_<split>.csv and to the downloader's records in
data/openfake_<split>/done/, so download_data.py finds everything done. A rename on the same disk: nothing is
copied. Safe to stop and to run again. Delete this script once it has run.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(line_buffering=True)

from src.data import load_rows, save_rows  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train.yaml")
data = Path(yaml.safe_load(open(parser.parse_args().config))["data"]["dir"])
moved, made = 0, set()


def place(row: dict, split: str) -> None:
    """Moves the image of a row to its folder if it is not there yet, and writes the new path into the row."""
    global moved
    old = Path(row["path"])
    folder = data / "fake" / row["generator"] if row["label"] else data / "real" / row["source"].split("/")[-1]
    if old.parts[-3:-1] != folder.parts[-2:]:
        new = folder / f"{split}_{old.name}"  # train and test files may share a name
        if folder not in made:
            folder.mkdir(parents=True, exist_ok=True)
            made.add(folder)
        if old.exists():  # missing: moved already, by an earlier run or through the other list
            old.rename(new)
            moved += 1
        row["path"] = str(new)


for split in ("train", "test"):
    old_dir, cache = data / f"openfake_{split}", data / f"openfake_{split}.csv"
    rows = load_rows(cache) if cache.exists() else []
    for r in rows:
        place(r, split)
    if rows:
        save_rows(rows, cache)
    records = sorted((old_dir / "done").glob("*.json"))  # the same images, as the downloader recorded them
    for record in records:
        part = json.loads(record.read_text())
        for r in part["rows"]:
            place(r, split)
        record.with_suffix(".tmp").write_text(json.dumps(part))
        record.with_suffix(".tmp").replace(record)
    rows = rows or [r for record in records for r in json.loads(record.read_text())["rows"]]
    missing = sum(not Path(r["path"]).exists() for r in rows)
    left = sum(p.is_file() for p in old_dir.iterdir()) if old_dir.is_dir() else 0
    print(f"OpenFake {split}: {len(rows)} images listed, {missing} of them not found, {left} files left in {old_dir}")

print(f"{moved} images moved to {data / 'real'} and {data / 'fake'}")
