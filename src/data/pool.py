"""The image folders and the split into train, val and test.

    data/real/<source>/...        real images: laion, docci, ...; any folder put here is picked up
    data/fake/<generator>/...     generated images; the folder name is the generator

sort_folders moves the downloaded OpenFake files there: a rename on the same disk, nothing is copied.
load_pool reads the two folders through all their subfolders, adds the folders of data.real_dirs and the
rendered non-photo reals, and sends every image to train, val or test by the hash of "<folder>/<file>".
So the split is the same on every run, and an image keeps its part when images are added or the data move
to another disk: nothing leaks from test into training.
"""

import zlib
from collections import Counter
from pathlib import Path

from .download import load_rows, save_rows
from .render import render_folder

IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def sort_folders(data) -> None:
    """Moves the images listed in data/openfake_train.csv and openfake_test.csv to data/real/<source>/ and
    data/fake/<generator>/ and writes the new paths to the two files (the prompts stay there).
    Safe to run again: an image already in its folder is left alone."""
    data = Path(data)
    for split in ("train", "test"):
        rows, moved, made = load_rows(data / f"openfake_{split}.csv"), 0, set()
        for r in rows:
            old = Path(r["path"])
            folder = data / "fake" / r["generator"] if r["label"] else data / "real" / r["source"].split("/")[-1]
            if old.parts[-3:-1] != folder.parts[-2:]:  # not in real/<source>/ or fake/<generator>/ yet
                new = folder / f"{split}_{old.name}"  # train and test files may share a name
                if folder not in made:
                    folder.mkdir(parents=True, exist_ok=True)
                    made.add(folder)
                if old.exists():  # missing: moved by a run that stopped before writing the csv
                    old.rename(new)
                    moved += 1
                r["path"] = str(new)
        save_rows(rows, data / f"openfake_{split}.csv")
        print(f"OpenFake {split}: {moved} of {len(rows)} images moved to {data / 'real'} and {data / 'fake'}")


def images(folder) -> list:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGES)


def part(path, val: float, test: float) -> str:
    h = zlib.crc32("/".join(Path(path).parts[-2:]).encode()) % 10000 / 10000
    return "test" if h < test else "val" if h < test + val else "train"


def load_pool(d: dict) -> dict:
    """d = the `data` block of the config. Returns {"train": rows, "val": rows, "test": rows}; a row has "path",
    "label" (0 real, 1 generated), "generator" (the folder under fake/) and "source" (where a real comes from)."""
    data = Path(d["dir"])
    assert (data / "fake").is_dir(), f"no {data / 'fake'}: run scripts/download_data.py first, it sorts the images into folders"
    rows = [{"path": str(p), "label": 1, "generator": p.relative_to(data / "fake").parts[0], "source": "fake"}
            for p in images(data / "fake") if len(p.relative_to(data / "fake").parts) > 1]
    for folder, name in [(data / "real", ""), *((Path(f), Path(f).name) for f in d.get("real_dirs") or [])]:
        assert folder.is_dir(), f"no folder {folder}"
        for p in images(folder):  # source: the first subfolder, for data.real_dirs with the folder's own name
            rows.append({"path": str(p), "label": 0, "generator": "",
                         "source": "/".join(filter(None, [name, *p.relative_to(folder).parts[:-1][:1]])) or "real"})
    parts = {"train": [], "val": [], "test": []}
    for r in rows:
        parts[part(r["path"], d["val_share"], d["test_share"])].append(r)

    def renders(n, fonts):  # text pages, UI screens, charts, logos: reals that are not photos
        return [{"path": str(it["path"]), "label": 0, "generator": "", "source": f"render/{it['content']}"}
                for it in render_folder(data / "renders", n, fonts)]

    n_train = round(d.get("render_train", 0.0) * sum(r["label"] == 0 for r in parts["train"]))
    print(f"rendering {n_train} text pages, UI screens, charts, logos as training reals (once, then from disk)")
    drawn = renders(d["n_render"], "test")  # other fonts than in training
    parts["train"] += renders(n_train, "train")
    parts["val"] += drawn[: len(drawn) // 2]
    parts["test"] += drawn[len(drawn) // 2:]
    for name, part_rows in parts.items():
        n = Counter(r["label"] for r in part_rows)
        print(f"{name}: {n[0]} reals from {len({r['source'] for r in part_rows if not r['label']})} sources, "
              f"{n[1]} generated images of {len({r['generator'] for r in part_rows if r['label']})} generators")
    return parts
