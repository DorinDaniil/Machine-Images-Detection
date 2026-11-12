"""The image folders and the split into train, val and test.

    data/real/<source>/...        real images: laion, docci, ...; any folder put here is picked up
    data/fake/<generator>/...     generated images; the folder name is the generator

The downloader writes OpenFake there (src/data/download.py). load_pool reads the two folders through all
their subfolders, adds the folders of data.real_dirs and the rendered non-photo reals, and sends every image
to train, val or test by the hash of "<folder>/<file>".
So the split is the same on every run, and an image keeps its part when images are added or the data move
to another disk: nothing leaks from test into training.
"""

import zlib
from collections import Counter
from pathlib import Path

from .render import render_folder

IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def images(folder) -> list:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGES)


def part(path, val: float, test: float) -> str:
    h = zlib.crc32("/".join(Path(path).parts[-2:]).encode()) % 10000 / 10000
    return "test" if h < test else "val" if h < test + val else "train"


def load_pool(d: dict) -> dict:
    """d = the `data` block of the config. Returns {"train": rows, "val": rows, "test": rows}; a row has "path",
    "label" (0 real, 1 generated), "generator" (the folder under fake/) and "source" (where a real comes from)."""
    data = Path(d["dir"])
    assert (data / "fake").is_dir(), f"no {data / 'fake'}: download the data first, scripts/download_data.py"
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


def data_table(parts: dict, families: dict) -> str:
    """The data as a table: a line per real source and per generator (with its family), the number of its images
    in train, val and test and the share of them stored as JPEG; after each kind, its total."""
    family = {g: name for name, models in families.items() for g in models}
    count, jpeg = {}, Counter()
    for name, rows in parts.items():
        for r in rows:
            key = (r["label"], family.get(r["generator"], "") if r["label"] else "", r["generator"] or r["source"])
            count.setdefault(key, Counter())[name] += 1
            jpeg[key] += r["path"].lower().endswith((".jpg", ".jpeg"))

    def line(title, fam, n, n_jpeg):
        total = sum(n.values())
        return f"| {title} | {fam} | {n['train']} | {n['val']} | {n['test']} | {total} | {n_jpeg / max(total, 1):.0%} |"

    lines = ["| images | family | train | val | test | all | JPEG |", "|---|---|---|---|---|---|---|"]
    for label, kind in ((0, "real"), (1, "fake")):
        keys = sorted((k for k in count if k[0] == label), key=lambda k: (not k[1], k))  # by family; no family last
        lines += [line(f"{kind}: {k[2]}", k[1], count[k], jpeg[k]) for k in keys]
        lines.append(line(f"**all {kind}**", "", sum((count[k] for k in keys), Counter()), sum(jpeg[k] for k in keys)))
    return "\n".join(lines)
