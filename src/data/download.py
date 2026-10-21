"""Open datasets from Hugging Face: only the kept images are written to disk.

OpenFake (ComplexDataLab/OpenFake, CC BY-NC 4.0): real photos and images of many generators, with the
    generator name for every image. core/train (2.3M rows, 3.3 TB in 608 parquet files, generators mixed in
    every file): reals from LAION and Pexels; fakes from SD 1.x to FLUX.2, GPT-Image-1, Imagen 4,
    Midjourney 6 and ~70 others. core/test (91k rows, 68 GB, grouped by generator): newer models
    (GPT-Image-1.5 and 2, Nano Banana Pro, Seedream 5, Midjourney 7, FLUX.2-klein-9B, Z-Image-Turbo, ...)
    and other real sources (DOCCI, ImageNet).
SID_Set (saberzl/SID_Set, CC BY 4.0): real (0), fully synthetic (1) and locally edited (2) images,
    with masks of the edited region.

Every image is stored the same way, whatever its label: long side at most 1024, JPEG q95. Images whose
short side is below `min_side` are skipped, also whatever the label. So file format and size carry no
information about the label.
"""

import csv
import io
import json
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from PIL import Image
from tqdm import tqdm

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")  # let hf_xet use the whole network channel

OPENFAKE = "ComplexDataLab/OpenFake"
COLUMNS = ["path", "label", "generator", "mask", "source"]
SKIP = {"tiny-random-sana"}  # OpenFake "generator" that is a random-weight test pipeline: pure noise
ATTEMPTS, RETRY_WAIT = 10, 30  # a broken file is tried up to 10 times, waiting 30 s, 60 s, ...


def save_rows(rows: list[dict], path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [{**r, "label": int(r["label"])} for r in csv.DictReader(f)]


def _store(data: bytes, path: Path, max_side: int, min_side: int):
    """Decode, skip if too small or broken, shrink to max_side, save JPEG q95. Returns the stored size or None."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    if min(im.size) < min_side:
        return None
    scale = max_side / max(im.size)
    if scale < 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    im.save(path, format="JPEG", quality=95)
    return im.size


def _stream(repo: str, config: str, split: str, columns: list[str]):
    from datasets import Image as HFImage
    from datasets import load_dataset

    ds = load_dataset(repo, config, split=split, streaming=True)
    for col in columns:
        ds = ds.cast_column(col, HFImage(decode=False))
    return ds


def _files(split: str) -> list[str]:
    """Parquet files of OpenFake core/<split>, e.g. core/train-00000-of-00032-00000.parquet."""
    from huggingface_hub import HfApi

    names = HfApi().list_repo_files(OPENFAKE, repo_type="dataset")
    return sorted(n for n in names if n.startswith(f"core/{split}-") and n.endswith(".parquet"))


def _rows(name: str, tmp_dir: Path):
    """Downloads one parquet file whole (hf_xet: many connections, resumes partial downloads), yields its rows
    as dicts {"image": {"bytes"}, "label", "model"}, then deletes the file."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(OPENFAKE, name, repo_type="dataset", local_dir=tmp_dir))
    try:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=["image", "label", "model"]):
            yield from batch.to_pylist()
    finally:
        path.unlink(missing_ok=True)


def _take_file(rows_iter, i: int, out_dir: Path, quota: dict, counts: Counter, taken: Counter, lock,
               real_sources, max_side: int, min_side: int) -> dict:
    """Keeps images of file i while their shared quota is not full; `taken` records what this attempt reserved.
    Returns {"read": rows read, "rows": kept rows}."""
    rows, read = [], 0
    for j, row in enumerate(rows_iter):
        read += 1
        label, model = int(row["label"] == "fake"), row["model"]
        if model in SKIP or (label == 0 and real_sources and model not in real_sources):
            continue
        key = model if label else "real"
        with lock:
            if counts[key] >= quota[label]:
                continue
            counts[key] += 1
            taken[key] += 1
        path = out_dir / f"{i:04d}_{j:05d}.jpg"
        if _store(row["image"]["bytes"], path, max_side, min_side):
            rows.append({"path": str(path), "label": label, "generator": model if label else "",
                         "mask": "", "source": "openfake" if label else f"openfake/{model}"})
        else:
            with lock:
                counts[key] -= 1
                taken[key] -= 1
    return {"read": read, "rows": rows}


def openfake(split: str, out_dir, n_real: int, n_per_model: int, max_rows=None, real_sources=None,
             workers: int = 8, max_side: int = 1024, min_side: int = 512) -> list[dict]:
    """Up to n_real reals (optionally only from real_sources) and up to n_per_model images of every generator
    (all of them if it has fewer), from OpenFake core/<split>. No new file is started once max_rows rows are
    read (None: whole split).

    `workers` threads each download a whole parquet file (~5.4 GB of temporary disk per thread), keep the
    images whose quota is not full, delete the file and record it in <out_dir>/done/. A file that breaks is
    retried; a rerun after a crash or Ctrl+C skips finished files. Result cached in <out_dir>.csv."""
    out_dir = Path(out_dir)
    cache, done_dir, tmp_dir = out_dir.with_suffix(".csv"), out_dir / "done", out_dir / "tmp"
    if cache.exists():
        return load_rows(cache)
    done_dir.mkdir(parents=True, exist_ok=True)

    files = _files(split)
    done = [json.loads(p.read_text()) for p in done_dir.glob("*.json")]
    counts = Counter(r["generator"] if r["label"] else "real" for d in done for r in d["rows"])
    state = {"read": sum(d["read"] for d in done), "kept": sum(len(d["rows"]) for d in done)}
    todo = [i for i in range(len(files)) if not (done_dir / f"{i:04d}.json").exists()]
    free = shutil.disk_usage(out_dir).free / 1e9
    print(f"OpenFake {split}: {len(files) - len(todo)} of {len(files)} files done, {free:.0f} GB free on disk")
    quota, lock = {0: n_real, 1: n_per_model}, Lock()

    with tqdm(total=len(files), initial=len(files) - len(todo), unit="file", desc=f"OpenFake {split}") as bar:
        def run(i):
            for attempt in range(1, ATTEMPTS + 1):
                if max_rows is not None and state["read"] >= max_rows:
                    return
                taken = Counter()
                try:
                    part = _take_file(_rows(files[i], tmp_dir), i, out_dir, quota, counts, taken, lock,
                                      real_sources, max_side, min_side)
                    break
                except Exception as e:  # network or file error: give the quota back, wait, try again
                    with lock:
                        counts.subtract(taken)
                    if attempt == ATTEMPTS:
                        raise
                    tqdm.write(f"{files[i]}: {e!r}; retry {attempt}/{ATTEMPTS - 1} in {RETRY_WAIT * attempt} s")
                    time.sleep(RETRY_WAIT * attempt)
            tmp = done_dir / f"{i:04d}.tmp"
            tmp.write_text(json.dumps(part))
            tmp.replace(done_dir / f"{i:04d}.json")  # appears only when the file is complete
            with lock:
                state["read"] += part["read"]
                state["kept"] += len(part["rows"])
                bar.set_postfix_str(f"rows read {state['read']}, images kept {state['kept']}")
                bar.update(1)

        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(run, todo))

    rows = [r for p in sorted(done_dir.glob("*.json")) for r in json.loads(p.read_text())["rows"]]
    save_rows(rows, cache)
    gens = Counter(r["generator"] for r in rows if r["label"])
    print(f"OpenFake {split}: {len(rows) - sum(gens.values())} reals, {sum(gens.values())} fakes from {len(gens)} generators")
    return rows


def sid_set(split: str, out_dir, n_per_class: int, max_rows: int, max_side: int = 1024, min_side: int = 512) -> list[dict]:
    """SID_Set: n_per_class reals, fully synthetic and locally edited images (with masks). Cached in <out_dir>.csv."""
    out_dir = Path(out_dir)
    cache = out_dir.with_suffix(".csv")
    if cache.exists():
        return load_rows(cache)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = {0: "real", 1: "full", 2: "edit"}
    rows, counts = [], Counter()
    for i, row in enumerate(tqdm(_stream("saberzl/SID_Set", "default", split, ["image", "mask"]), total=max_rows, desc=f"SID_Set {split}")):
        if i >= max_rows or min(counts[k] for k in names.values()) >= n_per_class:
            break
        kind = names[int(row["label"])]
        if counts[kind] >= n_per_class:
            continue
        path = out_dir / f"{i:07d}.jpg"
        size = _store(row["image"]["bytes"], path, max_side, min_side)
        if not size:
            continue
        mask = ""
        if kind == "edit" and row["mask"] and row["mask"]["bytes"]:
            mask = str(out_dir / f"{i:07d}_mask.png")
            with Image.open(io.BytesIO(row["mask"]["bytes"])) as m:
                m.convert("L").resize(size, Image.NEAREST).point(lambda v: 255 if v > 127 else 0).save(mask)
        counts[kind] += 1
        rows.append({"path": str(path), "label": int(kind != "real"), "generator": "" if kind == "real" else f"sid-{kind}",
                     "mask": mask, "source": f"sid/{kind}"})
    save_rows(rows, cache)
    print(f"SID_Set {split}: {dict(counts)}")
    return rows
