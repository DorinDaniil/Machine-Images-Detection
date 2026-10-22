"""Open datasets from Hugging Face. Parquet files are downloaded whole (hf_xet: many connections, partial
downloads resume), the kept images are written to disk and the file is deleted.

OpenFake (ComplexDataLab/OpenFake, v2, CC BY-NC 4.0): real photos and images of ~70 generators, with the
    generator name for every image. core/train: 2.3M rows, 3.3 TB, 608 files, generators mixed in every file.
    core/test: 91k rows, 68 GB, 13 files, newer generators only (GPT-Image-1.5 and 2, Nano Banana Pro,
    Midjourney 7, FLUX.2-klein-9B, Z-Image-Turbo, ...), reals from DOCCI and ImageNet.
SID_Set (saberzl/SID_Set, CC BY 4.0): real (0), fully synthetic (1) and locally edited (2) images with masks
    of the edited region; validation: 30k rows, 17 GB, 34 files.

Images are stored as their original bytes: no re-encoding, no resizing, nothing is lost. Fakes are mostly
PNG and reals mostly JPEG, so the file format does correlate with the label; the training pipeline
(dataset.py) removes that by JPEG-compressing every training image at a random quality. Images whose short
side is below `min_side` are skipped, whatever the label.
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

OPENFAKE, SID_SET = "ComplexDataLab/OpenFake", "saberzl/SID_Set"
META = ["prompt", "prompt_kind", "row_id"]  # OpenFake text columns kept with every image (pairs by prompt)
COLUMNS = ["path", "label", "generator", "mask", "source", *META]
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


def _store(data: bytes, stem: Path, min_side: int):
    """Writes the original bytes untouched, with the extension of their format (only the header is read).
    Skips broken images and images whose short side is below min_side. Returns (path, size) or None."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            size, fmt = im.size, im.format or "bin"
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    if min(size) < min_side:
        return None
    path = stem.with_suffix("." + {"JPEG": "jpg"}.get(fmt, fmt.lower()))
    path.write_bytes(data)
    return path, size


def _files(repo: str, prefix: str) -> list[str]:
    """Parquet files of a split, e.g. prefix "core/train-" -> core/train-00000-of-00032-00000.parquet, ..."""
    from huggingface_hub import HfApi

    names = HfApi().list_repo_files(repo, repo_type="dataset")
    return sorted(n for n in names if n.startswith(prefix) and n.endswith(".parquet"))


def _rows(repo: str, name: str, tmp_dir: Path, columns: list[str]):
    """Downloads one parquet file whole, yields its rows as dicts (images as {"bytes": ...}), deletes the file."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo, name, repo_type="dataset", local_dir=tmp_dir))
    try:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=columns):
            yield from batch.to_pylist()
    finally:
        path.unlink(missing_ok=True)


def _columns(repo: str, name: str, columns: list[str]) -> list[dict]:
    """Reads only `columns` of a remote parquet file (a few MB), without downloading its images."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(f"datasets/{repo}/{name}", "rb") as f:
        return pq.read_table(f, columns=columns).to_pylist()


def _take_file(rows_iter, i: int, out_dir: Path, quota: dict, counts: Counter, taken: Counter, lock,
               real_sources, generators, min_side: int) -> dict:
    """Keeps images of file i while their shared quota is not full; `taken` records what this attempt reserved."""
    rows, read = [], 0
    for j, row in enumerate(rows_iter):
        read += 1
        label, model = int(row["label"] == "fake"), row["model"]
        wanted = model not in SKIP and (not generators or model in generators) if label else (not real_sources or model in real_sources)
        if not wanted:
            continue
        key = model if label else "real"
        with lock:
            if counts[key] >= quota[label]:
                continue
            counts[key] += 1
            taken[key] += 1
        stored = _store(row["image"]["bytes"], out_dir / f"{i:04d}_{j:05d}", min_side)
        if stored:
            rows.append({"path": str(stored[0]), "label": label, "generator": model if label else "",
                         "mask": "", "source": "openfake" if label else f"openfake/{model}",
                         **{k: row.get(k) or "" for k in META}})
        else:
            with lock:
                counts[key] -= 1
                taken[key] -= 1
    return {"read": read, "rows": rows}


def openfake(split: str, out_dir, n_real=None, n_per_model=None, real_sources=None, generators=None,
             max_files=None, workers: int = 8, min_side: int = 512) -> list[dict]:
    """Up to n_real reals (optionally only from real_sources) and up to n_per_model images of every generator
    (optionally only those in `generators`; None: no limit) from the first max_files files of OpenFake
    core/<split> (None: all). With a generator list the download stops as soon as every quota is full.

    `workers` files are downloaded at once (~5.4 GB of temporary disk each). Every finished file is recorded
    in <out_dir>/done/, so a rerun after a crash or Ctrl+C skips it; a file that breaks is retried.
    Every row keeps the OpenFake prompt, prompt_kind and row_id; files finished before these were kept get
    them filled in on the next run from the text columns alone. Result cached in <out_dir>.csv."""
    out_dir = Path(out_dir)
    cache, done_dir, tmp_dir = out_dir.with_suffix(".csv"), out_dir / "done", out_dir / "tmp"
    if cache.exists():
        return load_rows(cache)
    done_dir.mkdir(parents=True, exist_ok=True)

    files = _files(OPENFAKE, f"core/{split}-")[:max_files]
    done = {p: json.loads(p.read_text()) for p in sorted(done_dir.glob("*.json"))}
    stale = [p for p, d in done.items() if any("row_id" not in r for r in d["rows"])]
    for p in tqdm(stale, desc=f"OpenFake {split}: prompts of finished files"):  # files done before META was kept
        meta = _columns(OPENFAKE, files[int(p.stem)], META)
        for r in done[p]["rows"]:
            r.update({k: meta[int(Path(r["path"]).stem.split("_")[1])][k] or "" for k in META})
        p.write_text(json.dumps(done[p]))
    done = list(done.values())
    counts = Counter(r["generator"] if r["label"] else "real" for d in done for r in d["rows"])
    state = {"read": sum(d["read"] for d in done), "kept": sum(len(d["rows"]) for d in done)}
    todo = [i for i in range(len(files)) if not (done_dir / f"{i:04d}.json").exists()]
    print(f"OpenFake {split}: {len(files) - len(todo)} of {len(files)} files done, "
          f"{shutil.disk_usage(out_dir).free / 1e9:.0f} GB free on disk")
    quota = {0: float("inf") if n_real is None else n_real, 1: float("inf") if n_per_model is None else n_per_model}
    lock = Lock()

    def full() -> bool:  # every quota reached: only knowable with a generator list
        with lock:
            return bool(generators) and counts["real"] >= quota[0] and all(counts[g] >= quota[1] for g in generators)

    with tqdm(total=len(files), initial=len(files) - len(todo), unit="file", desc=f"OpenFake {split}") as bar:
        def run(i):
            if full():
                return
            for attempt in range(1, ATTEMPTS + 1):
                taken = Counter()
                try:
                    part = _take_file(_rows(OPENFAKE, files[i], tmp_dir, ["image", "label", "model", *META]), i, out_dir,
                                      quota, counts, taken, lock, real_sources, generators, min_side)
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
    for g, n in gens.most_common():
        print(f"  {n:7d}  {g}")
    return rows


def sid_set(split: str, out_dir, n_per_class: int, min_side: int = 512) -> list[dict]:
    """SID_Set: n_per_class reals, fully synthetic and locally edited images (with masks). Rows are shuffled, so
    only the first few files are downloaded. Cached in <out_dir>.csv."""
    out_dir = Path(out_dir)
    cache = out_dir.with_suffix(".csv")
    if cache.exists():
        return load_rows(cache)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = {0: "real", 1: "full", 2: "edit"}
    rows, counts = [], Counter()
    for name in _files(SID_SET, f"data/{split}-"):
        for i, row in enumerate(_rows(SID_SET, name, out_dir / "tmp", ["image", "mask", "label"]), start=len(rows)):
            kind = names[int(row["label"])]
            if counts[kind] >= n_per_class:
                continue
            stored = _store(row["image"]["bytes"], out_dir / f"{i:06d}", min_side)
            if not stored:
                continue
            path, size = stored
            mask = ""
            if kind == "edit" and row["mask"] and row["mask"]["bytes"]:
                mask = str(out_dir / f"{i:06d}_mask.png")
                with Image.open(io.BytesIO(row["mask"]["bytes"])) as m:
                    m.convert("L").resize(size, Image.NEAREST).point(lambda v: 255 if v > 127 else 0).save(mask)
            counts[kind] += 1
            rows.append({"path": str(path), "label": int(kind != "real"), "generator": "" if kind == "real" else f"sid-{kind}",
                         "mask": mask, "source": f"sid/{kind}"})
        if min(counts[k] for k in names.values()) >= n_per_class:
            break
    save_rows(rows, cache)
    print(f"SID_Set {split}: {dict(counts)}")
    return rows
