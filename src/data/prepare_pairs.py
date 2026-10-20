"""Re-decode real images through the decoder bank and write aligned pairs.

For every real row of a manifest:
    <out>/<source>/<id>/real.png        the real image (shrunk to max_side, never enlarged)
    <out>/<source>/<id>/<decoder>.png   decode(encode(real)) for each decoder, same size
and one line in <out>/aligned_pairs.csv: pair_dir, decoders, content_type, source, split

    python -m src.data.prepare_pairs --manifest data/manifests/train_real.csv --out data/aligned --decoders sd15,sdxl,flux
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .dataset import PAIR_COLUMNS
from .decoders import DecoderBank
from .manifest import filter_manifest, read_manifest

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def shrink(image: Image.Image, max_side: int) -> Image.Image:
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    return image if scale == 1.0 else image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)


def pair_id(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]


def prepare_pairs(
    reals: pd.DataFrame,
    out: str | Path,
    decoders: list[str],
    max_side: int = 1024,
    min_side: int = 256,
    val_fraction: float = 0.02,
    device: str = "cuda",
    dtype: str = "float32",
    overwrite: bool = False,
) -> Path:
    """Writes pairs for every row of `reals` (label 0) and returns the path of aligned_pairs.csv."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pairs_csv = out / "aligned_pairs.csv"
    existing = pd.read_csv(pairs_csv, dtype=str, keep_default_na=False) if pairs_csv.exists() else pd.DataFrame(columns=PAIR_COLUMNS)

    bank = DecoderBank(decoders, device, DTYPES[dtype])
    rng = np.random.default_rng(0)
    rows, skipped = [], 0
    for _, row in tqdm(reals.iterrows(), total=len(reals), desc="re-decoding"):
        pair_dir = out / row["source"] / pair_id(row["path"])
        done = pair_dir.exists() and all((pair_dir / f"{d}.png").exists() for d in decoders)
        if done and not overwrite:
            rows.append(_row(pair_dir, out, decoders, row, rng, val_fraction))
            continue
        try:
            image = shrink(Image.open(row["path"]).convert("RGB"), max_side)
        except Exception as e:
            print(f"skip {row['path']}: {e}")
            skipped += 1
            continue
        if min(image.size) < min_side:
            skipped += 1
            continue
        pair_dir.mkdir(parents=True, exist_ok=True)
        image.save(pair_dir / "real.png")
        for name in decoders:
            bank.redecode(name, image).save(pair_dir / f"{name}.png")
        rows.append(_row(pair_dir, out, decoders, row, rng, val_fraction))

    new = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    merged = pd.concat([existing, new]).drop_duplicates("pair_dir", keep="last")
    merged.to_csv(pairs_csv, index=False)
    print(f"pairs: {len(merged)} total, {len(new)} from this run, {skipped} skipped -> {pairs_csv}")
    return pairs_csv


def _row(pair_dir: Path, out: Path, decoders: list[str], row, rng, val_fraction: float) -> dict:
    split = row["split"] if row["split"] in ("train", "val") else ("val" if rng.random() < val_fraction else "train")
    return {
        "pair_dir": str(pair_dir.relative_to(out)),
        "decoders": ";".join(decoders),
        "content_type": row["content_type"],
        "source": row["source"],
        "split": split,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="manifest CSV; only label=0 rows are used")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--decoders", default="sd15,sdxl,flux")
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--min-side", type=int, default=256, help="skip images smaller than this")
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=list(DTYPES))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    reals = filter_manifest(read_manifest(args.manifest, args.data_root), label=0)
    if args.limit:
        reals = reals.head(args.limit)
    prepare_pairs(reals, args.out, [d for d in args.decoders.split(",") if d], args.max_side, args.min_side,
                  args.val_fraction, args.device, args.dtype, args.overwrite)


if __name__ == "__main__":
    main()
