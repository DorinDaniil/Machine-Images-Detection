"""One command, one question: does the decoder trace learned on one VAE transfer to unseen
decoder families and to real generator outputs?

    python scripts/run_pilot.py --config configs/train.yaml
    python scripts/run_pilot.py --config configs/smoke.yaml --n-real 30 --test-decoders sdxl --bench-items 40  # CPU dry run

Steps: COCO reals -> 80/20 split -> train pairs with --train-decoders -> held-out pairs with
--test-decoders -> train -> thresholds on held-out reals -> metrics on held-out fakes and on the
benchmark (DailyBench: FLUX.2, Qwen-Image, Nano Banana 2, GPT-Image 2 and their edits).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import compose, random_region_mask, read_manifest, write_manifest  # noqa: E402
from src.data.manifest import COLUMNS  # noqa: E402
from src.data.prepare_pairs import prepare_pairs  # noqa: E402
from src.data.registry import build_source, load_registry  # noqa: E402
from src.evaluate import calibrate, score_manifest, summarize, to_markdown  # noqa: E402
from src.model import load_model  # noqa: E402
from src.train import build_loaders, build_model, load_config, train_model  # noqa: E402


def pull(reg: dict, name: str, max_items: int) -> pd.DataFrame:
    """Download one registry entry and return its manifest rows with absolute paths."""
    entry = {k: v for k, v in reg["datasets"][name].items() if k != "role"}
    source, _ = build_source(name, entry)
    raw_root = Path(reg["raw_root"])
    source.download(raw_root, max_items)
    df = pd.DataFrame(source.rows(raw_root), columns=COLUMNS).head(max_items)
    for col in ("path", "mask_path"):
        df[col] = [p if (not p or Path(p).is_absolute()) else str(raw_root / p) for p in df[col]]
    return df


def heldout_test_set(pairs_csv: Path, out: Path, rng: np.random.Generator) -> pd.DataFrame:
    """From held-out pairs make a test manifest: the real image, a full re-decode and a local paste per decoder."""
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(pairs_csv, dtype=str, keep_default_na=False)
    rows = []
    for i, row in pairs.iterrows():
        pair_dir = pairs_csv.parent / row["pair_dir"]
        real = np.asarray(Image.open(pair_dir / "real.png").convert("RGB"))
        rows.append(dict(path=str(pair_dir / "real.png"), label=0, mask_path="", generator="", content_type="photo",
                         source="heldout", split="test"))
        for dec in row["decoders"].split(";"):
            fake = np.asarray(Image.open(pair_dir / f"{dec}.png").convert("RGB"))
            full, _ = compose(real, fake, np.ones(real.shape[:2], np.float32), 1.0)
            mask = random_region_mask(*real.shape[:2], rng)
            local, _ = compose(real, fake, mask, 1.0)
            for kind, img, m in (("full", full, np.ones_like(mask)), ("local", local, mask)):
                stem = out / f"{i:05d}_{dec}_{kind}"
                Image.fromarray(img).save(f"{stem}.png")
                Image.fromarray((m > 0.5).astype(np.uint8) * 255).save(f"{stem}_mask.png")
                rows.append(dict(path=f"{stem}.png", label=1, mask_path=f"{stem}_mask.png", generator=f"{dec}-vae-{kind}",
                                 content_type="photo", source="heldout", split="test"))
    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--registry", default="configs/datasets.yaml")
    p.add_argument("--n-real", type=int, default=2000, help="COCO photos to use (80 % train, 20 % held out)")
    p.add_argument("--train-decoders", default="sd15")
    p.add_argument("--test-decoders", default="sdxl,flux", help="unseen decoder families for the held-out test")
    p.add_argument("--bench", default="dailybench", help="registry entry of the external benchmark, '' to skip")
    p.add_argument("--bench-items", type=int, default=1000)
    p.add_argument("--max-side", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    p.add_argument("--alpha", type=float, default=0.01, help="target false-positive rate for the threshold")
    p.add_argument("--out", default="runs/pilot")
    args = p.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    # 1. real photos, split into train and held-out
    reg = load_registry(args.registry)
    reals = pull(reg, "coco_real", args.n_real)
    reals = reals.sample(frac=1.0, random_state=0).reset_index(drop=True)
    n_train = int(0.8 * len(reals))
    train_reals, test_reals = reals.iloc[:n_train], reals.iloc[n_train:]
    print(f"reals: {len(train_reals)} train, {len(test_reals)} held out")

    # 2. aligned pairs: seen decoders for training, unseen ones for the held-out test
    train_pairs = prepare_pairs(train_reals, out / "pairs_train", args.train_decoders.split(","), args.max_side, device=device)
    test_pairs = prepare_pairs(test_reals, out / "pairs_test", args.test_decoders.split(","), args.max_side, val_fraction=0.0, device=device)

    # 3. train
    cfg["data"]["pairs_csv"] = str(train_pairs)
    cfg["training"]["checkpoint_dir"] = str(out / "checkpoints")
    cfg["training"]["log_dir"] = str(out / "logs")
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    torch.manual_seed(cfg.get("seed", 0))
    model = build_model(cfg)
    best = train_model(model, *build_loaders(cfg, model), cfg)

    # 4. test sets: held-out decoders (with masks) and the external benchmark
    heldout = heldout_test_set(test_pairs, out / "heldout_test", rng)
    write_manifest(heldout, out / "heldout_test.csv")
    frames = [heldout]
    if args.bench:
        bench = pull(reg, args.bench, args.bench_items)
        write_manifest(bench, out / f"{args.bench}.csv")
        frames.append(bench)
    test_df = pd.concat(frames, ignore_index=True)

    # 5. thresholds on held-out reals only, then metrics
    model = load_model(best, torch.device(device))
    scored = score_manifest(model, test_df, torch.device(device), args.max_side, num_workers=0)
    thresholds = calibrate(scored[(scored["label"] == 0) & (scored["source"] == "heldout")], args.alpha, min_per_type=1)
    summary = summarize(scored, thresholds)
    scored.to_csv(out / "scores.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    print(f"\nthreshold (FPR <= {args.alpha}): {thresholds['global']:.4f}\n")
    print(to_markdown(summary))


if __name__ == "__main__":
    main()
