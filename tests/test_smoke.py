"""End-to-end smoke test on CPU with a randomly initialised tiny backbone.

Builds fake "aligned pairs" from synthetic images (no VAE, no download), then runs
dataset -> model -> loss -> one training epoch -> scoring -> summary -> calibration.

    python -m pytest tests -q
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from trace_detector.calibrate_detector import calibrate
from trace_detector.data import AlignedPairsDataset, compose, random_region_mask, read_manifest, write_manifest
from trace_detector.data.dataset import PAIR_COLUMNS
from trace_detector.dino_trace import TraceDetector, trace_loss
from trace_detector.evaluate_detector import load_model, precision_at_recall, score_manifest, summarize
from trace_detector.train_detector import build_loaders, load_config, train_model

CONFIG = Path(__file__).resolve().parents[1] / "trace_detector" / "configs" / "train_config_smoke.yaml"


def smooth_image(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """Low-frequency colour noise that looks vaguely like a photo."""
    coarse = rng.uniform(0, 255, size=(8, 8, 3)).astype(np.float32)
    return np.asarray(Image.fromarray(coarse.astype(np.uint8)).resize((size, size), Image.BICUBIC))


def fake_redecode(real: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stand-in for a VAE round trip: slight blur plus a faint periodic grid."""
    blurred = np.asarray(Image.fromarray(real).resize((real.shape[1] // 2, real.shape[0] // 2)).resize(real.shape[1::-1]))
    grid = (np.indices(real.shape[:2]).sum(0) % 8 == 0)[..., None] * 6.0
    return np.clip(blurred.astype(np.float32) + grid + rng.normal(0, 2, real.shape), 0, 255).astype(np.uint8)


def make_pairs(root: Path, n: int = 8) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        pair_dir = root / "synthetic" / f"{i:04d}"
        pair_dir.mkdir(parents=True)
        real = smooth_image(rng)
        Image.fromarray(real).save(pair_dir / "real.png")
        Image.fromarray(fake_redecode(real, rng)).save(pair_dir / "sd15.png")
        rows.append({"pair_dir": str(pair_dir.relative_to(root)), "decoders": "sd15", "content_type": "photo",
                     "source": "synthetic", "split": "train" if i < n - 2 else "val"})
    csv = root / "aligned_pairs.csv"
    pd.DataFrame(rows, columns=PAIR_COLUMNS).to_csv(csv, index=False)
    return csv


def make_test_manifest(root: Path, n: int = 6) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        real = smooth_image(rng)
        label = i % 2
        path, mask_path = root / f"img_{i}.png", ""
        if label:
            mask = random_region_mask(256, 256, rng)
            image, _ = compose(real, fake_redecode(real, rng), mask, 1.0)
            mask_path = root / f"mask_{i}.png"
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
        else:
            image = real
        Image.fromarray(image).save(path)
        rows.append({"path": str(path), "label": label, "mask_path": str(mask_path), "generator": "sd15" if label else "",
                     "content_type": "photo", "source": "synthetic", "split": "test"})
    manifest = root / "test.csv"
    write_manifest(pd.DataFrame(rows), manifest)
    return manifest


def test_dataset_and_loss(tmp_path):
    cfg = load_config(CONFIG)
    pairs = make_pairs(tmp_path / "aligned")
    dataset = AlignedPairsDataset(pairs, cfg["data"], split="train")
    item = dataset[0]
    assert item["image"].shape == (3, 224, 224) and item["target"].shape == (1, 224, 224)
    assert 0.0 <= float(item["label"]) <= 1.0

    model = TraceDetector.from_config(cfg["model"])
    out = model(item["image"][None])
    assert out["patch_logits"].shape == (1, 1, 16, 16) and out["image_logit"].shape == (1,)
    losses = trace_loss(out["patch_logits"], out["image_logit"], item["target"][None], cfg["loss"])
    losses["total"].backward()
    assert torch.isfinite(losses["total"])


def test_train_score_calibrate(tmp_path):
    cfg = load_config(CONFIG)
    cfg["data"]["pairs_csv"] = str(make_pairs(tmp_path / "aligned"))
    cfg["training"]["checkpoint_dir"] = str(tmp_path / "ckpt")
    cfg["training"]["log_dir"] = str(tmp_path / "logs")

    model = TraceDetector.from_config(cfg["model"])
    train_loader, val_loader = build_loaders(cfg, model)
    best = train_model(model, train_loader, val_loader, cfg)
    assert best.exists()

    device = torch.device("cpu")
    model = load_model(best, device)
    df = read_manifest(make_test_manifest(tmp_path / "test"))
    scored = score_manifest(model, df, device, max_side=256, num_workers=0, amp=False)
    assert scored["score"].between(0, 1).all()
    assert scored.loc[scored["label"] == 1, "pixel_f1"].notna().all()

    summary = summarize(scored)
    assert summary.iloc[0]["group"] == "ALL" and "auroc" in summary.columns

    thresholds = calibrate(scored[scored["label"] == 0], alpha=0.5, min_per_type=1)
    assert 0.0 <= thresholds["global"] <= 1.0
    json.dumps(thresholds)


def test_precision_at_recall():
    labels = np.array([1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert precision_at_recall(labels, scores, 0.5) == 1.0
    assert abs(precision_at_recall(labels, scores, 1.0) - 2 / 3) < 1e-9
