"""python scripts/run_test.py --config configs/test.yaml

Scores the test manifests with trained weights. If the config names a real-only
calibration manifest, thresholds with FPR <= alpha are fitted first and applied.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import filter_manifest, read_manifest  # noqa: E402
from src.evaluate import calibrate, score_manifest, summarize, to_markdown  # noqa: E402
from src.model import load_model  # noqa: E402
from src.train import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = load_model(cfg["weights"], device)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = None
    if cfg.get("calib_manifest"):
        real = filter_manifest(read_manifest(cfg["calib_manifest"], cfg.get("data_root")), split=cfg.get("calib_split", "calib"), label=0)
        print(f"calibrating on {len(real)} real images, alpha={cfg.get('alpha', 0.01)}")
        thresholds = calibrate(score_manifest(model, real, device, cfg.get("max_side", 1024), cfg.get("num_workers", 4)), cfg.get("alpha", 0.01))
        json.dump(thresholds, open(out_dir / "thresholds.json", "w", encoding="utf-8"), indent=2)

    frames = [filter_manifest(read_manifest(m, cfg.get("data_root")), split=cfg.get("split")) for m in cfg["manifests"]]
    df = pd.concat(frames, ignore_index=True)
    print(f"scoring {len(df)} images on {device}")
    scored = score_manifest(model, df, device, cfg.get("max_side", 1024), cfg.get("num_workers", 4))
    summary = summarize(scored, thresholds)

    scored.to_csv(out_dir / "scores.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    print(to_markdown(summary))


if __name__ == "__main__":
    main()
