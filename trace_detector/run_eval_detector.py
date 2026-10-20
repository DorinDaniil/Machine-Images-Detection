"""Evaluate exported weights on one or more manifests.

    python -m trace_detector.run_eval_detector --config trace_detector/configs/eval_config_detector.yaml
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .data import filter_manifest, read_manifest
from .evaluate_detector import load_model, score_manifest, summarize, to_markdown
from .train_detector import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = load_model(cfg["weights"], device)
    thresholds = json.load(open(cfg["thresholds"], encoding="utf-8")) if cfg.get("thresholds") else None

    frames = [filter_manifest(read_manifest(m, cfg.get("data_root")), split=cfg.get("split")) for m in cfg["manifests"]]
    df = pd.concat(frames, ignore_index=True)
    print(f"scoring {len(df)} images on {device}")

    scored = score_manifest(model, df, device, cfg.get("max_side", 1024), cfg.get("num_workers", 4))
    summary = summarize(scored, thresholds)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_dir / "scores.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    print(to_markdown(summary))


if __name__ == "__main__":
    main()
