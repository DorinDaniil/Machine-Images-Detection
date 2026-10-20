"""Thresholds with a false-positive guarantee, fitted on real images only.

Conformal rule: with n real calibration scores, the threshold is the
ceil((n + 1)(1 - alpha))-th smallest score. A new real image then exceeds it with
probability at most alpha. One threshold per content type when enough data, else global.

    python -m trace_detector.calibrate_detector --weights checkpoints/.../best.pt \
        --manifest data/manifests/all.csv --alpha 0.01 --out checkpoints/.../thresholds.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import filter_manifest, read_manifest
from .evaluate_detector import load_model, score_manifest


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(scores)
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha)) - 1
    return float(s[k]) if k < n else float("inf")


def calibrate(scored_real: pd.DataFrame, alpha: float, min_per_type: int = 200) -> dict:
    out = {"alpha": alpha, "n": int(len(scored_real)), "global": conformal_threshold(scored_real["score"].to_numpy(), alpha)}
    out["per_content_type"] = {
        ct: conformal_threshold(g["score"].to_numpy(), alpha)
        for ct, g in scored_real.groupby("content_type")
        if len(g) >= min_per_type
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="calib")
    parser.add_argument("--alpha", type=float, default=0.01, help="target false-positive rate")
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)
    df = filter_manifest(read_manifest(args.manifest, args.data_root), split=args.split, label=0)
    print(f"calibrating on {len(df)} real images")

    scored = score_manifest(model, df, device, args.max_side)
    thresholds = calibrate(scored, args.alpha)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(thresholds, open(args.out, "w", encoding="utf-8"), indent=2)
    print(json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    main()
