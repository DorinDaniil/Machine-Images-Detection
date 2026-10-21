import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .model import TraceDetector


@torch.no_grad()
def score(model: TraceDetector, dataset, device: str):
    """Image scores and pixel F1 of the patch map (NaN where there is no mask). One image at a time."""
    model.eval().to(device)
    scores, f1 = np.zeros(len(dataset)), np.full(len(dataset), np.nan)
    for i, item in enumerate(DataLoader(dataset, batch_size=1, num_workers=2)):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            patch_logits, image_logit = model(item["image"].to(device))
        scores[i] = torch.sigmoid(image_logit.float()).item()
        if item["has_mask"].item():
            mask = item["mask"].to(device) > 0.5
            pred = F.interpolate(patch_logits.float(), size=mask.shape[-2:], mode="bilinear").sigmoid() > 0.5
            f1[i] = (2 * (pred & mask).sum() / (pred.sum() + mask.sum()).clamp(min=1)).item()
    return scores, f1


def threshold_at(calib_real_scores: np.ndarray, alpha: float) -> float:
    """Conformal threshold from real images only: a new real image exceeds it with probability <= alpha."""
    s = np.sort(calib_real_scores)
    k = math.ceil((len(s) + 1) * (1 - alpha)) - 1
    return float(s[min(k, len(s) - 1)])


def report(calib_real: np.ndarray, test_real: np.ndarray, groups: dict, alpha: float = 0.01) -> str:
    """Markdown table: every fake group against the same test reals.

    The threshold comes from separate calibration reals, so the FPR on the test reals is a measurement, not a given.
    """
    thr = threshold_at(calib_real, alpha)
    lines = [f"threshold for {alpha:.0%} FPR from {len(calib_real)} calibration reals: {thr:.4f}",
             f"measured FPR on {len(test_real)} test reals: {(test_real > thr).mean():.3f}", "",
             "| group | n | AUROC | TPR at threshold | pixel F1 |", "|---|---|---|---|---|"]
    for name, (scores, f1) in groups.items():
        labels = np.r_[np.zeros(len(test_real)), np.ones(len(scores))]
        auroc = roc_auc_score(labels, np.r_[test_real, scores])
        pixel = f"{np.nanmean(f1):.3f}" if np.isfinite(f1).any() else ""
        lines.append(f"| {name} | {len(scores)} | {auroc:.3f} | {(scores > thr).mean():.3f} | {pixel} |")
    return "\n".join(lines)
