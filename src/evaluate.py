import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .model import Detector


@torch.no_grad()
def score(model: Detector, dataset, device: str, tile: int = 512, workers: int = 4):
    """Image scores and pixel F1 of the patch map (NaN where there is no mask). Whole images, one at a time;
    large ones are looked at tile by tile (Detector.full), so no part of an image is left out."""
    model.eval().to(device)
    scores, f1 = np.zeros(len(dataset)), np.full(len(dataset), np.nan)
    for i, item in enumerate(DataLoader(dataset, batch_size=1, num_workers=workers)):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            patch_logits, image_logit = model.full(item["image"].to(device), tile)
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


def report(threshold: float, reals: dict, fakes: list) -> str:
    """reals: {name: scores} -> measured FPR at the threshold.
    fakes: [(name, real scores of the same source, fake scores, pixel F1)] -> AUROC and TPR per group."""
    lines = ["| real set | n | FPR at threshold |", "|---|---|---|"]
    lines += [f"| {name} | {len(s)} | {(s > threshold).mean():.3f} |" for name, s in reals.items()]
    lines += ["", "| fake group | n | AUROC vs its reals | TPR at threshold | pixel F1 |", "|---|---|---|---|---|"]
    for name, real, fake, f1 in fakes:
        auroc = roc_auc_score(np.r_[np.zeros(len(real)), np.ones(len(fake))], np.r_[real, fake])
        pixel = f"{np.nanmean(f1):.3f}" if np.isfinite(f1).any() else ""
        lines.append(f"| {name} | {len(fake)} | {auroc:.3f} | {(fake > threshold).mean():.3f} | {pixel} |")
    return "\n".join(lines)
