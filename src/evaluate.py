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
    for i, item in enumerate(DataLoader(dataset, batch_size=1)):
        patch_logits, image_logit = model(item["image"].to(device))
        scores[i] = torch.sigmoid(image_logit).item()
        if item["has_mask"].item():
            mask = item["mask"].to(device) > 0.5
            pred = F.interpolate(patch_logits, size=mask.shape[-2:], mode="bilinear").sigmoid() > 0.5
            f1[i] = (2 * (pred & mask).sum() / (pred.sum() + mask.sum()).clamp(min=1)).item()
    return scores, f1


def report(real_scores: np.ndarray, groups: dict, fpr: float = 0.01) -> str:
    """Markdown table: every fake group against the same real images.

    The threshold is the score that only `fpr` of the real images exceed.
    """
    threshold = np.quantile(real_scores, 1 - fpr)
    lines = [f"real images: {len(real_scores)}, threshold at {fpr:.0%} FPR: {threshold:.4f}", "",
             "| group | n | AUROC | TPR at threshold | pixel F1 |", "|---|---|---|---|---|"]
    for name, (scores, f1) in groups.items():
        labels = np.r_[np.zeros(len(real_scores)), np.ones(len(scores))]
        auroc = roc_auc_score(labels, np.r_[real_scores, scores])
        tpr = (scores > threshold).mean()
        pixel = f"{np.nanmean(f1):.3f}" if np.isfinite(f1).any() else ""
        lines.append(f"| {name} | {len(scores)} | {auroc:.3f} | {tpr:.3f} | {pixel} |")
    return "\n".join(lines)
