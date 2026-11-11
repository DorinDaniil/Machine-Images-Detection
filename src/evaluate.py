import math

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .model import Detector, score


@torch.no_grad()
def predict(model: Detector, dataset, device: str, batch_size: int = 32, workers: int = 4):
    """Image scores and, for every image, the index of the head that gave the score."""
    model.eval().to(device)
    scores, heads = [], []
    for image, _ in DataLoader(dataset, batch_size=batch_size, num_workers=workers):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            s, head = score(model(image.to(device)))
        scores.append(s.cpu())
        heads.append(head.cpu())
    return torch.cat(scores).double().numpy(), torch.cat(heads).numpy()


def threshold_at(calib_real_scores: np.ndarray, alpha: float) -> float:
    """Conformal threshold from real images only: a new real image exceeds it with probability <= alpha."""
    s = np.sort(calib_real_scores)
    k = math.ceil((len(s) + 1) * (1 - alpha)) - 1
    return float(s[min(k, len(s) - 1)])


def report(threshold: float, reals: dict, fakes: list, families: list) -> str:
    """reals: {name: scores} -> measured FPR at the threshold.
    fakes: [(name, real scores of the same source, fake scores, heads)] -> AUROC, TPR and the head that
    answers for the group most often."""
    lines = ["| real set | n | FPR at threshold |", "|---|---|---|"]
    lines += [f"| {name} | {len(s)} | {(s > threshold).mean():.3f} |" for name, s in reals.items()]
    lines += ["", "| fake group | n | AUROC vs its reals | TPR at threshold | head |", "|---|---|---|---|---|"]
    for name, real, fake, heads in fakes:
        auroc = roc_auc_score(np.r_[np.zeros(len(real)), np.ones(len(fake))], np.r_[real, fake])
        head = families[np.bincount(heads).argmax()]
        lines.append(f"| {name} | {len(fake)} | {auroc:.3f} | {(fake > threshold).mean():.3f} | {head} |")
    return "\n".join(lines)
