import math
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import Detector


def groups(models: list, family_of: dict) -> dict:
    """{family: the columns of its classes} for the classes of the general expert. family_of: generator -> family
    (model.families of the config, turned over). A class that is not there stays on its own: a generator without
    a family, or a class that is a family already."""
    cols = {}
    for j, name in enumerate(models):
        cols.setdefault(family_of.get(name, name), []).append(j)
    return cols


@torch.no_grad()
def predict(model: Detector, dataset, device: str, batch_size: int = 32, workers: int = 4):
    """N images through the general expert. odds [N]: the log-odds that the image is generated, log(p / (1 - p)).
    share [N, m]: the probabilities of its m classes for a generated image, every row sums to 1."""
    model.eval().to(device)
    odds, share = [], []
    for step, (image, _) in enumerate(DataLoader(dataset, batch_size=batch_size, num_workers=workers), 1):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(image.to(device), ["all"])["all"].float()
        odds.append((logits[:, 1:].logsumexp(dim=1) - logits[:, 0]).cpu())
        share.append(logits[:, 1:].softmax(dim=1).cpu())
        if step % 250 == 0:
            print(f"{step * batch_size}/{len(dataset)} images")
    return torch.cat(odds).double().numpy(), torch.cat(share).double().numpy()


def threshold(real: np.ndarray, alpha: float) -> float:
    """From real images only: the lowest of the log-odds of n calibration reals that at most alpha of them exceed
    (n + 1 as in conformal prediction: a new real exceeds it with probability <= alpha)."""
    n = len(real)
    return float(np.sort(real)[min(math.ceil((n + 1) * (1 - alpha)), n) - 1])


def name_family(share: np.ndarray, cols: dict) -> np.ndarray:
    """[N] the most likely family of every image: the probabilities of the classes of a family are summed."""
    return np.array(list(cols))[np.stack([share[:, c].sum(axis=1) for c in cols.values()], axis=1).argmax(axis=1)]


def report(hit: np.ndarray, named: np.ndarray, group: np.ndarray, truth: np.ndarray) -> str:
    """Tables for N test images. hit [N]: called generated. named [N]: the family the model names.
    group [N]: the real source or the generator. truth [N]: "" for a real, the family of a generated image,
    "?" when the model was not trained on its generator."""
    lines = ["| real source | n | FPR |", "|---|---|---|"]
    lines += [f"| {g} | {(group == g).sum()} | {hit[group == g].mean():.3f} |" for g in sorted(set(group[truth == ""]))]
    lines += ["", "| generator | family | n | TPR | named as | right family |", "|---|---|---|---|---|---|"]
    for g in sorted(set(group[truth != ""])):
        at = group == g
        right = "" if truth[at][0] == "?" else f"{(named[at] == truth[at]).mean():.3f}"
        lines.append(f"| {g} | {truth[at][0]} | {at.sum()} | {hit[at].mean():.3f} | "
                     f"{Counter(named[at]).most_common(1)[0][0]} | {right} |")
    lines += ["", "| family | n | TPR | right family | most often taken for |", "|---|---|---|---|---|"]
    for f in sorted(set(truth) - {"", "?"}):
        at = truth == f
        taken = ", ".join(f"{name} {n / at.sum():.3f}" for name, n in Counter(named[at & (named != f)]).most_common(2))
        lines.append(f"| {f} | {at.sum()} | {hit[at].mean():.3f} | {(named[at] == f).mean():.3f} | {taken} |")
    return "\n".join(lines)
