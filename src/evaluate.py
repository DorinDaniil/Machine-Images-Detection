import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import Detector


@torch.no_grad()
def predict(model: Detector, dataset, device: str, batch_size: int = 32, workers: int = 4):
    """N images through the model. odds [N]: the log-odds that the image is generated, log(p / (1 - p)) with
    p = 1 - p(real); above 0 means p > 0.5. index [N]: the most likely of the model's classes if it is generated."""
    model.eval().to(device)
    odds, index = [], []
    for step, (image, _) in enumerate(DataLoader(dataset, batch_size=batch_size, num_workers=workers), 1):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(image.to(device)).float()
        odds.append((logits[:, 1:].logsumexp(dim=1) - logits[:, 0]).cpu())
        index.append(logits[:, 1:].argmax(dim=1).cpu())
        if step % 250 == 0:
            print(f"{step * batch_size}/{len(dataset)} images")
    return torch.cat(odds).double().numpy(), torch.cat(index).numpy()


def threshold(real: np.ndarray, alpha: float) -> float:
    """From real images only: the lowest of the log-odds of n calibration reals that at most alpha of them exceed
    (n + 1 as in conformal prediction: a new real exceeds it with probability <= alpha)."""
    n = len(real)
    return float(np.sort(real)[min(math.ceil((n + 1) * (1 - alpha)), n) - 1])


def calibrate(model: Detector, real: np.ndarray, alpha: float) -> float:
    """real [N]: the log-odds of held-out real images. Adds their threshold to the bias of the class "real": after
    it at most alpha of these reals have p(generated) > 0.5, so the model needs no threshold beside it.
    Returns the shift."""
    shift = threshold(real, alpha)
    with torch.no_grad():
        model.head.bias[0] += shift
    return shift


def report(hit: np.ndarray, named: np.ndarray, group: np.ndarray, truth: np.ndarray):
    """Tables for N test images. hit [N]: called generated. named [N]: the family the model names.
    group [N]: the real source or the generator. truth [N]: "" for a real, the family of a generated image,
    "?" when the model has no class for its generator.
    Returns the tables as text and their first numbers: {"FPR <real source>": .., "TPR <generator>": ..}."""
    numbers = {}
    lines = ["| real source | n | FPR |", "|---|---|---|"]
    for g in sorted(set(group[truth == ""])):
        numbers[f"FPR {g}"] = float(hit[group == g].mean())
        lines.append(f"| {g} | {(group == g).sum()} | {numbers[f'FPR {g}']:.3f} |")
    lines += ["", "| generator | family | n | TPR | named as | right family |", "|---|---|---|---|---|---|"]
    for g in sorted(set(group[truth != ""])):
        at = group == g
        numbers[f"TPR {g}"] = float(hit[at].mean())
        right = "" if truth[at][0] == "?" else f"{(named[at] == truth[at]).mean():.3f}"
        lines.append(f"| {g} | {truth[at][0]} | {at.sum()} | {numbers[f'TPR {g}']:.3f} | "
                     f"{Counter(named[at]).most_common(1)[0][0]} | {right} |")
    lines += ["", "| family | n | TPR | right family | most often taken for |", "|---|---|---|---|---|"]
    for f in sorted(set(truth) - {"", "?"}):
        at = truth == f
        taken = ", ".join(f"{name} {n / at.sum():.3f}" for name, n in Counter(named[at & (named != f)]).most_common(2))
        lines.append(f"| {f} | {at.sum()} | {hit[at].mean():.3f} | {(named[at] == f).mean():.3f} | {taken} |")
    return "\n".join(lines), numbers


def compare(folder) -> str:
    """One table for the tested runs of `folder`: a column per run, a row per number of its report.json."""
    runs = {p.parent.name: json.loads(p.read_text()) for p in sorted(Path(folder).glob("*/report.json"))}
    keys = list(dict.fromkeys(k for numbers in runs.values() for k in numbers))
    lines = ["| | " + " | ".join(runs) + " |", "|---" * (len(runs) + 1) + "|"]
    for k in keys:
        lines.append("| " + " | ".join([k, *(f"{n[k]:.3f}" if k in n else "" for n in runs.values())]) + " |")
    return "\n".join(lines)
