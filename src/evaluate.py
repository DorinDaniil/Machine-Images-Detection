import math
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import Detector, score


@torch.no_grad()
def predict(model: Detector, dataset, device: str, batch_size: int = 32, workers: int = 4):
    """N images through the K experts of the model. prob [N, K]: the probability that the image is generated,
    by every expert. index [N, K]: the most likely model of every expert, as its place in expert.models."""
    model.eval().to(device)
    probs, indices = [], []
    for image, _ in DataLoader(dataset, batch_size=batch_size, num_workers=workers):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            answers = [score(logits) for logits in model(image.to(device)).values()]
        probs.append(torch.stack([p for p, _ in answers], dim=1).cpu())
        indices.append(torch.stack([i for _, i in answers], dim=1).cpu())
    return torch.cat(probs).double().numpy(), torch.cat(indices).numpy()


def thresholds(real: np.ndarray, alpha: float) -> np.ndarray:
    """One threshold per expert, from real images only: real [n, K] are the probabilities the K experts gave to n
    calibration reals. Every expert gets the same share of the false alarms: its threshold is the same quantile
    of its own probabilities on the reals, and the quantile is the lowest one at which at most alpha of the
    reals pass any of the K thresholds."""
    n = len(real)
    place = real.argsort(axis=0).argsort(axis=0)  # [n, K]: the place of every real among the reals, by every expert
    highest = np.sort(place.max(axis=1))          # the highest place of every real over the experts
    return np.sort(real, axis=0)[highest[min(math.ceil((n + 1) * (1 - alpha)), n) - 1]]


def verdict(prob: np.ndarray, thr: np.ndarray):
    """generated [N]: some expert is above its threshold. winner [N]: the expert furthest above its own."""
    return (prob > thr).any(axis=1), (prob - thr).argmax(axis=1)


def report(model: Detector, thr: np.ndarray, reals: dict, fakes: list) -> str:
    """reals: {set: prob [n, K]} -> measured FPR. fakes: [(group, prob [n, K], index [n, K])] -> TPR, the expert
    that answers for the group most often, the model it names most often and, for a generator seen in training,
    how often the model is named exactly."""
    names, seen = list(model.experts), set(model.experts["all"].models)
    lines = ["| real set | n | FPR |", "|---|---|---|"]
    lines += [f"| {name} | {len(prob)} | {verdict(prob, thr)[0].mean():.3f} |" for name, prob in reals.items()]
    lines += ["", "| fake group | n | TPR | expert | named as | exact model |", "|---|---|---|---|---|---|"]
    for group, prob, index in fakes:
        hit, winner = verdict(prob, thr)
        named = [model.experts[names[w]].models[index[i, w]] for i, w in enumerate(winner)]
        expert, model_name = Counter(names[w] for w in winner).most_common(1)[0][0], Counter(named).most_common(1)[0][0]
        exact = f"{np.mean([name == group for name in named]):.3f}" if group in seen else ""
        lines.append(f"| {group}{' (in train)' if group in seen else ''} | {len(prob)} | {hit.mean():.3f} | {expert} | "
                     f"{model_name} | {exact} |")
    return "\n".join(lines)
