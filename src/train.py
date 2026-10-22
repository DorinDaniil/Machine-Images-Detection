import csv
import math
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .losses import patch_loss
from .model import Detector


class BinaryMetrics:
    """Epoch-level accuracy, precision, recall, F1 and FPR: TP / FP / FN / TN are accumulated over all
    images of the epoch at a fixed probability threshold, so batches of different size weigh correctly."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        pred = torch.sigmoid(logits.detach().float()).cpu() >= self.threshold
        true = labels.detach().cpu() > 0.5
        self.tp += int((pred & true).sum())
        self.fp += int((pred & ~true).sum())
        self.fn += int((~pred & true).sum())
        self.tn += int((~pred & ~true).sum())

    @staticmethod
    def _ratio(a: int, b: int) -> float:
        return a / b if b > 0 else 0.0

    def summary(self) -> dict:
        p, r = self._ratio(self.tp, self.tp + self.fp), self._ratio(self.tp, self.tp + self.fn)
        return {"acc": self._ratio(self.tp + self.tn, self.tp + self.fp + self.fn + self.tn), "prec": p, "rec": r,
                "f1": self._ratio(2 * p * r, p + r), "fpr": self._ratio(self.fp, self.fp + self.tn)}


@torch.no_grad()
def validate(model: Detector, loader: DataLoader, device: str, threshold: float) -> dict:
    """Loss averaged over images and the binary metrics of the whole validation set."""
    model.eval()
    metrics, loss_sum, n = BinaryMetrics(threshold), 0.0, 0
    for batch in loader:
        image, target = batch["image"].to(device), batch["target"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            patch_logits, image_logit = model(image)
        loss_sum += patch_loss(patch_logits.float(), image_logit.float(), target).item() * len(image)
        n += len(image)
        metrics.update(image_logit, batch["label"])
    return {"loss": loss_sum / max(n, 1), **metrics.summary()}


def balanced_weights(rows: list[dict], floor: int = 5000) -> list[float]:
    """Sampling weights: half of the draws are reals, the other half is spread over generators equally,
    except that a generator with fewer than `floor` images weighs in proportion to its size. So no image
    is drawn more often than an image of a generator with `floor` images."""
    counts = Counter(r["generator"] for r in rows)
    mass = {g: min(n, floor) for g, n in counts.items() if g}
    total = sum(mass.values())
    return [0.5 / counts[""] if not r["generator"] else 0.5 * mass[r["generator"]] / (total * counts[r["generator"]])
            for r in rows]


def fmt(stats: dict) -> str:
    return " ".join(f"{k} {v:.4f}" for k, v in stats.items())


def train(model: Detector, train_set, val_set, cfg: dict) -> Path:
    """cfg = the `train` block of configs/train.yaml. Metrics of every epoch go to <out>/metrics.csv;
    the checkpoint with the best validation F1 is kept as best.pt. Returns its path."""
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    threshold = cfg.get("threshold", 0.5)
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    weights = balanced_weights(train_set.rows, cfg.get("balance_floor", 5000))
    sampler = WeightedRandomSampler(weights, num_samples=len(train_set), replacement=True)
    loader = DataLoader(train_set, batch_size=cfg["batch_size"], sampler=sampler, drop_last=True,
                        num_workers=cfg["num_workers"], pin_memory=device == "cuda")
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])

    model.to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = cfg["epochs"] * len(loader)
    warmup = min(200, total // 10 + 1)
    schedule = lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * step / total))  # noqa: E731
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    best, log = -1.0, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_set.epoch = epoch
        metrics, loss_sum, n = BinaryMetrics(threshold), 0.0, 0
        for step, batch in enumerate(loader):
            image, target = batch["image"].to(device), batch["target"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                patch_logits, image_logit = model(image)
            loss = patch_loss(patch_logits.float(), image_logit.float(), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(image)
            n += len(image)
            metrics.update(image_logit, batch["label"])
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {loss.item():.4f}")

        train_stats = {"loss": loss_sum / max(n, 1), **metrics.summary()}
        val_stats = validate(model, val_loader, device, threshold)
        print(f"epoch {epoch}: train {fmt(train_stats)}\n{' ' * (len(str(epoch)) + 7)} val   {fmt(val_stats)}")
        log.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_stats.items()}})
        with open(out / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, log[0].keys())
            writer.writeheader()
            writer.writerows(log)

        model.save(out / "last.pt")
        if val_stats["f1"] >= best:
            best = val_stats["f1"]
            model.save(out / "best.pt")
    return out / "best.pt"
