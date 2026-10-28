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
        metrics.update(image_logit, target.flatten(1).amax(1))  # label of the view, as in the loss
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


def train(model: Detector, train_set, val_set, cfg: dict, resume: bool = False) -> Path:
    """cfg = the `train` block of configs/train.yaml. Images differ in size, so they go through the model one
    at a time and the gradient of `batch_size` images is accumulated before each optimizer step (a remainder
    smaller than `batch_size` at the end of an epoch is dropped). An epoch is `samples_per_epoch` balanced draws
    (default: the size of the train set). Metrics of every epoch go to <out>/metrics.csv; best.pt is the epoch
    with the lowest validation loss: it does not depend on a threshold, and the test sets its own one.

    Every `save_every` optimizer steps and at the end of every epoch the whole training state (head, optimizer,
    schedule, progress in the epoch, metrics so far) goes to <out>/state.pt. With resume=True training goes on
    from it: the rest of the interrupted epoch is drawn anew from the same balanced sampler.
    Returns the path of best.pt."""
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    threshold, accum, save_every = cfg.get("threshold", 0.5), cfg["batch_size"], cfg.get("save_every", 1000)
    samples = cfg.get("samples_per_epoch") or len(train_set)
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    weights = balanced_weights(train_set.rows, cfg.get("balance_floor", 5000))
    val_loader = DataLoader(val_set, batch_size=1, num_workers=cfg["num_workers"])

    model.to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = max(1, samples // accum)
    total = cfg["epochs"] * steps
    warmup = min(200, total // 10 + 1)
    schedule = lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * step / total))  # noqa: E731
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    state_path = out / "state.pt"
    best, best_epoch, log, start, progress = math.inf, 0, [], 1, None
    if resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best, best_epoch, log, start, progress = (state[k] for k in ("best", "best_epoch", "log", "epoch", "progress"))
        print(f"resuming from {state_path}: epoch {start}, {progress['n'] if progress else 0} of {samples} images done")
    elif state_path.exists():
        print(f"{state_path} exists but resume is off: training starts from scratch and will overwrite it")

    def save_state(epoch: int, progress) -> None:
        tmp = out / "state.tmp"
        torch.save({"head": model.head.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "best": best, "best_epoch": best_epoch, "log": log, "epoch": epoch, "progress": progress}, tmp)
        tmp.replace(state_path)  # a crash while saving leaves the previous state whole

    for epoch in range(start, cfg["epochs"] + 1):
        model.train()
        train_set.epoch = epoch
        p = progress or {"n": 0, "step": 0, "loss_sum": 0.0, "counts": [0, 0, 0, 0]}
        progress = None
        metrics, loss_sum, n, step = BinaryMetrics(threshold), p["loss_sum"], p["n"], p["step"]
        metrics.tp, metrics.fp, metrics.fn, metrics.tn = p["counts"]
        left = samples - n
        loader = DataLoader(train_set, batch_size=1, num_workers=cfg["num_workers"], pin_memory=device == "cuda",
                            sampler=WeightedRandomSampler(weights, num_samples=left, replacement=True)) if left > 0 else []
        optimizer.zero_grad()
        for batch in loader:
            image, target = batch["image"].to(device), batch["target"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                patch_logits, image_logit = model(image)
            loss = patch_loss(patch_logits.float(), image_logit.float(), target)
            (loss / accum).backward()
            loss_sum += loss.item()
            n += 1
            metrics.update(image_logit, target.flatten(1).amax(1))  # a crop can cut an edit out
            if n % accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % 50 == 0:
                    print(f"epoch {epoch} step {step}/{steps} mean loss {loss_sum / n:.4f}")
                if step % save_every == 0:
                    save_state(epoch, {"n": n, "step": step, "loss_sum": loss_sum,
                                       "counts": [metrics.tp, metrics.fp, metrics.fn, metrics.tn]})

        train_stats = {"loss": loss_sum / max(n, 1), **metrics.summary()}
        val_stats = validate(model, val_loader, device, threshold)
        print(f"epoch {epoch}: train {fmt(train_stats)}\n{' ' * (len(str(epoch)) + 7)} val   {fmt(val_stats)}")
        log.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_stats.items()}})
        with open(out / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, log[0].keys())
            writer.writeheader()
            writer.writerows(log)

        model.save(out / "last.pt")
        if val_stats["loss"] <= best:
            best, best_epoch = val_stats["loss"], epoch
            model.save(out / "best.pt")
        save_state(epoch + 1, None)  # this epoch is done
    print(f"best.pt: epoch {best_epoch}, validation loss {best:.4f}")
    return out / "best.pt"
