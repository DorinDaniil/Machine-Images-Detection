import csv
import math
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .model import Detector, family_loss, score


def stats(logits: torch.Tensor, label: torch.Tensor) -> dict:
    """Loss, and accuracy, precision, recall, F1, FPR of "generated or real" at image score 0.5."""
    pred, fake = score(logits)[0] > 0.5, label > 0
    tp, fp = (pred & fake).sum().item(), (pred & ~fake).sum().item()
    fn, tn = (~pred & fake).sum().item(), (~pred & ~fake).sum().item()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"loss": family_loss(logits, label).item(), "acc": (tp + tn) / len(label), "prec": prec, "rec": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9), "fpr": fp / max(fp + tn, 1)}


@torch.no_grad()
def validate(model: Detector, loader: DataLoader, device: str) -> dict:
    model.eval()
    logits, labels = [], []
    for image, label in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits.append(model(image.to(device)).float().cpu())
        labels.append(label)
    return stats(torch.cat(logits), torch.cat(labels))


def balanced_weights(rows: list[dict], floor: int = 10000) -> list[float]:
    """Sampling weights by label (0 real, k = family k): half of the draws are reals, the other half is spread
    over the families equally, so every head gets as many images of its family as the others. A family with
    fewer than `floor` images weighs in proportion to its size: its images are not repeated too often."""
    counts = Counter(r["label"] for r in rows)
    mass = {k: min(n, floor) for k, n in counts.items() if k}
    total = sum(mass.values())
    return [0.5 / counts[0] if not r["label"] else 0.5 * mass[r["label"]] / (total * counts[r["label"]]) for r in rows]


def fmt(s: dict) -> str:
    return " ".join(f"{k} {v:.4f}" for k, v in s.items())


def train(model: Detector, train_set, val_set, cfg: dict, resume: bool = False) -> Path:
    """cfg = the `train` block of configs/train.yaml. An epoch is `samples_per_epoch` balanced draws, then
    validation. Every epoch: metrics to <out>/metrics.csv, the whole training state to <out>/state.pt
    (resume=True goes on from it), and best.pt if the validation loss is the lowest so far. Returns best.pt."""
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    batch_size = cfg["batch_size"]
    steps = (cfg.get("samples_per_epoch") or len(train_set)) // batch_size
    weights = balanced_weights(train_set.rows, cfg.get("balance_floor", 10000))
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=cfg["num_workers"])

    model.to(device)
    blocks = [p for p in model.vit.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": blocks, "lr": cfg["lr"]},
                                   {"params": model.head.parameters(), "lr": cfg["head_lr"]}], weight_decay=cfg["weight_decay"])
    total = cfg["epochs"] * steps
    warmup = min(500, total // 10 + 1)
    schedule = lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * step / total))  # noqa: E731
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    best, log, start = math.inf, [], 1
    if resume and (out / "state.pt").exists():
        state = torch.load(out / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best, log, start = state["best"], state["log"], state["epoch"]
        print(f"resuming from {out / 'state.pt'}: epoch {start}")

    for epoch in range(start, cfg["epochs"] + 1):
        model.train()
        train_set.epoch = epoch  # new random views every epoch
        sampler = WeightedRandomSampler(weights, num_samples=steps * batch_size, replacement=True)
        loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=cfg["num_workers"],
                            pin_memory=device == "cuda")
        logits, labels = [], []
        for step, (image, label) in enumerate(loader, 1):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out_logits = model(image.to(device)).float()
            loss = family_loss(out_logits, label.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            logits.append(out_logits.detach().cpu())
            labels.append(label)
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{steps} loss {loss.item():.4f}")

        train_stats, val_stats = stats(torch.cat(logits), torch.cat(labels)), validate(model, val_loader, device)
        print(f"epoch {epoch}: train {fmt(train_stats)}\n{' ' * (len(str(epoch)) + 7)} val   {fmt(val_stats)}")
        log.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_stats.items()}})
        with open(out / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, log[0].keys())
            writer.writeheader()
            writer.writerows(log)

        if val_stats["loss"] <= best:
            best = val_stats["loss"]
            model.save(out / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "best": best, "log": log, "epoch": epoch + 1}, out / "state.tmp")
        (out / "state.tmp").replace(out / "state.pt")  # a crash while saving leaves the previous state whole
    print(f"best.pt: validation loss {best:.4f}")
    return out / "best.pt"
