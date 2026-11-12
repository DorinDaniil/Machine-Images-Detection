import csv
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ImageDataset
from .model import Detector, score


def stats(logits: torch.Tensor, label: torch.Tensor) -> dict:
    """Loss; accuracy, precision, recall and FPR of "generated or real" at probability 0.5; and the share of the
    generated images whose model is named right."""
    prob, model = score(logits)
    pred, fake = prob > 0.5, label > 0
    tp, fp = (pred & fake).sum().item(), (pred & ~fake).sum().item()
    fn, tn = (~pred & fake).sum().item(), (~pred & ~fake).sum().item()
    return {"loss": F.cross_entropy(logits, label).item(), "acc": (tp + tn) / len(label),
            "prec": tp / max(tp + fp, 1), "rec": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1),
            "model_acc": (model[fake] + 1 == label[fake]).float().mean().item() if fake.any() else 0.0}


@torch.no_grad()
def validate(model: Detector, name: str, loader: DataLoader, device: str) -> dict:
    model.eval()
    logits, labels = [], []
    for image, label in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits.append(model(image.to(device), [name])[name].float().cpu())
        labels.append(label)
    return stats(torch.cat(logits), torch.cat(labels))


def trained(out, name: str, epochs: int) -> bool:
    """Expert `name` has all its epochs in <out>/metrics/<name>.csv."""
    path = Path(out) / "metrics" / f"{name}.csv"
    return path.exists() and len(path.read_text().splitlines()) == epochs + 1


def fmt(s: dict) -> str:
    return " ".join(f"{k} {v:.4f}" for k, v in s.items())


def train(model: Detector, name: str, reals: list, fakes: list, val: list, cfg: dict, resume: bool = False) -> None:
    """Trains expert `name`: reals (label 0) against its fakes (label j = the j-th model of the expert).
    The loss is the cross-entropy over [real, model 1, ..., model m]. It is the sum of two losses: "generated or
    real" for every image, and "which model" for the generated ones. The expert "all" is trained together with
    the trunk, any other expert on the frozen trunk. cfg = the `train` block of configs/train.yaml.

    An epoch: every fake of the expert once and as many reals, drawn anew every epoch. Then validation,
    metrics to <out>/metrics/<name>.csv, the training state to <out>/state.pt (resume=True goes on from it)
    and, if the validation loss is the lowest so far, the expert to <out>/experts/<name>.pt."""
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    out = Path(cfg["out"])
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    batch_size, epochs = cfg["batch_size"], cfg["epochs"]
    steps = 2 * len(fakes) // batch_size
    val_loader = DataLoader(ImageDataset(val, cfg["size"]), batch_size=batch_size, num_workers=cfg["num_workers"])

    model.to(device)
    expert = model.experts[name]
    model.trunk.requires_grad_(name == "all")
    body = [p for part in (model.trunk, expert.stages, expert.norm) for p in part.parameters() if p.requires_grad]
    head = list(expert.head.parameters())
    optimizer = torch.optim.AdamW([{"params": body, "lr": cfg["lr"]}, {"params": head, "lr": cfg["head_lr"]}],
                                  weight_decay=cfg["weight_decay"])
    total = epochs * steps
    warmup = min(500, total // 10 + 1)
    schedule = lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * step / total))  # noqa: E731
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    best, log, start = math.inf, [], 1
    if resume and (out / "state.pt").exists():
        state = torch.load(out / "state.pt", map_location="cpu", weights_only=False)
        if state["name"] == name:
            model.trunk.load_state_dict(state["trunk"])
            expert.load_state_dict(state["expert"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            best, log, start = state["best"], state["log"], state["epoch"]
            print(f"{name}: resuming from epoch {start}")

    for epoch in range(start, epochs + 1):
        model.train()
        rows = fakes + random.Random(epoch).choices(reals, k=len(fakes))  # half generated, half real
        loader = DataLoader(ImageDataset(rows, cfg["size"], train=True, seed=epoch), batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=cfg["num_workers"], pin_memory=device == "cuda")
        logits, labels = [], []
        for step, (image, label) in enumerate(loader, 1):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out_logits = model(image.to(device), [name])[name].float()
            loss = F.cross_entropy(out_logits, label.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(body + head, 1.0)
            optimizer.step()
            scheduler.step()
            logits.append(out_logits.detach().cpu())
            labels.append(label)
            if step % 50 == 0:
                print(f"{name} epoch {epoch} step {step}/{steps} loss {loss.item():.4f}")

        train_stats, val_stats = stats(torch.cat(logits), torch.cat(labels)), validate(model, name, val_loader, device)
        print(f"{name} epoch {epoch}: train {fmt(train_stats)}")
        print(f"{name} epoch {epoch}: val   {fmt(val_stats)}")
        log.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_stats.items()}})

        if val_stats["loss"] <= best:
            best = val_stats["loss"]
            model.save(out, name)
        torch.save({"name": name, "trunk": model.trunk.state_dict(), "expert": expert.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "best": best, "log": log, "epoch": epoch + 1}, out / "state.tmp")
        (out / "state.tmp").replace(out / "state.pt")  # a crash while saving leaves the previous state whole
        with open(out / "metrics" / f"{name}.csv", "w", newline="") as f:  # written last: all its rows = the expert is trained
            writer = csv.DictWriter(f, log[0].keys())
            writer.writeheader()
            writer.writerows(log)
    print(f"{name}: best validation loss {best:.4f}")
