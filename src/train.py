import csv
import math
import random
from collections import Counter
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ImageDataset
from .model import Detector, score

BANK = 4096  # image vectors of the last steps kept for the contrastive loss: a batch alone has too few pairs


def contrastive(z: torch.Tensor, y: torch.Tensor, bank_z: torch.Tensor, bank_y: torch.Tensor, tau: float = 0.1):
    """Supervised InfoNCE. z [B, D]: unit vectors of the batch, y [B]: their classes. bank_z [K, D], bank_y [K]:
    the vectors they are compared with, the batch itself first. For an image the images of its class are
    positives, all the others negatives; its loss is the mean over its positives p of
        -log( exp(z·p / tau) / sum over the bank of exp(z·k / tau) )
    so images of one class are pulled together and images of different classes are pushed apart."""
    own = torch.eye(len(z), len(bank_z), dtype=torch.bool, device=z.device)  # an image is not compared with itself
    log_p = (z @ bank_z.T / tau).masked_fill(own, -torch.inf).log_softmax(dim=1)  # [B, K]
    pos = (y[:, None] == bank_y[None]) & ~own
    n = pos.sum(dim=1)
    loss = -log_p.masked_fill(~pos, 0).sum(dim=1) / n.clamp(min=1)  # 0 for an image without positives
    return loss.sum() / (n > 0).sum().clamp(min=1)


def stats(logits: torch.Tensor, label: torch.Tensor) -> dict:
    """Loss; accuracy, precision, recall and FPR of "generated or real" at probability 0.5; and the share of the
    generated images whose family is named right."""
    prob, family = score(logits)
    pred, fake = prob > 0.5, label > 0
    tp, fp = (pred & fake).sum().item(), (pred & ~fake).sum().item()
    fn, tn = (~pred & fake).sum().item(), (~pred & ~fake).sum().item()
    return {"loss": F.cross_entropy(logits, label).item(), "acc": (tp + tn) / len(label),
            "prec": tp / max(tp + fp, 1), "rec": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1),
            "family_acc": (family[fake] + 1 == label[fake]).float().mean().item() if fake.any() else 0.0}


@torch.no_grad()
def validate(model: Detector, loader: DataLoader, device: str) -> dict:
    model.eval()
    logits, labels = [], []
    for image, label in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits.append(model(image.to(device)).float().cpu())
        labels.append(label.clone())  # a copy: what the loader gives holds shared memory and an open file each
    return stats(torch.cat(logits), torch.cat(labels))


def fmt(s: dict) -> str:
    return " ".join(f"{k} {v:.4f}" for k, v in s.items())


def train(model: Detector, reals: list, fakes: list, val: list, cfg: dict, resume: bool = False) -> None:
    """Trains the model: reals (label 0) against generated images (label j = the j-th class of the model).
    cfg = the `train` block of configs/train.yaml.

    Loss: the cross-entropy over [real, family 1, ..., family m]. It is the sum of two losses: "generated or real"
    for every image, and "which family" for the generated ones. With cfg["contrast"] > 0 the supervised InfoNCE
    of the image vectors is added with this weight (contrastive).

    An epoch: as many generated images as the training part has, and as many reals, drawn anew every epoch.
    With cfg["balance"] = 0 these are all the generated images, once each. Above 0 they are drawn so that an
    image of a generator with n images comes n ** -balance times as often: small generators are seen more.

    After an epoch: validation, a row in <out>/metrics.csv and, if the validation loss is the lowest so far, the
    model to <out>/model.pt. The training state goes to <out>/state.pt after every epoch and every
    cfg["save_every"] steps inside it; resume=True goes on from it, the rest of an unfinished epoch drawn anew."""
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    batch_size, epochs = cfg["batch_size"], cfg["epochs"]
    steps = 2 * len(fakes) // batch_size
    val_loader = DataLoader(ImageDataset(val, cfg["size"]), batch_size=batch_size, num_workers=cfg["num_workers"])
    count = Counter(r["generator"] for r in fakes)
    weights = [count[r["generator"]] ** -cfg["balance"] for r in fakes]

    model.to(device)
    body, head = list(model.net.parameters()), list(model.head.parameters())
    optimizer = torch.optim.AdamW([{"params": body, "lr": cfg["lr"]}, {"params": head, "lr": cfg["head_lr"]}],
                                  weight_decay=cfg["weight_decay"])
    total = epochs * steps
    warmup = min(500, total // 10 + 1)
    schedule = lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * step / total))  # noqa: E731
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    best, log, start, done = math.inf, [], 1, 0  # done: steps of epoch `start` that are behind
    if resume and (out / "state.pt").exists():
        state = torch.load(out / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best, log, start, done = state["best"], state["log"], state["epoch"], state["step"]
        print(f"resuming from epoch {start}, step {done + 1}" if start <= epochs else "all epochs are done")

    def save_state(epoch, step):  # the next step to make is `step` + 1 of `epoch`
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "best": best, "log": log, "epoch": epoch, "step": step}, out / "state.tmp")
        (out / "state.tmp").replace(out / "state.pt")  # a crash while saving leaves the previous state whole

    bank_z = torch.empty(0, model.head.in_features, device=device)  # empty again after a resume
    bank_y = torch.empty(0, dtype=torch.long, device=device)
    for epoch in range(start, epochs + 1):
        model.train()
        rng = random.Random(epoch)
        rows = rng.choices(reals, k=len(fakes))  # half real, half generated
        rows += rng.choices(fakes, weights, k=len(fakes)) if cfg["balance"] else fakes
        loader = DataLoader(ImageDataset(rows, cfg["size"], train=True, seed=epoch), batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=cfg["num_workers"], pin_memory=device == "cuda")
        logits, labels = [], []
        for step, (image, label) in enumerate(islice(loader, steps - done), done + 1):
            label = label.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                vector = model.features(image.to(device))
                out_logits = model.head(vector).float()
            loss = F.cross_entropy(out_logits, label)
            if cfg["contrast"]:
                z = F.normalize(vector.float(), dim=1)
                bank_z, bank_y = torch.cat([z, bank_z]), torch.cat([label, bank_y])
                loss = loss + cfg["contrast"] * contrastive(z, label, bank_z, bank_y)
                bank_z, bank_y = bank_z.detach()[:BANK], bank_y[:BANK]  # the oldest vectors drop out
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(body + head, 1.0)
            optimizer.step()
            scheduler.step()
            logits.append(out_logits.detach().cpu())
            labels.append(label.cpu())  # a copy: what the loader gives sits in pinned memory, one allocation per step
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{steps} loss {loss.item():.4f}")
            if step % cfg["save_every"] == 0 and step < steps:
                save_state(epoch, step)
        done = 0

        train_stats, val_stats = stats(torch.cat(logits), torch.cat(labels)), validate(model, val_loader, device)
        print(f"epoch {epoch}: train {fmt(train_stats)}")
        print(f"epoch {epoch}: val   {fmt(val_stats)}")
        log.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_stats.items()}})

        if val_stats["loss"] <= best:
            best = val_stats["loss"]
            model.save(out / "model.pt")
        save_state(epoch + 1, 0)
        with open(out / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, log[0].keys())
            writer.writeheader()
            writer.writerows(log)
    print(f"best validation loss {best:.4f}")
