import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .losses import trace_loss
from .model import TraceDetector


@torch.no_grad()
def val_auroc(model: TraceDetector, loader: DataLoader, device: str) -> float:
    model.eval()
    labels, scores = [], []
    for batch in loader:
        _, image_logit = model(batch["image"].to(device))
        scores.append(image_logit.float().cpu())
        labels.append(batch["label"] > 0)
    labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()
    return roc_auc_score(labels, scores) if 0 < labels.sum() < len(labels) else float("nan")


def train(model: TraceDetector, train_set, val_set, cfg: dict) -> Path:
    """cfg = the `train` block of configs/train.yaml. Returns the path of the best checkpoint."""
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                        num_workers=cfg["num_workers"], pin_memory=device == "cuda")
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])

    model.to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = cfg["epochs"] * len(loader)
    warmup = min(200, total // 10 + 1)
    schedule = lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * step / total))  # noqa: E731
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    best = -1.0
    for epoch in range(cfg["epochs"]):
        model.train()
        train_set.epoch = epoch
        losses = []
        for step, batch in enumerate(loader):
            image, target = batch["image"].to(device), batch["target"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                patch_logits, image_logit = model(image)
            loss = trace_loss(patch_logits.float(), image_logit.float(), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            if step % 50 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(loader)} loss {np.mean(losses[-50:]):.4f}")

        auc = val_auroc(model, val_loader, device)
        print(f"epoch {epoch + 1}: train loss {np.mean(losses):.4f}, val AUROC {auc:.4f}")
        model.save(out / "last.pt")
        if auc >= best or math.isnan(auc):
            best = auc
            model.save(out / "best.pt")
    return out / "best.pt"
