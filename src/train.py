"""Training loop for TraceDetector."""

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data import AlignedPairsDataset, ManifestDataset, filter_manifest, read_manifest
from .evaluate import pixel_scores
from .losses import trace_loss
from .model import TraceDetector


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> TraceDetector:
    return TraceDetector.from_config(cfg["model"])


def build_loaders(cfg: dict, model: TraceDetector):
    d, t = dict(cfg["data"]), cfg["training"]
    d["crop_size"] = d["crop_size"] // model.patch_size * model.patch_size  # grid must tile the crop exactly
    train_set = AlignedPairsDataset(d["pairs_csv"], d, split="train", seed=cfg.get("seed", 0))

    if d.get("val_manifest"):
        df = filter_manifest(read_manifest(d["val_manifest"], d.get("data_root")), split="val")
        val_set = ManifestDataset(df, model.patch_size, fixed_size=d["crop_size"])
    else:
        val_set = AlignedPairsDataset(d["pairs_csv"], d, split="val", seed=cfg.get("seed", 0) + 1)

    train_loader = DataLoader(
        train_set, batch_size=t["batch_size"], shuffle=True, drop_last=True,
        num_workers=d.get("num_workers", 0), pin_memory=torch.cuda.is_available(), persistent_workers=d.get("num_workers", 0) > 0,
    )
    val_loader = DataLoader(val_set, batch_size=t["batch_size"], shuffle=False, num_workers=d.get("num_workers", 0))
    return train_loader, val_loader


def make_optimizer(model: TraceDetector, cfg: dict) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])


def make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """Linear warm-up, then cosine decay to 5 % of the base learning rate."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def validate(model: TraceDetector, loader: DataLoader, device: torch.device, amp: bool) -> dict:
    model.eval()
    labels, scores, f1s = [], [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = (batch["target"] if "target" in batch else batch["mask"]).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            out = model(image)
        scores.append(torch.sigmoid(out["image_logit"].float()).cpu())
        labels.append(batch["label"].cpu())
        is_fake = batch["label"] >= 0.5  # pixel F1 at threshold 0.5 is meaningful only for alpha >= 0.5
        if "has_mask" in batch:
            is_fake &= batch["has_mask"]
        if bool(is_fake.any()):
            f1, _ = pixel_scores(out["patch_logits"].float()[is_fake.to(device)], target[is_fake.to(device)])
            f1s.append(f1.cpu())

    labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()
    binary = (labels > 0.5).astype(int)
    auroc = roc_auc_score(binary, scores) if 0 < binary.sum() < len(binary) else float("nan")
    return {"auroc": auroc, "pixel_f1": float(torch.cat(f1s).mean()) if f1s else float("nan")}


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, cfg: dict, best: float) -> None:
    torch.save(
        {
            "model_cfg": cfg["model"],
            "state_dict": model.trainable_state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_auroc": best,
        },
        path,
    )


def train_model(model: TraceDetector, train_loader: DataLoader, val_loader: DataLoader, cfg: dict, resume: bool = False) -> Path:
    t = cfg["training"]
    device = torch.device(t["device"] if torch.cuda.is_available() or t["device"] == "cpu" else "cpu")
    amp = bool(t.get("amp", True)) and device.type == "cuda"
    ckpt_dir, log_dir = Path(t["checkpoint_dir"]), Path(t["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    model.to(device)
    optimizer = make_optimizer(model, cfg)
    total_steps = t["epochs"] * len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps, t.get("warmup_steps", 0))

    start_epoch, best_auroc = 0, -1.0
    last = ckpt_dir / "last.pt"
    if resume and last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_trainable_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_auroc = state["epoch"], state["best_auroc"]
        print(f"Resumed from epoch {start_epoch}")

    step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, t["epochs"]):
        model.train()
        model.backbone.eval()  # frozen blocks keep eval behaviour; trainable blocks have no dropout anyway
        train_loader.dataset.set_epoch(epoch)
        running, t0 = [], time.time()

        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(image)
                losses = trace_loss(out["patch_logits"].float(), out["image_logit"].float(), target, cfg["loss"])

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            running.append(losses["total"].item())
            if step % t.get("log_every", 50) == 0:
                for name, value in losses.items():
                    writer.add_scalar(f"train/{name}", value.item(), step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                print(f"epoch {epoch + 1} step {step} loss {np.mean(running[-t.get('log_every', 50):]):.4f}")

        metrics = validate(model, val_loader, device, amp)
        for name, value in metrics.items():
            writer.add_scalar(f"val/{name}", value, epoch + 1)
        print(
            f"epoch {epoch + 1}/{t['epochs']}  train loss {np.mean(running):.4f}  "
            f"val auroc {metrics['auroc']:.4f}  val pixel f1 {metrics['pixel_f1']:.4f}  "
            f"({time.time() - t0:.0f}s)"
        )

        if metrics["auroc"] >= best_auroc or math.isnan(metrics["auroc"]):
            best_auroc = metrics["auroc"]
            torch.save({"model_cfg": cfg["model"], "state_dict": model.trainable_state_dict()}, ckpt_dir / "best.pt")
        save_checkpoint(last, model, optimizer, scheduler, epoch + 1, cfg, best_auroc)

    writer.close()
    return ckpt_dir / "best.pt"
