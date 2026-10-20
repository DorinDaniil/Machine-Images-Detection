"""Score a manifest and summarise metrics per source, generator and content type."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from .data import ManifestDataset
from .dino_trace import TraceDetector

RECALL_LEVELS = (0.3, 0.5, 0.7)


def pixel_scores(patch_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    """Pixel F1 and IoU of the up-sampled patch map against a target map, per sample."""
    probs = torch.sigmoid(F.interpolate(patch_logits, size=target.shape[-2:], mode="bilinear", align_corners=False))
    pred, gt = probs > threshold, target > threshold
    inter = (pred & gt).flatten(1).sum(1).float()
    f1 = 2 * inter / (pred.flatten(1).sum(1) + gt.flatten(1).sum(1)).clamp(min=1).float()
    iou = inter / (pred | gt).flatten(1).sum(1).clamp(min=1).float()
    return f1, iou


@torch.no_grad()
def score_manifest(
    model: TraceDetector,
    df: pd.DataFrame,
    device: torch.device,
    max_side: int = 1024,
    num_workers: int = 4,
    amp: bool = True,
) -> pd.DataFrame:
    """Adds columns score, pixel_f1, pixel_iou (NaN where the row has no mask)."""
    model.eval().to(device)
    loader = DataLoader(ManifestDataset(df, model.patch_size, None, max_side), batch_size=1, num_workers=num_workers)
    scores = np.full(len(df), np.nan)
    f1s, ious = np.full(len(df), np.nan), np.full(len(df), np.nan)

    for batch in loader:
        idx = int(batch["index"])
        image = batch["image"].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            out = model(image)
        scores[idx] = torch.sigmoid(out["image_logit"].float()).item()
        if bool(batch["has_mask"][0]):
            f1, iou = pixel_scores(out["patch_logits"].float(), batch["mask"].to(device))
            f1s[idx], ious[idx] = f1.item(), iou.item()

    out_df = df.copy()
    out_df["score"], out_df["pixel_f1"], out_df["pixel_iou"] = scores, f1s, ious
    return out_df


def precision_at_recall(labels: np.ndarray, scores: np.ndarray, recall: float) -> float:
    """Precision at the loosest threshold that still reaches the requested recall."""
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    n_pos = labels.sum()
    if n_pos == 0:
        return float("nan")
    k = int(np.argmax(tp / n_pos >= recall))  # first index reaching the recall
    return float(tp[k] / (k + 1))


def resolve_threshold(thresholds: dict | None, content_type: str) -> float:
    if not thresholds:
        return 0.5
    return float(thresholds.get("per_content_type", {}).get(content_type, thresholds.get("global", 0.5)))


def _group_metrics(g: pd.DataFrame, thresholds: dict | None) -> dict:
    y, s = g["label"].to_numpy(), g["score"].to_numpy()
    thr = np.array([resolve_threshold(thresholds, ct) for ct in g["content_type"]])
    pred = s >= thr
    row = {"n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())}
    if row["n_real"] and row["n_fake"]:
        row["auroc"] = roc_auc_score(y, s)
        row["ap"] = average_precision_score(y, s)
        row["bal_acc"] = 0.5 * ((pred[y == 1]).mean() + (~pred[y == 0]).mean())
        for r in RECALL_LEVELS:
            row[f"prec@rec{r}"] = precision_at_recall(y, s, r)
    if row["n_fake"]:
        row["tpr@thr"] = float(pred[y == 1].mean())
    if row["n_real"]:
        row["fpr@thr"] = float(pred[y == 0].mean())
    if g["pixel_f1"].notna().any():
        row["pixel_f1"] = float(g["pixel_f1"].mean())
        row["pixel_iou"] = float(g["pixel_iou"].mean())
    return row


def summarize(scored: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    """One row for ALL, then per content type, per source and per generator."""
    rows = [{"group": "ALL", **_group_metrics(scored, thresholds)}]
    for column in ("content_type", "source", "generator"):
        for value, g in scored.groupby(column):
            if value == "":
                continue
            rows.append({"group": f"{column}={value}", **_group_metrics(g, thresholds)})
    return pd.DataFrame(rows)


def to_markdown(summary: pd.DataFrame) -> str:
    cols = [c for c in summary.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in summary.iterrows():
        cells = [f"{v:.3f}" if isinstance(v, float) and not np.isnan(v) else ("" if isinstance(v, float) else str(v)) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_model(weights: str | Path, device: torch.device) -> TraceDetector:
    payload = torch.load(weights, map_location="cpu", weights_only=False)
    model = TraceDetector.from_config(payload["model_cfg"])
    model.load_trainable_state_dict(payload["state_dict"])
    return model.to(device).eval()
