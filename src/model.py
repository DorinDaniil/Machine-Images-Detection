"""TraceDetector: frozen DINO backbone -> per-patch logits -> top-k pooled image logit."""

import torch
import torch.nn as nn

from .backbone import FrozenViTBackbone


class PatchHead(nn.Module):
    """One logit per patch. Input [B, D, gh, gw] -> output [B, 1, gh, gw]."""

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.net(grid)


def topk_mean(patch_logits: torch.Tensor, ratio: float) -> torch.Tensor:
    """Image logit = mean of the top `ratio` fraction of patch logits, so a small edit still counts."""
    flat = patch_logits.flatten(1)
    k = max(1, int(round(ratio * flat.shape[1])))
    return flat.topk(k, dim=1).values.mean(dim=1)


class TraceDetector(nn.Module):
    """forward(pixel_values [B, 3, H, W] in [0, 1]) -> patch_logits [B, 1, H//p, W//p], image_logit [B]."""

    def __init__(self, backbone: FrozenViTBackbone, head_dim: int = 256, topk_ratio: float = 0.05):
        super().__init__()
        self.backbone = backbone
        self.head = PatchHead(backbone.hidden_size, head_dim)
        self.topk_ratio = topk_ratio

    @classmethod
    def from_config(cls, cfg: dict) -> "TraceDetector":
        """cfg keys: backbone, train_last_blocks, head_dim, topk_ratio."""
        if cfg["backbone"] == "tiny-dinov2":
            backbone = FrozenViTBackbone.tiny_for_tests()
        else:
            backbone = FrozenViTBackbone(cfg["backbone"], cfg.get("train_last_blocks", 0))
        return cls(backbone, cfg.get("head_dim", 256), cfg.get("topk_ratio", 0.05))

    @property
    def patch_size(self) -> int:
        return self.backbone.patch_size

    def forward(self, pixel_values: torch.Tensor) -> dict:
        patch_logits = self.head(self.backbone(pixel_values))
        return {"patch_logits": patch_logits, "image_logit": topk_mean(patch_logits, self.topk_ratio)}

    def trainable_state_dict(self) -> dict:
        """Only what was trained: the head and any unfrozen backbone blocks."""
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in trainable}

    def load_trainable_state_dict(self, state: dict) -> None:
        _, unexpected = self.load_state_dict(state, strict=False)
        if unexpected:
            raise KeyError(f"Unexpected keys in weights: {unexpected[:5]}")


def load_model(weights: str, device: torch.device) -> TraceDetector:
    """Load best.pt / last.pt saved by train.py."""
    payload = torch.load(weights, map_location="cpu", weights_only=False)
    model = TraceDetector.from_config(payload["model_cfg"])
    model.load_trainable_state_dict(payload["state_dict"])
    return model.to(device).eval()
