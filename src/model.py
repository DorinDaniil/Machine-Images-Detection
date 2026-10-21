"""Detector: frozen backbone -> one logit per patch -> image logit."""

import torch
import torch.nn as nn

from .backbone import Backbone


class TraceDetector(nn.Module):
    """forward(x [B, 3, H, W]) -> patch logits [B, 1, H // p, W // p], image logit [B].

    The head sees the patch features of the chosen backbone blocks side by side. The image logit is the
    mean of the top `topk` share of patch logits, so a small edited region still moves the whole score.
    """

    def __init__(self, backbone: str, layers=(-1,), head_dim: int = 256, topk: float = 0.05):
        super().__init__()
        self.config = {"backbone": backbone, "layers": list(layers), "head_dim": head_dim, "topk": topk}
        self.backbone = Backbone(backbone, layers)
        in_dim = self.backbone.dim * len(self.backbone.layers)
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, head_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(head_dim, head_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(head_dim, 1, kernel_size=1),
        )
        self.topk = topk

    @property
    def patch_size(self) -> int:
        return self.backbone.patch_size

    def forward(self, x: torch.Tensor):
        grid = torch.cat(self.backbone(x)["grids"], dim=1)
        patch_logits = self.head(grid)
        flat = patch_logits.flatten(1)
        k = max(1, round(self.topk * flat.shape[1]))
        image_logit = flat.topk(k, dim=1).values.mean(dim=1)
        return patch_logits, image_logit

    def save(self, path) -> None:
        torch.save({"config": self.config, "head": self.head.state_dict()}, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "TraceDetector":
        ckpt = torch.load(path, map_location="cpu")
        model = cls(**ckpt["config"])
        model.head.load_state_dict(ckpt["head"])
        return model.to(device).eval()
