"""TraceDetector = frozen DINO backbone + patch head + top-k pooling."""

import torch
import torch.nn as nn

from .backbone import FrozenViTBackbone
from .head import PatchTraceHead, topk_mean


class TraceDetector(nn.Module):
    """Detects generated content at patch level and aggregates to an image score.

    forward(pixel_values [B, 3, H, W] in [0, 1]) ->
        patch_logits [B, 1, H // patch, W // patch]
        image_logit  [B]
    """

    def __init__(self, backbone: FrozenViTBackbone, head_dim: int = 256, topk_ratio: float = 0.05):
        super().__init__()
        self.backbone = backbone
        self.head = PatchTraceHead(backbone.hidden_size, head_dim)
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
        grid = self.backbone(pixel_values)
        patch_logits = self.head(grid)
        image_logit = topk_mean(patch_logits, self.topk_ratio)
        return {"patch_logits": patch_logits, "image_logit": image_logit}

    def trainable_state_dict(self) -> dict:
        """Only the parameters that were trained: the head and unfrozen backbone blocks."""
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in trainable}

    def load_trainable_state_dict(self, state: dict) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected:
            raise KeyError(f"Unexpected keys in exported weights: {unexpected[:5]}")
