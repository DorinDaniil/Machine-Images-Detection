"""Detector: frozen backbone -> one logit per patch -> image logit."""

import torch
import torch.nn as nn

from .backbone import Backbone


class Detector(nn.Module):
    """forward(x [B, 3, H, W]) -> patch logits [B, 1, H // p, W // p], image logit [B].

    The head sees the patch features of the chosen backbone blocks side by side. The image logit is the
    mean of the top `topk` share of patch logits, so a small generated region still moves the whole score.
    full(x, tile) scores a whole image of any size tile by tile; forward is for views of at most tile x tile.
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

    def patches(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, 3, H, W], H and W multiples of the patch size -> patch logits [B, 1, H // p, W // p]."""
        return self.head(torch.cat(self.backbone(x)["grids"], dim=1))

    def pool(self, patch_logits: torch.Tensor) -> torch.Tensor:
        flat = patch_logits.float().flatten(1)  # float32: under bf16 autocast scores would tie at the threshold
        k = max(1, round(self.topk * flat.shape[1]))
        return flat.topk(k, dim=1).values.mean(dim=1)

    def forward(self, x: torch.Tensor):
        patch_logits = self.patches(x)
        return patch_logits, self.pool(patch_logits)

    @torch.no_grad()
    def full(self, x: torch.Tensor, tile: int = 1024):
        """A whole image, nothing cropped or resized: cut into nearly equal tiles of at most tile x tile on the
        patch grid, each tile through the model, the patch logits stitched back into one map, then pooled."""
        p, m = self.patch_size, tile // self.patch_size

        def cuts(n: int) -> list[int]:  # n patches -> pixel offsets of ceil(n / m) nearly equal parts
            parts = -(-n // m)
            return [round(i * n / parts) * p for i in range(parts + 1)]

        ys, xs = cuts(x.shape[2] // p), cuts(x.shape[3] // p)
        rows = [torch.cat([self.patches(x[:, :, y0:y1, x0:x1]) for x0, x1 in zip(xs, xs[1:])], dim=3)
                for y0, y1 in zip(ys, ys[1:])]
        patch_logits = torch.cat(rows, dim=2)
        return patch_logits, self.pool(patch_logits)

    def save(self, path) -> None:
        torch.save({"config": self.config, "head": self.head.state_dict()}, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "Detector":
        ckpt = torch.load(path, map_location="cpu")
        model = cls(**ckpt["config"])
        model.head.load_state_dict(ckpt["head"])
        return model.to(device).eval()
