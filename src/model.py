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
    def full(self, x: torch.Tensor, tile: int = 512, step: int | None = None):
        """A whole image, nothing cropped or resized, looked at in tiles of tile x tile, the size of the training
        views (a side shorter than that is one tile). Tiles start every `step` px (default: tile, side by side)
        and the last one ends at the image edge; where tiles overlap, their patch logits are averaged. Then the
        map is pooled as in forward. step = tile // 2 removes seams between tiles at about 4x the cost."""
        p, m = self.patch_size, tile // self.patch_size
        s = max(1, (step or tile) // p)
        h, w = x.shape[2] // p, x.shape[3] // p

        def starts(n: int) -> list[int]:  # tile starts in patches along a side of n patches
            return [0] if n <= m else list(range(0, n - m, s)) + [n - m]

        total, count = x.new_zeros(x.shape[0], 1, h, w, dtype=torch.float32), x.new_zeros(1, 1, h, w)
        for y in starts(h):
            for x0 in starts(w):
                total[:, :, y:y + m, x0:x0 + m] += self.patches(x[:, :, y * p:(y + m) * p, x0 * p:(x0 + m) * p]).float()
                count[:, :, y:y + m, x0:x0 + m] += 1
        patch_logits = total / count
        return patch_logits, self.pool(patch_logits)

    def save(self, path) -> None:
        torch.save({"config": self.config, "head": self.head.state_dict()}, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "Detector":
        ckpt = torch.load(path, map_location="cpu")
        model = cls(**ckpt["config"])
        model.head.load_state_dict(ckpt["head"])
        return model.to(device).eval()
