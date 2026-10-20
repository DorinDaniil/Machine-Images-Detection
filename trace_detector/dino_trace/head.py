"""Patch-level head and image-level pooling."""

import torch
import torch.nn as nn


class PatchTraceHead(nn.Module):
    """One logit per patch: "this patch came out of a generator decoder".

    Input:  feature grid [B, D, gh, gw].
    Output: patch logits [B, 1, gh, gw].
    """

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
    """Image logit = mean of the top `ratio` fraction of patch logits.

    A small edited region moves the image score even if most patches look real.
    """
    flat = patch_logits.flatten(1)
    k = max(1, int(round(ratio * flat.shape[1])))
    return flat.topk(k, dim=1).values.mean(dim=1)
