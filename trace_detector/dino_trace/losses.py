"""Losses on soft target maps.

The target map is a float image in [0, 1]: 0 = original pixels, alpha = pixels blended
with a decoder output at mixing ratio alpha. The image label is the maximum of the map.
"""

import torch
import torch.nn.functional as F


def pool_target_to_grid(target_map: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    """Average the pixel target map over each patch: [B, 1, H, W] -> [B, 1, gh, gw]."""
    return F.adaptive_avg_pool2d(target_map, grid_hw)


def soft_dice(patch_probs: torch.Tensor, patch_target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Dice on fake samples only (real samples have an empty target and no region to overlap)."""
    is_fake = patch_target.flatten(1).amax(dim=1) > 0
    if not bool(is_fake.any()):
        return patch_probs.sum() * 0.0
    p = patch_probs[is_fake].flatten(1)
    t = patch_target[is_fake].flatten(1)
    dice = (2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)
    return (1 - dice).mean()


def trace_loss(
    patch_logits: torch.Tensor,
    image_logit: torch.Tensor,
    target_map: torch.Tensor,
    weights: dict,
) -> dict:
    """Weighted sum of patch BCE, image BCE and Dice. Returns every term for logging."""
    patch_target = pool_target_to_grid(target_map, patch_logits.shape[-2:])
    image_target = target_map.flatten(1).amax(dim=1)

    patch_bce = F.binary_cross_entropy_with_logits(patch_logits, patch_target)
    image_bce = F.binary_cross_entropy_with_logits(image_logit, image_target)
    dice = soft_dice(torch.sigmoid(patch_logits), patch_target)

    total = weights["patch"] * patch_bce + weights["image"] * image_bce + weights["dice"] * dice
    return {"total": total, "patch_bce": patch_bce, "image_bce": image_bce, "dice": dice}
