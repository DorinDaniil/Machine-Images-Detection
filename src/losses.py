import torch
import torch.nn.functional as F


def patch_loss(patch_logits: torch.Tensor, image_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """target [B, 1, H, W]: 1 where pixels are generated, 0 where they are real.

    Patch target = share of generated pixels in the patch; image target = max of the target.
    """
    patch_target = F.adaptive_avg_pool2d(target, patch_logits.shape[-2:])
    image_target = target.flatten(1).amax(dim=1)
    patch = F.binary_cross_entropy_with_logits(patch_logits, patch_target)
    image = F.binary_cross_entropy_with_logits(image_logit, image_target)
    return patch + image
