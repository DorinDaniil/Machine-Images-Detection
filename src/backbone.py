"""Frozen ViT from timm.

DINOv3 weights are on the Hub without an access request (timm/vit_base_patch16_dinov3.lvd1689m,
DINOv3 license); DINOv2 with registers (timm/vit_base_patch14_reg4_dinov2.lvd142m) is Apache-2.0.
"""

import timm
import torch
import torch.nn as nn

TINY = dict(depth=2, embed_dim=64, num_heads=2)  # random weights for CPU tests, nothing is downloaded


class Backbone(nn.Module):
    """x [B, 3, H, W] in [0, 1] -> {"grids": [B, D, H // p, W // p] for every block in `layers`, "cls": [B, D]}.

    Never trained: no gradients, always in eval mode.
    """

    def __init__(self, name: str, layers=(-1,)):
        super().__init__()
        tiny = name == "tiny"
        self.vit = timm.create_model("vit_small_patch16_dinov3" if tiny else name, pretrained=not tiny,
                                     dynamic_img_size=True, **(TINY if tiny else {}))
        n_blocks = len(self.vit.blocks)
        self.layers = [i % n_blocks for i in layers]
        self.patch_size = self.vit.patch_embed.patch_size[0]
        self.dim = self.vit.embed_dim
        cfg = self.vit.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))
        self.vit.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vit.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict:
        out = self.vit.forward_intermediates((x - self.mean) / self.std, indices=self.layers, norm=True,
                                             return_prefix_tokens=True, output_fmt="NCHW", intermediates_only=True)
        return {"grids": [grid for grid, _ in out], "cls": out[-1][1][:, 0]}
