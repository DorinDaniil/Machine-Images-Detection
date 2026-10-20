"""Frozen DINO ViT that returns a grid of patch features."""

import torch
import torch.nn as nn
from transformers import AutoModel

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def tiny_vit() -> nn.Module:
    """Random 2-block DINOv2 for CPU tests, nothing is downloaded."""
    from transformers import Dinov2Config, Dinov2Model

    return Dinov2Model(Dinov2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                                    intermediate_size=128, patch_size=14))


class Backbone(nn.Module):
    """[B, 3, H, W] in [0, 1] -> patch features [B, D, H // p, W // p]. Never trained."""

    def __init__(self, name: str):
        super().__init__()
        self.vit = tiny_vit() if name == "tiny" else AutoModel.from_pretrained(name)
        cfg = self.vit.config
        self.patch_size = cfg.patch_size
        self.dim = cfg.hidden_size
        self.n_special = 1 + (getattr(cfg, "num_register_tokens", 0) or 0)  # CLS + register tokens
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        self.vit.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vit.eval()  # frozen: always in eval mode
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        tokens = self.vit(pixel_values=(x - self.mean) / self.std).last_hidden_state
        patches = tokens[:, self.n_special:]  # drop CLS and registers
        return patches.transpose(1, 2).reshape(b, self.dim, h // self.patch_size, w // self.patch_size)
