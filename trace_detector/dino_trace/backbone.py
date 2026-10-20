"""Frozen ViT backbone (DINOv3 / DINOv2) that returns a grid of patch features."""

from contextlib import nullcontext

import torch
import torch.nn as nn
from transformers import AutoModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrozenViTBackbone(nn.Module):
    """Hugging Face DINO model wrapped as a dense feature extractor.

    Input:  pixel_values [B, 3, H, W], values in [0, 1], any H and W.
    Output: patch grid   [B, D, H // patch, W // patch].

    Everything is frozen except the last `train_last_blocks` transformer blocks.
    """

    def __init__(self, name: str, train_last_blocks: int = 0, model: nn.Module | None = None):
        super().__init__()
        self.name = name
        self.model = model if model is not None else AutoModel.from_pretrained(name)
        cfg = self.model.config
        self.patch_size = int(cfg.patch_size)
        self.hidden_size = int(cfg.hidden_size)
        # CLS token plus register tokens come first in the token sequence.
        self.num_special_tokens = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.train_last_blocks = train_last_blocks
        self._freeze(train_last_blocks)

    @classmethod
    def tiny_for_tests(cls) -> "FrozenViTBackbone":
        """Randomly initialised 2-block DINOv2 for CPU smoke tests (no download)."""
        from transformers import Dinov2Config, Dinov2Model

        cfg = Dinov2Config(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            patch_size=14,
            image_size=224,
        )
        return cls(name="tiny-dinov2", model=Dinov2Model(cfg))

    def _blocks(self) -> nn.ModuleList:
        """Transformer blocks live under different attribute names in DINOv2 and DINOv3."""
        lists = [m for m in self.model.modules() if isinstance(m, nn.ModuleList)]
        return max(lists, key=len)

    def _freeze(self, train_last_blocks: int) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        if train_last_blocks > 0:
            for block in self._blocks()[-train_last_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

    @property
    def is_fully_frozen(self) -> bool:
        return self.train_last_blocks == 0

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        b, _, h, w = pixel_values.shape
        x = (pixel_values - self.mean) / self.std
        ctx = torch.no_grad() if self.is_fully_frozen else nullcontext()
        with ctx:
            tokens = self.model(pixel_values=x).last_hidden_state  # [B, special + N, D]
        patches = tokens[:, self.num_special_tokens :, :]
        gh, gw = h // self.patch_size, w // self.patch_size
        return patches.transpose(1, 2).reshape(b, self.hidden_size, gh, gw)
