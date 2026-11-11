"""Detector: DINOv3 ViT -> mean of the patch tokens -> one linear head per generator family."""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

TINY = dict(depth=2, embed_dim=64, num_heads=2)  # random weights for CPU tests, nothing is downloaded


class Detector(nn.Module):
    """x [B, 3, H, W] in [0, 1] -> logits [B, K]: head k answers "made by family k" against "real".

    The heads are specialists on one shared backbone: an ensemble for the price of one model. An image is
    called generated when any head fires (`score`). The last `train_blocks` blocks of the ViT are trained
    together with the heads, the blocks before them stay as pretrained.
    """

    def __init__(self, backbone: str, families, train_blocks: int = 8, pretrained: bool = True):
        super().__init__()
        self.config = {"backbone": backbone, "families": list(families), "train_blocks": train_blocks}
        tiny = backbone == "tiny"
        self.vit = timm.create_model("vit_small_patch16_dinov3" if tiny else backbone, pretrained=pretrained and not tiny,
                                     num_classes=0, dynamic_img_size=True, **(TINY if tiny else {}))
        self.head = nn.Linear(self.vit.embed_dim, len(self.config["families"]))
        cfg = self.vit.pretrained_cfg  # ImageNet mean / std of the weights
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))
        self.vit.requires_grad_(False)
        if train_blocks:
            for part in [*self.vit.blocks[-train_blocks:], self.vit.norm]:
                part.requires_grad_(True)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.vit.forward_features((x - self.mean) / self.std)  # [B, class + registers + patches, D]
        return tokens[:, self.vit.num_prefix_tokens:].mean(dim=1)       # mean over the patches: [B, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def save(self, path) -> None:
        torch.save({"config": self.config, "weights": self.state_dict()}, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "Detector":
        ckpt = torch.load(path, map_location="cpu")
        model = cls(**ckpt["config"], pretrained=False)  # every weight is in the file
        model.load_state_dict(ckpt["weights"])
        return model.to(device).eval()


def family_loss(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """label [B]: 0 = real, k = made by family k. Two means of binary cross-entropy: over the real images, where
    every head must say "real", and over the generated ones, where the head of the image's family must say
    "mine"; the other heads are not asked about a generated image. Reals and fakes weigh the same, so every
    head is a balanced detector "its family against real"."""
    real = label == 0
    own = F.one_hot(label, logits.shape[1] + 1)[:, 1:].bool()  # [B, K]: the head of the image's family
    loss = logits.new_zeros(())
    if real.any():
        loss = loss + F.softplus(logits[real]).mean()   # target 0
    if own.any():
        loss = loss + F.softplus(-logits[own]).mean()   # target 1
    return loss


def score(logits: torch.Tensor):
    """Image score in [0, 1]: the most confident head, and which head it is."""
    return torch.sigmoid(logits.float()).max(dim=1)
