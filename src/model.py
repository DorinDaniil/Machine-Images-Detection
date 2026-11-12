"""Detector: one trunk shared by every expert, and one expert per generator family.

    image -> trunk: the stem and the first stages of a small ConvNeXt, computed once per image
          -> expert k: its own last stage -> mean over the map -> norm -> linear
             = softmax over [real, model 1 of family k, model 2 of family k, ...]

Every expert is trained on its own: reals against the generators of one family. The expert "all" is the general
one: every generator, trained first and together with the trunk. The family experts are then trained on the
frozen trunk, each starting as a copy of "all". One image costs one pass of the trunk and one cheap last stage
per expert, so the whole ensemble runs on a CPU.
"""

import copy
from pathlib import Path

import timm
import torch
import torch.nn as nn

TINY = dict(depths=(1, 1, 1, 1), dims=(8, 16, 32, 64))  # random weights for CPU tests, nothing is downloaded


class Expert(nn.Module):
    """Trunk features [B, C, h, w] -> logits [B, 1 + models]: class 0 is "real", class j is the j-th generator of
    the expert's family. One softmax answers both "is it generated" and "by which model"."""

    def __init__(self, stages: nn.Module, norm: nn.Module, models):
        super().__init__()
        self.models = list(models)
        self.stages, self.norm = stages, norm
        self.head = nn.Linear(norm.normalized_shape[0], 1 + len(self.models))

    def features(self, f: torch.Tensor) -> torch.Tensor:
        return self.norm(self.stages(f).mean(dim=(2, 3), keepdim=True)).flatten(1)  # [B, D]

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(f))


class Detector(nn.Module):
    """x [B, 3, H, W] in [0, 1] -> {expert name: logits [B, 1 + its models]}."""

    def __init__(self, backbone: str, shared: int, experts: dict, pretrained: bool = True):
        """experts: {name: its generators}. shared: how many of the 5 parts [stem, stage 1, ..., stage 4] of the
        ConvNeXt make the trunk; the rest of them is copied into every expert."""
        super().__init__()
        self.config = {"backbone": backbone, "shared": shared}
        tiny = backbone == "tiny"
        net = timm.create_model("convnext_tiny" if tiny else backbone, pretrained=pretrained and not tiny,
                                num_classes=0, **(TINY if tiny else {}))
        parts = [net.stem, *net.stages]
        self.trunk = nn.Sequential(*parts[:shared])
        self.experts = nn.ModuleDict({name: Expert(copy.deepcopy(nn.Sequential(*parts[shared:])),
                                                   copy.deepcopy(net.head.norm), models)
                                      for name, models in experts.items()})
        cfg = net.pretrained_cfg  # ImageNet mean / std of the weights
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1), persistent=False)

    def add(self, name: str, models) -> None:
        """A new expert: a copy of the general one, with a head for its own models."""
        base = self.experts["all"]
        self.experts[name] = Expert(copy.deepcopy(base.stages), copy.deepcopy(base.norm), models).to(self.mean.device)

    def forward(self, x: torch.Tensor, names=None) -> dict:
        f = self.trunk((x - self.mean) / self.std)  # once for every expert
        return {name: self.experts[name](f) for name in names or self.experts}

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """The image vector of the general expert, [B, D]."""
        return self.experts["all"].features(self.trunk((x - self.mean) / self.std))

    def save(self, folder, name: str) -> None:
        """<folder>/experts/<name>.pt; with "all" also <folder>/trunk.pt, the trunk it was trained with."""
        folder = Path(folder)
        (folder / "experts").mkdir(parents=True, exist_ok=True)
        expert = self.experts[name]
        torch.save({"models": expert.models, "weights": expert.state_dict()}, folder / "experts" / f"{name}.pt")
        if name == "all":
            torch.save({"config": self.config, "weights": self.trunk.state_dict()}, folder / "trunk.pt")

    @classmethod
    def load(cls, folder, device: str = "cpu", names=None) -> "Detector":
        """The trunk and the experts of a run folder: every expert found there, or only `names`."""
        folder = Path(folder)
        trunk = torch.load(folder / "trunk.pt", map_location="cpu")
        files = {name: torch.load(folder / "experts" / f"{name}.pt", map_location="cpu")
                 for name in names or sorted(p.stem for p in (folder / "experts").glob("*.pt"))}
        model = cls(**trunk["config"], experts={name: f["models"] for name, f in files.items()}, pretrained=False)
        model.trunk.load_state_dict(trunk["weights"])
        for name, f in files.items():
            model.experts[name].load_state_dict(f["weights"])
        return model.to(device).eval()


def score(logits: torch.Tensor):
    """Logits [B, 1 + models] of one expert -> the probability that the image is generated [B] (1 - "real"),
    and the index of the expert's most likely model [B]."""
    return 1 - torch.softmax(logits.float(), dim=1)[:, 0], logits[:, 1:].argmax(dim=1)
