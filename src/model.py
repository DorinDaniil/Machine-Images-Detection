"""Detector: a plain image classifier.

    image [3, 512, 512] -> ConvNeXt-Tiny (DINOv3) -> mean over the map -> norm -> vector [768]
                        -> linear -> softmax over [real, family 1, ..., family m]

The softmax answers two questions at once: p(generated) = 1 - p(real), and the family of the generator is the
largest of the other classes. A family is the generators of one maker or one architecture: OpenAI, Google,
Flux, ... (model.families of the config). After training the bias of the class "real" is set on held-out real
images (src/evaluate.py, calibrate), so the model is used as it is: generated when p(generated) > 0.5.
"""

import timm
import torch
import torch.nn as nn

TINY = dict(depths=(1, 1, 1, 1), dims=(8, 16, 32, 64))  # random weights for CPU tests, nothing is downloaded


class Detector(nn.Module):
    """x [B, 3, H, W] in [0, 1] -> logits [B, 1 + m]: class 0 is "real", class j is the j-th of the m `classes`."""

    def __init__(self, backbone: str, classes, pretrained: bool = True):
        super().__init__()
        self.config = {"backbone": backbone, "classes": list(classes)}
        tiny = backbone == "tiny"
        self.net = timm.create_model("convnext_tiny" if tiny else backbone, pretrained=pretrained and not tiny,
                                     num_classes=0, **(TINY if tiny else {}))
        self.head = nn.Linear(self.net.num_features, 1 + len(self.classes))
        cfg = self.net.pretrained_cfg  # ImageNet mean / std of the weights
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1), persistent=False)

    @property
    def classes(self) -> list:
        """The names of the classes 1..m: the families, and the generators that are in no family."""
        return self.config["classes"]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """The image vector [B, D]."""
        return self.net((x - self.mean) / self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def save(self, path) -> None:
        torch.save({"config": self.config, "weights": self.state_dict()}, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "Detector":
        file = torch.load(path, map_location="cpu")
        model = cls(**file["config"], pretrained=False)
        model.load_state_dict(file["weights"])
        return model.to(device).eval()


def score(logits: torch.Tensor):
    """Logits [B, 1 + m] -> the probability that the image is generated [B] (1 - "real"), and the index of the
    most likely of the m classes [B]."""
    return 1 - torch.softmax(logits.float(), dim=1)[:, 0], logits[:, 1:].argmax(dim=1)
