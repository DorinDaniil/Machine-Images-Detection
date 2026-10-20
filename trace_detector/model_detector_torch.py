"""Standalone inference wrapper: one image in, a score and a heat map out."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data.dataset import fit_to_multiple, to_tensor
from .dino_trace import TraceDetector


class TraceDetectorInference:
    """
    detector = TraceDetectorInference("weights/trace_detector.pt", device="cuda")
    result = detector.predict(Image.open("photo.jpg"))
    result["score"]     probability that the image contains generated content
    result["map"]       float32 [H, W] per-pixel probability, same size as the analysed image
    result["verdict"]   "generated" / "real" using calibrated thresholds, if provided
    """

    def __init__(self, weights: str | Path, device: str = "cuda", max_side: int = 1024, thresholds: str | Path | None = None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        payload = torch.load(weights, map_location="cpu", weights_only=False)
        self.model = TraceDetector.from_config(payload["model_cfg"])
        self.model.load_trainable_state_dict(payload["state_dict"])
        self.model.to(self.device).eval()
        self.max_side = max_side
        self.thresholds = json.load(open(thresholds, encoding="utf-8")) if thresholds else None

    @torch.no_grad()
    def predict(self, image: Image.Image, content_type: str = "photo") -> dict:
        image = fit_to_multiple(image.convert("RGB"), self.max_side, self.model.patch_size)
        x = to_tensor(np.asarray(image))[None].to(self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            out = self.model(x)
        heat = torch.sigmoid(F.interpolate(out["patch_logits"].float(), size=x.shape[-2:], mode="bilinear", align_corners=False))
        score = float(torch.sigmoid(out["image_logit"].float()))

        result = {"score": score, "map": heat[0, 0].cpu().numpy(), "size": image.size}
        if self.thresholds:
            thr = self.thresholds.get("per_content_type", {}).get(content_type, self.thresholds["global"])
            result["threshold"] = thr
            result["verdict"] = "generated" if score >= thr else "real"
        return result

    def predict_paths(self, paths: list[str | Path], content_type: str = "photo") -> list[dict]:
        return [{"path": str(p), **self.predict(Image.open(p), content_type)} for p in paths]
