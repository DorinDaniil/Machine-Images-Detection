"""python scripts/run_test.py --config configs/train.yaml

Test images:
    real        held-out COCO photos
    <vae> full  the same photos fully re-decoded by a VAE not seen in training
    <vae> local the same photos with one re-decoded region (mask saved, pixel F1 reported)
    <model>     images of real generators from Qwen/Qwen-Image-Bench
Threshold: the score exceeded by 1 % of the real photos. Result: runs/pilot/report.md
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ImageDataset, blend, download_fakes, make_pairs, random_mask  # noqa: E402
from src.evaluate import report, score  # noqa: E402
from src.model import TraceDetector  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train.yaml")
cfg = yaml.safe_load(open(parser.parse_args().config))
d, t = cfg["data"], cfg["train"]
data, out = Path(d["dir"]), Path(t["out"])
device = t["device"] if torch.cuda.is_available() else "cpu"
model = TraceDetector.load(out / "best.pt", device)

# held-out photos and their copies re-decoded by unseen VAEs
split = json.load(open(data / "split.json"))
pairs = make_pairs(split["test"], d["test_vae"], data / "pairs_test", device, d["max_side"])
reals = [{"path": p / "real.png", "label": 0} for p in pairs]
groups = {f"{vae} {kind}": [] for vae in d["test_vae"] for kind in ("full", "local")}
rng = np.random.default_rng(0)
for p in pairs:
    real = np.asarray(Image.open(p / "real.png"), dtype=np.float32)
    for vae in d["test_vae"]:
        fake = np.asarray(Image.open(p / f"{vae}.png"), dtype=np.float32)
        mask = random_mask(*real.shape[:2], rng)
        for kind, m in (("full", np.ones_like(mask)), ("local", mask)):
            Image.fromarray(blend(real, fake, m)).save(p / f"test_{vae}_{kind}.png")
            Image.fromarray((m * 255).astype(np.uint8)).save(p / f"test_{vae}_{kind}_mask.png")
            groups[f"{vae} {kind}"].append({"path": p / f"test_{vae}_{kind}.png", "label": 1,
                                            "mask": p / f"test_{vae}_{kind}_mask.png"})

# images of real generators
for gen, paths in download_fakes(data / "qwen_image_bench", d["generators"], d["n_per_generator"]).items():
    groups[gen] = [{"path": p, "label": 1} for p in paths]

# score and report
real_scores, _ = score(model, ImageDataset(reals, model.patch_size, d["max_side"]), device)
results = {name: score(model, ImageDataset(items, model.patch_size, d["max_side"]), device) for name, items in groups.items()}
table = report(real_scores, results)
(out / "report.md").write_text(table, encoding="utf-8")
print(table)
