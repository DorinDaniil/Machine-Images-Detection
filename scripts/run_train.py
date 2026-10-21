"""python scripts/run_train.py --config configs/train.yaml

Real images (COCO photos + rendered text / UI / charts / logos) -> 70/15/15 split ->
pairs (real, re-decoded by train_decoders, same JPEG history) -> training -> runs/pilot/best.pt
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import PairDataset, download_coco, make_pairs, render_folder  # noqa: E402
from src.model import TraceDetector  # noqa: E402
from src.train import train  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train.yaml")
cfg = yaml.safe_load(open(parser.parse_args().config))
d, t = cfg["data"], cfg["train"]
data = Path(d["dir"])
device = t["device"] if torch.cuda.is_available() else "cpu"

model = TraceDetector(**cfg["model"])
crop = t["crop"] // model.patch_size * model.patch_size


def split(paths: list, seed: int = 0) -> dict:
    paths = sorted(str(p) for p in paths)
    random.Random(seed).shuffle(paths)
    a, b = int(0.7 * len(paths)), int(0.85 * len(paths))
    return {"train": paths[:a], "calib": paths[a:b], "test": paths[b:]}


# real images, split once and saved for run_test.py
photos = split(download_coco(data / "coco", d["n_real"], min_side=crop))
renders = split(item["path"] for item in render_folder(data / "renders", d["n_render"], "train"))
json.dump({"photos": photos, "renders": renders}, open(data / "split.json", "w"), indent=1)

# training fakes: the same images re-decoded by train_decoders
reals = photos["train"] + renders["train"]
pairs = make_pairs(reals, d["train_decoders"], data / "pairs", device, d["max_side"])
random.Random(1).shuffle(pairs)
n_val = max(1, len(pairs) // 20)
probs = (t["p_real"], t["p_full"], t["p_local"])
train_set = PairDataset(pairs[n_val:], d["train_decoders"], crop, *probs, seed=0)
val_set = PairDataset(pairs[:n_val], d["train_decoders"], crop, *probs, seed=1)
print(f"train pairs {len(train_set)}, val pairs {len(val_set)}, crop {crop}")

best = train(model, train_set, val_set, t)
print(f"best checkpoint: {best}")
