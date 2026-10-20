"""python scripts/run_train.py --config configs/train.yaml

COCO photos -> 80/20 split -> pairs (real, re-decoded by train_vae) -> training -> runs/pilot/best.pt
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import PairDataset, download_coco, make_pairs  # noqa: E402
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

# real photos, split once and saved for run_test.py
photos = [str(p) for p in download_coco(data / "coco", d["n_real"], min_side=crop)]
random.Random(0).shuffle(photos)
n_train = int(0.8 * len(photos))
split = {"train": photos[:n_train], "test": photos[n_train:]}
json.dump(split, open(data / "split.json", "w"), indent=1)

# fakes for training: the same photos re-decoded by train_vae
pairs = make_pairs(split["train"], d["train_vae"], data / "pairs_train", device, d["max_side"])
n_val = max(1, len(pairs) // 20)
probs = (t["p_real"], t["p_full"], t["p_local"])
train_set = PairDataset(pairs[n_val:], d["train_vae"], crop, *probs, seed=0)
val_set = PairDataset(pairs[:n_val], d["train_vae"], crop, *probs, seed=1)
print(f"train pairs {len(train_set)}, val pairs {len(val_set)}, crop {crop}")

best = train(model, train_set, val_set, t)
print(f"best checkpoint: {best}")
