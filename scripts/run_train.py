"""python scripts/run_train.py --config configs/train.yaml

OpenFake core/train (reals + ~70 generators) -> reals split into train / calibration ->
generator-balanced training of the patch head on a frozen DINOv3 -> runs/pilot/best.pt
"""

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import TrainDataset, openfake, render_folder  # noqa: E402
from src.model import Detector  # noqa: E402
from src.train import train  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train.yaml")
cfg = yaml.safe_load(open(parser.parse_args().config))
d, t = cfg["data"], cfg["train"]
data = Path(d["dir"])

model = Detector(**cfg["model"])

rows = openfake("train", data / "openfake_train", workers=d["workers"], **d["train"])
random.Random(0).shuffle(rows)
reals, fakes = [r for r in rows if r["label"] == 0], [r for r in rows if r["label"] == 1]
n_calib = int(0.15 * len(reals))
calib, reals = reals[:n_calib], reals[n_calib:]

renders = [str(it["path"]) for it in render_folder(data / "renders", d["n_render"], "test")]
json.dump({"openfake_reals": [r["path"] for r in calib], "renders": renders[: len(renders) // 2],
           "renders_test": renders[len(renders) // 2:]}, open(data / "calibration.json", "w"), indent=1)

pool = reals + fakes
random.Random(1).shuffle(pool)
n_val = min(t.get("val_size", 5000), max(2, len(pool) // 20))
train_set = TrainDataset(pool[n_val:], t["max_side"], model.patch_size, seed=0)
val_set = TrainDataset(pool[:n_val], t["max_side"], model.patch_size, seed=1)
generators = sorted({r["generator"] for r in fakes})
print(f"train {len(train_set)} images ({len(reals)} reals, {len(generators)} generators), val {len(val_set)}, "
      f"own resolution up to {t['max_side']} px a side")

best = train(model, train_set, val_set, t)
print(f"best checkpoint: {best}")
