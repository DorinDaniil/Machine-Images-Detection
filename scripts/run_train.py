"""python scripts/run_train.py --config configs/train.yaml [--resume]

OpenFake core/train (reals + generators) and SID_Set train (reals, fully synthetic and locally edited images
with masks) -> 15 % of the reals kept apart for calibration -> generator-balanced training of the patch head
on a frozen DINOv3, whole images with augmentations -> runs/pilot/best.pt
--resume: go on from runs/pilot/state.pt after a crash or Ctrl+C (saved every save_every steps and every epoch)
"""

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(line_buffering=True)  # every line at once, also through `| tee`

from src.data import TrainDataset, openfake, render_folder, sid_set  # noqa: E402
from src.model import Detector  # noqa: E402
from src.train import train  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train.yaml")
parser.add_argument("--resume", action="store_true", help="go on from <out>/state.pt")
args = parser.parse_args()
cfg = yaml.safe_load(open(args.config))
d, t = cfg["data"], cfg["train"]
data, out = Path(d["dir"]), Path(t["out"])
out.mkdir(parents=True, exist_ok=True)

model = Detector(**cfg["model"])

rows = openfake("train", data / "openfake_train", workers=d["workers"], **d["train"])
if (d.get("sid_train") or {}).get("n_per_class"):
    rows += sid_set("train", data / "sid_train", **d["sid_train"])
random.Random(0).shuffle(rows)
reals, fakes = [r for r in rows if r["label"] == 0], [r for r in rows if r["label"] == 1]
n_calib = int(0.15 * len(reals))
calib, reals = reals[:n_calib], reals[n_calib:]

renders = [str(it["path"]) for it in render_folder(data / "renders", d["n_render"], "test")]
json.dump({"reals": [r["path"] for r in calib], "renders": renders[: len(renders) // 2],
           "renders_test": renders[len(renders) // 2:]}, open(out / "calibration.json", "w"), indent=1)  # next to best.pt

pool = reals + fakes
random.Random(1).shuffle(pool)
n_val = min(t.get("val_size", 5000), max(2, len(pool) // 20))
augment = {"crop_p": t.get("crop_p", 0.5), "down_p": t.get("down_p", 0.3)}
train_set = TrainDataset(pool[n_val:], t["max_side"], model.patch_size, seed=0, **augment)
val_set = TrainDataset(pool[:n_val], t["max_side"], model.patch_size, seed=1, crop_p=0, down_p=0)  # whole images
generators = sorted({r["generator"] for r in fakes})
masked = sum(bool(r.get("mask")) for r in fakes)
print(f"train {len(train_set)} images ({len(reals)} reals, {len(generators)} generators, {masked} with masks), "
      f"val {len(val_set)}, calibration {len(calib)} reals; whole images, crop {augment['crop_p']:.0%}, "
      f"downscale {augment['down_p']:.0%}")

best = train(model, train_set, val_set, t, resume=args.resume)
print(f"best checkpoint: {best}")
