"""python scripts/run_train.py --config configs/train.yaml [--set KEY=VALUE ...] [--resume]

The images of data/real, data/fake and data.real_dirs, split by the hash of the file path into train (85 %),
val (5 %) and test (10 %); rendered text / UI / charts / logos join the reals. Then one model is trained: reals
against the generated images, classes [real, family 1, ..., family m] (model.families of the config).
After the last epoch the bias of the class "real" is set on validation reals: about train.alpha of them have
p(generated) > 0.5. Result in <out>: model.pt, metrics.csv, config.yaml (the config of this run).

--set     values of the config for this run, e.g. another variant next to the first one:
              --set train.out=runs/contrast train.contrast=0.5
--resume  after a crash or Ctrl+C: goes on from the last saved state (after every epoch and every
          train.save_every steps). Give the config of the run: --config runs/contrast/config.yaml --resume
"""

import argparse
import faulthandler
import random
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(line_buffering=True)  # every line at once, also through `| tee`
faulthandler.enable()  # a crash inside torch or the GPU driver leaves the Python stack in the log, not a silent exit

from src.data import ImageDataset, distribution, load_pool  # noqa: E402
from src.evaluate import calibrate, predict  # noqa: E402
from src.model import Detector  # noqa: E402
from src.train import train  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train.yaml")
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="e.g. train.contrast=0.5")
parser.add_argument("--resume", action="store_true", help="go on from the last saved state")
args = parser.parse_args()
cfg = yaml.safe_load(open(args.config))
for item in args.set:  # train.contrast=0.5 -> cfg["train"]["contrast"] = 0.5
    key, value = item.split("=", 1)
    *blocks, name = key.split(".")
    node = cfg
    for block in blocks:
        node = node[block]
    node[name] = yaml.safe_load(value)
d, m, t = cfg["data"], cfg["model"], cfg["train"]
out = Path(t["out"])
out.mkdir(parents=True, exist_ok=True)
yaml.safe_dump(cfg, open(out / "config.yaml", "w"), sort_keys=False)  # run_test.py and the notebooks read it

parts = load_pool(d)
print(distribution(parts, m["families"]))  # images per generator
reals = {k: [r for r in rows if r["label"] == 0] for k, rows in parts.items()}
fakes = {k: [r for r in rows if r["label"] == 1] for k, rows in parts.items()}

# classes: the families that have images, then the generators that are in no family, each on its own
family_of = {g: name for name, generators in m["families"].items() for g in generators}
present = {r["generator"] for r in fakes["train"]}
classes = [name for name, generators in m["families"].items() if present & set(generators)] + sorted(present - set(family_of))
number = {name: j for j, name in enumerate(classes, 1)}  # label: 0 real, j = the j-th class


def labelled(rows):
    family = [family_of.get(r["generator"], r["generator"]) for r in rows]
    return [{**r, "label": number[f]} for r, f in zip(rows, family) if f in number]


val = labelled(fakes["val"])  # up to val_size images, half generated and half real
val = random.Random(0).sample(val, min(len(val), t["val_size"] // 2))
val += random.Random(0).sample(reals["val"], min(len(reals["val"]), len(val)))
print(f"classes: real, {', '.join(classes)}")
print(f"whole images resized to {t['size']} x {t['size']}, batch {t['batch_size']}, "
      f"balance {t['balance']}, contrast {t['contrast']}, val {len(val)}")
train(Detector(m["backbone"], classes), reals["train"], labelled(fakes["train"]), val, t, resume=args.resume)

device = t["device"] if torch.cuda.is_available() else "cpu"
model = Detector.load(out / "model.pt", device)  # the epoch with the lowest validation loss
calib = random.Random(0).sample(reals["val"], min(len(reals["val"]), t["max_calib"]))
odds = predict(model, ImageDataset(calib, t["size"]), device, t["batch_size"], t["num_workers"])[0]
shift = calibrate(model, odds, t["alpha"])
model.save(out / "model.pt")
print(f"calibrated on {len(calib)} validation reals: the logit of \"real\" moved by {shift:.3f}, "
      f"{(odds > shift).mean():.1%} of them have p(generated) > 0.5")
