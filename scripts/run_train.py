"""python scripts/run_train.py --config configs/train.yaml [--resume]

OpenFake core/train (reals + generators) -> 15 % of the reals kept apart for calibration -> rendered text / UI /
charts / logos added to the other reals (render_train) -> every generator gets the number of its family
(model.families) -> family-balanced training of a DINOv3 with one head per family, whole images resized
to size x size, augmented -> <out>/best.pt
--resume: go on from <out>/state.pt after a crash or Ctrl+C (saved every epoch)
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(line_buffering=True)  # every line at once, also through `| tee`

from src.data import ImageDataset, openfake, render_folder  # noqa: E402
from src.model import Detector  # noqa: E402
from src.train import train  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train.yaml")
parser.add_argument("--resume", action="store_true", help="go on from <out>/state.pt")
args = parser.parse_args()
cfg = yaml.safe_load(open(args.config))
d, m, t = cfg["data"], cfg["model"], cfg["train"]
data, out = Path(d["dir"]), Path(t["out"])
out.mkdir(parents=True, exist_ok=True)

rows = openfake("train", data / "openfake_train", workers=d["workers"], **d["train"])
random.Random(0).shuffle(rows)
reals, fakes = [r for r in rows if r["label"] == 0], [r for r in rows if r["label"] == 1]
n_calib = int(0.15 * len(reals))
calib, reals = reals[:n_calib], reals[n_calib:]
n_drawn = round(d.get("render_train", 0.0) * len(reals))  # non-photo reals against false positives on graphics
print(f"rendering {n_drawn} text pages, UI screens, charts, logos as training reals (once, then from disk)")
reals += [{"path": str(it["path"]), "label": 0, "generator": "", "mask": "", "source": f"render/{it['content']}"}
          for it in render_folder(data / "renders", n_drawn, "train")]

renders = [str(it["path"]) for it in render_folder(data / "renders", d["n_render"], "test")]
json.dump({"reals": [r["path"] for r in calib], "renders": renders[: len(renders) // 2],
           "renders_test": renders[len(renders) // 2:]}, open(out / "calibration.json", "w"), indent=1)  # next to best.pt

families = list(m["families"])
number = {g: k for k, name in enumerate(families, 1) for g in m["families"][name]}  # label: 0 real, k = family k
missing = sorted({r["generator"] for r in fakes} - set(number))
assert not missing, f"generators without a family in model.families: {missing}"
for r in fakes:
    r["label"] = number[r["generator"]]
counts = Counter(r["label"] for r in fakes)
print("images per family: " + ", ".join(f"{name} {counts[k]}" for k, name in enumerate(families, 1)))

pool = reals + fakes
random.Random(1).shuffle(pool)
n_val = min(t.get("val_size", 5000), max(2, len(pool) // 20))
train_set = ImageDataset(pool[n_val:], t["size"], train=True)
val_set = ImageDataset(pool[:n_val], t["size"])  # whole images, resized, as in testing
print(f"train {len(train_set)} images ({len(reals)} reals incl. {n_drawn} renders, {len(number)} generators in "
      f"{len(families)} families), val {len(val_set)}, calibration {len(calib)} reals; "
      f"whole images resized to {t['size']} x {t['size']}, batch {t['batch_size']}")

model = Detector(m["backbone"], families, m["train_blocks"])
best = train(model, train_set, val_set, t, resume=args.resume)
print(f"best checkpoint: {best}")
