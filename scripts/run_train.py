"""python scripts/run_train.py --config configs/train.yaml [--resume] [--only google,qwen]

OpenFake core/train (reals + generators) -> 15 % of the reals kept apart for calibration -> rendered text / UI /
charts / logos added to the other reals (render_train) -> the experts, one after another:
    all       every generator against the reals, trained together with the trunk -> <out>/trunk.pt
    family    the generators of one family (model.families) against the reals, on the frozen trunk,
              starting as a copy of "all"
Every expert is saved to <out>/experts/<name>.pt at its best validation epoch.
--resume  after a crash or Ctrl+C: trained experts are skipped, the unfinished one goes on from its last epoch
--only    train only these experts, e.g. a family added to the config later. After "all" is retrained every
          family expert has to be retrained too: they stand on its trunk
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

from src.data import openfake, render_folder  # noqa: E402
from src.model import Detector  # noqa: E402
from src.train import train, trained  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train.yaml")
parser.add_argument("--resume", action="store_true", help="skip trained experts, go on with the unfinished one")
parser.add_argument("--only", default="", help="experts to train, comma-separated; default: all of them")
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
           "renders_test": renders[len(renders) // 2:]}, open(out / "calibration.json", "w"), indent=1)  # next to the experts

experts = {"all": [g for models in m["families"].values() for g in models], **m["families"]}
missing = sorted({r["generator"] for r in fakes} - set(experts["all"]))
assert not missing, f"generators without a family in model.families: {missing}"

# validation: some of the reals, and the first and then every 20th image of every generator
random.Random(1).shuffle(reals)
n_val = t.get("val_size", 5000) // 2
val_reals, reals = reals[:n_val], reals[n_val:]
val_fakes, train_fakes, count = [], [], Counter()
for r in fakes:
    count[r["generator"]] += 1
    (val_fakes if count[r["generator"]] % 20 == 1 else train_fakes).append(r)
print(f"{len(reals)} train reals incl. {n_drawn} renders, {len(train_fakes)} generated images of {len(count)} generators; "
      f"calibration {len(calib)} reals; whole images resized to {t['size']} x {t['size']}, batch {t['batch_size']}")

for name in args.only.split(",") if args.only else experts:
    if args.resume and trained(out, name, t["epochs"]):
        print(f"{name}: trained, skipped")
        continue
    number = {g: j for j, g in enumerate(experts[name], 1)}  # label: 0 real, j = the j-th model of the expert
    mine = [{**r, "label": number[r["generator"]]} for r in train_fakes if r["generator"] in number]
    val = [{**r, "label": number[r["generator"]]} for r in val_fakes if r["generator"] in number][:n_val]
    val += val_reals[: len(val)]
    if name == "all":
        model = Detector(m["backbone"], m["shared"], {"all": experts["all"]})  # pretrained DINOv3
    else:
        model = Detector.load(out, names=["all"])  # the trained trunk
        model.add(name, experts[name])             # a copy of the general expert
    print(f"expert {name}: {len(mine)} generated images of {len(number)} models and as many reals per epoch, val {len(val)}")
    train(model, name, reals, mine, val, t, resume=args.resume)
