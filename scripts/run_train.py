"""python scripts/run_train.py --config configs/train.yaml [--resume] [--only google,qwen]

The images of data/real, data/fake and data.real_dirs, split by the hash of the file path into train (85 %),
val (5 %) and test (10 %); rendered text / UI / charts / logos join the reals. Then the experts, one after another:
    all       every generator against the reals, trained together with the trunk -> <out>/trunk.pt
    family    the generators of one family (model.families) against the reals, on the frozen trunk,
              starting as a copy of "all"
Every expert is saved to <out>/experts/<name>.pt at its best validation epoch.
--resume  after a crash or Ctrl+C: trained experts are skipped, the unfinished one goes on from its last saved
          state (after every epoch and every train.save_every steps)
--only    train only these experts, e.g. a family added to the config later. After "all" is retrained every
          family expert has to be retrained too: they stand on its trunk
"""

import argparse
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(line_buffering=True)  # every line at once, also through `| tee`

from src.data import distribution, load_pool  # noqa: E402
from src.model import Detector  # noqa: E402
from src.train import train, trained  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train.yaml")
parser.add_argument("--resume", action="store_true", help="skip trained experts, go on with the unfinished one")
parser.add_argument("--only", default="", help="experts to train, comma-separated; default: all of them")
args = parser.parse_args()
cfg = yaml.safe_load(open(args.config))
d, m, t = cfg["data"], cfg["model"], cfg["train"]
out = Path(t["out"])
out.mkdir(parents=True, exist_ok=True)

parts = load_pool(d)
print(distribution(parts, m["families"]))  # images per generator
reals = {k: [r for r in rows if r["label"] == 0] for k, rows in parts.items()}
fakes = {k: [r for r in rows if r["label"] == 1] for k, rows in parts.items()}
for rows in (reals["val"], fakes["val"]):  # folder order -> mixed, the first val_size of them are used
    random.Random(0).shuffle(rows)
n_val = t.get("val_size", 5000) // 2

in_family = [g for models in m["families"].values() for g in models]
alone = sorted({r["generator"] for r in fakes["train"]} - set(in_family))  # no family: the general expert only
experts = {"all": in_family + alone, **m["families"]}
print(f"generators without a family, known to the general expert only: {', '.join(alone) or 'none'}")
print(f"whole images resized to {t['size']} x {t['size']}, batch {t['batch_size']}")

for name in args.only.split(",") if args.only else experts:
    if args.resume and trained(out, name, t["epochs"]):
        print(f"{name}: trained, skipped")
        continue
    number = {g: j for j, g in enumerate(experts[name], 1)}  # label: 0 real, j = the j-th model of the expert
    mine = [{**r, "label": number[r["generator"]]} for r in fakes["train"] if r["generator"] in number]
    val = [{**r, "label": number[r["generator"]]} for r in fakes["val"] if r["generator"] in number][:n_val]
    assert mine and val, f"{name}: no images of {experts[name]} in {d['dir']}/fake"
    val += reals["val"][: len(val)]
    if name == "all":
        model = Detector(m["backbone"], m["shared"], {"all": experts["all"]})  # pretrained DINOv3
    else:
        model = Detector.load(out, names=["all"])  # the trained trunk
        model.add(name, experts[name])             # a copy of the general expert
    print(f"expert {name}: {len(mine)} generated images of {len(number)} models and as many reals per epoch, val {len(val)}")
    train(model, name, reals["train"], mine, val, t, resume=args.resume)
