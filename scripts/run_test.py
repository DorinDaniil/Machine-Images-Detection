"""python scripts/run_test.py --config runs/base/config.yaml

The config of a run, written by run_train.py into its folder. Whole images resized to the training view size,
size x size: nothing is cut off. The model is used as it is: an image is called generated when
p(generated) > 0.5, and the family is the largest of the other classes.
Tests:
    test part            the 10 % of every real source and every generator that were never trained on
    SID_Set validation   another dataset: fully synthetic and locally edited images against its reals
Result: <out>/report.md and report.json. At the end the tested runs of the folder next to <out> are put side by
side: <runs>/compare.md.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(line_buffering=True)  # every line at once, also through `| tee`

from src.data import ImageDataset, load_pool, sid_set  # noqa: E402
from src.evaluate import compare, predict, report, threshold  # noqa: E402
from src.model import Detector  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train.yaml")
cfg = yaml.safe_load(open(parser.parse_args().config))
d, t = cfg["data"], cfg["train"]
out = Path(t["out"])
device = t["device"] if torch.cuda.is_available() else "cpu"
model = Detector.load(out / "model.pt", device)
classes = np.array(model.classes)
family_of = {g: name for name, generators in cfg["model"]["families"].items() for g in generators}


def run(rows):
    return predict(model, ImageDataset(rows, t["size"]), device, t["batch_size"], t["num_workers"])


parts = load_pool(d)
calib = [r for r in parts["val"] if r["label"] == 0]
calib = random.Random(0).sample(calib, min(len(calib), t["max_calib"]))  # the reals the model was calibrated on
print(f"classes: real, {', '.join(classes)}\nvalidation reals: {len(calib)} images")
calib = run(calib)[0]

rows = parts["test"] + sid_set("validation", Path(d["dir"]) / "sid_validation", **d["sid"])
print(f"test: {len(rows)} images")
odds, index = run(rows)
group = np.array([r["generator"] or r["source"] for r in rows])  # a generator, or where a real comes from
real, sid = np.array([r["label"] == 0 for r in rows]), np.array([r["source"].startswith("sid/") for r in rows])
truth = np.array(["" if r["label"] == 0 else family_of.get(r["generator"], r["generator"]) for r in rows])
truth[~real & ~np.isin(truth, classes)] = "?"  # a generator the model has no class for
named = classes[index]
generators = sorted(set(group[~real & ~sid]))


def totals(hit):
    tp, fp = hit[~real & ~sid].sum(), hit[real & ~sid].sum()
    return {"FPR reals": hit[real & ~sid].mean(), "FPR SID reals": hit[real & sid].mean(),
            "TPR generators": np.mean([hit[group == g].mean() for g in generators]),
            "TPR sid-full": hit[group == "sid-full"].mean(), "precision": tp / max(tp + fp, 1)}


# the model as it is, and what a lower or a higher share of false alarms would cost
lines = ["| generated when | FPR reals | FPR SID reals | TPR generators | TPR sid-full | precision |", "|---|---|---|---|---|---|"]
cuts = {"p > 0.5, the model as it is": 0.0}
for alpha in (0.001, 0.005, 0.01, 0.02, 0.05):  # thresholds from the validation reals, never from test images
    cut = threshold(calib, alpha)
    cuts[f"p > {1 / (1 + math.exp(-cut)):.4f}, target FPR {alpha:.1%}"] = cut
for name, cut in cuts.items():
    lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in totals(odds > cut).values()) + " |")

hit, known = odds > 0, ~real & ~sid & (truth != "?")
tables, numbers = report(hit, named, group, truth)
numbers = {**{k: float(v) for k, v in totals(hit).items()}, "family named right": float((named[known] == truth[known]).mean()), **numbers}
text = (f"The model as it is: an image is called generated when p(generated) > 0.5. {(calib > 0).mean():.1%} of "
        f"{len(calib)} validation reals are above; the bias of the class \"real\" was set for {t['alpha']:.1%} after training.\n\n"
        f"Test part: {(real & ~sid).sum()} reals and {(~real & ~sid).sum()} generated images that were never trained on. "
        f"FPR reals and precision: on the test part as it is. TPR generators: mean over its generators. "
        f"The family is named right for {numbers['family named right']:.1%} of its generated images.\n\n"
        + "\n".join(lines) + f"\n\n{tables}\n")
print(text)
(out / "report.md").write_text(text, encoding="utf-8")
json.dump(numbers, open(out / "report.json", "w"), indent=1)

table = compare(out.parent)  # every tested run of the folder, side by side
print(f"runs of {out.parent}:\n\n{table}")
(out.parent / "compare.md").write_text(table + "\n", encoding="utf-8")
