"""python scripts/run_test.py --config configs/train.yaml

Whole images resized to the training view size, size x size: nothing is cut off.
The general expert decides: an image is called generated when its log-odds are above one threshold, the value
that test.alpha of the validation reals exceed; never fitted on test images. For a generated image the family
of the generator is named: the probabilities of the models of a family (model.families) are summed.
Tests:
    test part            the 10 % of every real source and every generator that were never trained on
    SID_Set validation   another dataset: fully synthetic and locally edited images against its reals
Result: <out>/thresholds.json and <out>/report.md
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
from src.evaluate import groups, name_family, predict, report, threshold  # noqa: E402
from src.model import Detector  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train.yaml")
cfg = yaml.safe_load(open(parser.parse_args().config))
d, m, t, test = cfg["data"], cfg["model"], cfg["train"], cfg["test"]
out = Path(t["out"])
device = t["device"] if torch.cuda.is_available() else "cpu"
model = Detector.load(out, device, names=["all"])
family_of = {g: name for name, generators in m["families"].items() for g in generators}
cols = groups(model.experts["all"].models, family_of)  # family -> the classes of the general expert


def run(rows):
    return predict(model, ImageDataset(rows, t["size"]), device, t["batch_size"], t["num_workers"])


parts = load_pool(d)
calib = [r for r in parts["val"] if r["label"] == 0]
calib = random.Random(0).sample(calib, min(len(calib), test.get("max_calib", 10000)))
print(f"families: {', '.join(cols)}\nthreshold: {len(calib)} validation reals")
calib = run(calib)[0]
thr = threshold(calib, test["alpha"])
json.dump({"alpha": test["alpha"], "threshold": thr}, open(out / "thresholds.json", "w"), indent=1)

rows = parts["test"] + sid_set("validation", Path(d["dir"]) / "sid_validation", **d["sid"])
print(f"test: {len(rows)} images")
odds, share = run(rows)
group = np.array([r["generator"] or r["source"] for r in rows])  # a generator, or where a real comes from
real, sid = np.array([r["label"] == 0 for r in rows]), np.array([r["source"].startswith("sid/") for r in rows])
truth = np.array(["" if r["label"] == 0 else family_of.get(r["generator"], r["generator"]) for r in rows])
truth[~real & ~np.isin(truth, list(cols))] = "?"  # a generator the model was not trained on

# the same test at other thresholds: what a lower or a higher share of false alarms would cost
generators = sorted(set(group[~real & ~sid]))
lines = ["| target FPR | threshold as probability | FPR reals | FPR SID reals | TPR generators | TPR sid-full | precision |",
         "|---|---|---|---|---|---|---|"]
for alpha in sorted({0.001, 0.005, 0.01, 0.02, 0.05, test["alpha"]}):
    at = threshold(calib, alpha)
    hit = odds > at
    tp, fp = hit[~real & ~sid].sum(), hit[real & ~sid].sum()
    lines.append(f"| {alpha:.1%}{' (used)' if alpha == test['alpha'] else ''} | {1 / (1 + math.exp(-at)):.5f} | "
                 f"{hit[real & ~sid].mean():.3f} | {hit[real & sid].mean():.3f} | "
                 f"{np.mean([hit[group == g].mean() for g in generators]):.3f} | "
                 f"{hit[group == 'sid-full'].mean():.3f} | {tp / max(tp + fp, 1):.3f} |")

text = (f"The general expert, one threshold: log-odds {thr:.3f}, exceeded by {test['alpha']:.1%} of {len(calib)} "
        f"validation reals. Score of an image = sigmoid(log-odds - threshold): above 0.5 it is called generated.\n\n"
        f"Test part: {(real & ~sid).sum()} reals and {(~real & ~sid).sum()} generated images that were never trained on. "
        f"FPR reals and precision: on the test part as it is. TPR generators: mean over its generators.\n\n"
        + "\n".join(lines) + f"\n\n{report(odds > thr, name_family(share, cols), group, truth)}\n")
print(text)
(out / "report.md").write_text(text, encoding="utf-8")
