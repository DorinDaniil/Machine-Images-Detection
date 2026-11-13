"""python scripts/run_test.py --config configs/train.yaml

Whole images resized to the training view size, size x size: nothing is cut off.
Thresholds: one per expert, together exceeded by alpha of the validation reals; never fitted on test images.
An image is called generated when any expert is above its threshold.
Tests:
    test part            the 10 % of every real source and every generator that were never trained on
    SID_Set validation   another dataset: fully synthetic and locally edited images against its reals
Result: <out>/thresholds.json and <out>/report.md
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(line_buffering=True)  # every line at once, also through `| tee`

from src.data import ImageDataset, load_pool, sid_set  # noqa: E402
from src.evaluate import predict, report, thresholds, verdict  # noqa: E402
from src.model import Detector  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train.yaml")
cfg = yaml.safe_load(open(parser.parse_args().config))
d, t, test = cfg["data"], cfg["train"], cfg["test"]
out = Path(t["out"])
device = t["device"] if torch.cuda.is_available() else "cpu"
model = Detector.load(out, device)
names = list(model.experts)


def run(rows):
    return predict(model, ImageDataset(rows, t["size"]), device, t["batch_size"], t["num_workers"])


parts = load_pool(d)
calib = [r for r in parts["val"] if r["label"] == 0]
calib = random.Random(0).sample(calib, min(len(calib), test.get("max_calib", 10000)))
print(f"experts: {', '.join(names)}\nthresholds: {len(calib)} validation reals")
calib = run(calib)[0]
thr = thresholds(calib, test["alpha"])
json.dump(dict(zip(names, thr.round(6).tolist())), open(out / "thresholds.json", "w"), indent=1)

rows = parts["test"] + sid_set("validation", Path(d["dir"]) / "sid_validation", **d["sid"])
print(f"test: {len(rows)} images")
prob, index = run(rows)
group = np.array([r["generator"] or r["source"] for r in rows])  # a generator, or where a real comes from
real, sid = np.array([r["label"] == 0 for r in rows]), np.array([r["source"].startswith("sid/") for r in rows])
reals = {g: prob[group == g] for g in sorted(set(group[real]))}
fakes = [(g, prob[group == g], index[group == g]) for g in sorted(set(group[~real]))]

# do the family experts help? The same test with a part of the experts, each part with its own thresholds
subsets = {"all experts": names, "only the general one": ["all"], "only the families": [n for n in names if n != "all"]}
lines = ["| experts | FPR reals | FPR SID reals | TPR generators | TPR sid-full | precision |", "|---|---|---|---|---|---|"]
for title, part in subsets.items():
    cols = [names.index(n) for n in part]
    if cols:
        hit = verdict(prob[:, cols], thresholds(calib[:, cols], test["alpha"]))[0]
        tpr = np.mean([hit[group == g].mean() for g, _, _ in fakes if not g.startswith("sid-")])
        tp, fp = hit[~real & ~sid].sum(), hit[real & ~sid].sum()
        lines.append(f"| {title} | {hit[real & ~sid].mean():.3f} | {hit[real & sid].mean():.3f} | {tpr:.3f} | "
                     f"{hit[group == 'sid-full'].mean():.3f} | {tp / max(tp + fp, 1):.3f} |")

text = (f"thresholds for {test['alpha']:.0%} FPR on {len(calib)} validation reals: "
        f"{', '.join(f'{n} {v:.4f}' for n, v in zip(names, thr))}\n\n"
        f"Test part: {(real & ~sid).sum()} reals and {(~real & ~sid).sum()} generated images that were never trained on. "
        f"FPR reals and precision: on the test part as it is. TPR generators: mean over its generators.\n\n"
        + "\n".join(lines) + f"\n\n{report(model, thr, reals, fakes)}\n")
print(text)
(out / "report.md").write_text(text, encoding="utf-8")
