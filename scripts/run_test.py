"""python scripts/run_test.py --config configs/train.yaml

Whole images resized to the training view size, size x size: nothing is cut off.
Thresholds: one per expert, together exceeded by alpha of the calibration reals (held-out train reals + renders),
never fitted on test images. An image is called generated when any expert is above its threshold.
Tests:
    OpenFake core/test   newer generators against DOCCI reals; generators also seen in training are marked
    SID_Set validation   fully synthetic and locally edited images against SID reals
    renders              text pages, UI screens, charts, logos: false positives on non-photo reals
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

from src.data import ImageDataset, openfake, sid_set  # noqa: E402
from src.evaluate import predict, report, thresholds, verdict  # noqa: E402
from src.model import Detector  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train.yaml")
cfg = yaml.safe_load(open(parser.parse_args().config))
d, t, test = cfg["data"], cfg["train"], cfg["test"]
data, out = Path(d["dir"]), Path(t["out"])
device = t["device"] if torch.cuda.is_available() else "cpu"
model = Detector.load(out, device)
names, seen = list(model.experts), set(model.experts["all"].models)

calib_files = json.load(open(out / "calibration.json"))
calib = calib_files["reals"] + calib_files["renders"]
calib = [{"path": p, "label": 0} for p in random.Random(0).sample(calib, min(len(calib), test.get("max_calib", 10000)))]
openfake_rows = openfake("test", data / "openfake_test", workers=d["workers"], **d["test"])
sid_rows = sid_set("validation", data / "sid_validation", **d["sid"])


def run(rows):
    return predict(model, ImageDataset(rows, t["size"]), device, t["batch_size"], t["num_workers"])


calib = run(calib)[0]
thr = thresholds(calib, test["alpha"])
json.dump(dict(zip(names, thr.round(6).tolist())), open(out / "thresholds.json", "w"), indent=1)

docci = run([r for r in openfake_rows if r["label"] == 0])[0]
renders = run([{"path": p, "label": 0} for p in calib_files["renders_test"]])[0]
reals = {"DOCCI (OpenFake test)": docci, "SID_Set reals": run([r for r in sid_rows if r["label"] == 0])[0],
         "renders (non-photo)": renders}
fakes = [(gen, *run([r for r in openfake_rows if r["generator"] == gen]))
         for gen in sorted({r["generator"] for r in openfake_rows if r["label"] == 1})]
sid = [(kind, *run([r for r in sid_rows if r["generator"] == kind])) for kind in ("sid-full", "sid-edit")]

# do the family experts help? The same test with a part of the experts, each part with its own thresholds
parts = {"all experts": names, "only the general one": ["all"], "only the families": [n for n in names if n != "all"]}
lines = ["| experts | FPR DOCCI | FPR renders | TPR in train | TPR unseen | precision |", "|---|---|---|---|---|---|"]
for title, part in parts.items():
    cols = [names.index(n) for n in part]
    if cols:
        part_thr = thresholds(calib[:, cols], test["alpha"])
        hit = lambda prob: verdict(prob[:, cols], part_thr)[0]  # noqa: E731
        tpr = lambda groups: np.mean(groups) if groups else float("nan")  # noqa: E731
        tp, fp = sum(hit(prob).sum() for _, prob, _ in fakes), hit(docci).sum()
        lines.append(f"| {title} | {hit(docci).mean():.3f} | {hit(renders).mean():.3f} | "
                     f"{tpr([hit(prob).mean() for g, prob, _ in fakes if g in seen]):.3f} | "
                     f"{tpr([hit(prob).mean() for g, prob, _ in fakes if g not in seen]):.3f} | {tp / max(tp + fp, 1):.3f} |")

text = (f"thresholds for {test['alpha']:.0%} FPR on {len(calib)} calibration reals: "
        f"{', '.join(f'{n} {v:.4f}' for n, v in zip(names, thr))}\n\n"
        f"TPR: mean over the OpenFake test generators seen in training and over the unseen ones. "
        f"Precision: OpenFake test as it is, {sum(len(p) for _, p, _ in fakes)} generated and {len(docci)} DOCCI images.\n\n"
        + "\n".join(lines) + f"\n\n{report(model, thr, reals, fakes + sid)}\n")
print(text)
(out / "report.md").write_text(text, encoding="utf-8")
