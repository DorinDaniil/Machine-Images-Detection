"""python scripts/run_test.py --config configs/train.yaml

Whole images resized to the training view size, size x size: nothing is cut off.
Threshold: exceeded by alpha of the calibration reals (held-out train reals + renders), never by test images.
Tests:
    OpenFake core/test   newer generators against DOCCI reals; generators also seen in training are marked
    SID_Set validation   fully synthetic and locally edited images (masks -> pixel F1) against SID reals
    renders              text pages, UI screens, charts, logos: false positives on non-photo reals
Result: runs/pilot/report.md
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

from src.data import ImageDataset, load_rows, openfake, sid_set  # noqa: E402
from src.evaluate import report, score, threshold_at  # noqa: E402
from src.model import Detector  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train.yaml")
cfg = yaml.safe_load(open(parser.parse_args().config))
d, t, test = cfg["data"], cfg["train"], cfg["test"]
data, out = Path(d["dir"]), Path(t["out"])
device = t["device"] if torch.cuda.is_available() else "cpu"
model = Detector.load(out / "best.pt", device)

calib_files = json.load(open(out / "calibration.json" if (out / "calibration.json").exists() else data / "calibration.json"))
calib = (calib_files.get("reals") or calib_files["openfake_reals"]) + calib_files["renders"]  # older runs: other key, in data/
calib = [{"path": p, "label": 0} for p in random.Random(0).sample(calib, min(len(calib), test.get("max_calib", 10000)))]
renders = [{"path": p, "label": 0} for p in calib_files["renders_test"]]
seen = {r["generator"] for name in ("openfake_train.csv", "sid_train.csv") if (data / name).exists()
        for r in load_rows(data / name)}

openfake_rows = openfake("test", data / "openfake_test", workers=d["workers"], **d["test"])
sid_rows = sid_set("validation", data / "sid_validation", **d["sid"])

def run(rows):
    return score(model, ImageDataset(rows, t["size"]), device, t["batch_size"], t["num_workers"])


thr = threshold_at(run(calib)[0], test["alpha"])
docci, _ = run([r for r in openfake_rows if r["label"] == 0])
sid_real, _ = run([r for r in sid_rows if r["label"] == 0])
reals = {"DOCCI (OpenFake test)": docci, "SID_Set reals": sid_real, "renders (non-photo)": run(renders)[0]}

fakes = []
for gen in sorted({r["generator"] for r in openfake_rows if r["label"] == 1}):
    s, f1 = run([r for r in openfake_rows if r["generator"] == gen])
    fakes.append((f"{gen}{' (in train)' if gen in seen else ''}", docci, s, f1))
for kind in ("sid-full", "sid-edit"):
    s, f1 = run([r for r in sid_rows if r["generator"] == kind])
    fakes.append((f"{kind}{' (in train)' if kind in seen else ''}", sid_real, s, f1))

unseen = [f for name, _, f, _ in fakes if not name.endswith("(in train)") and not name.startswith("sid")]
summary = f"unseen OpenFake generators: mean TPR {np.mean([(f > thr).mean() for f in unseen]):.3f}" if unseen else ""
text = (f"threshold {thr:.4f} for {test['alpha']:.0%} FPR on {len(calib)} calibration reals. {summary}\n\n"
        f"{report(thr, reals, fakes)}\n")
print(text)
(out / "report.md").write_text(text, encoding="utf-8")
