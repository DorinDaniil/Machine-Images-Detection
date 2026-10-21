"""python scripts/run_test.py --config configs/train.yaml

Threshold: the score exceeded by alpha of the CALIBRATION reals (held out, never used for metrics).
Test images:
    real photos / real renders   held-out test reals; their FPR at the threshold is measured
    <decoder> full / local       the same test reals re-decoded by test_decoders (local: one region, mask kept)
    <generator>                  images of closed generators from Qwen/Qwen-Image-Bench
Protocols: "equalized" (same resize, crop and JPEG for every image) and "native". Result: runs/pilot/report.md
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ImageDataset, blend, download_fakes, family, make_pairs, random_mask  # noqa: E402
from src.data.decoders import find  # noqa: E402
from src.evaluate import report, score, threshold_at  # noqa: E402
from src.model import TraceDetector  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train.yaml")
cfg = yaml.safe_load(open(parser.parse_args().config))
d, t, test = cfg["data"], cfg["train"], cfg["test"]
data, out = Path(d["dir"]), Path(t["out"])
device = t["device"] if torch.cuda.is_available() else "cpu"
model = TraceDetector.load(out / "best.pt", device)
split = json.load(open(data / "split.json"))


def items(paths, label=0):
    return [{"path": p, "label": label} for p in paths]


calib = items(split["photos"]["calib"] + split["renders"]["calib"])
reals = {"real photos": items(split["photos"]["test"]), "real renders": items(split["renders"]["test"])}

# held-out reals re-decoded by test decoders: full copy and one local region (mask saved)
fakes = {}
test_reals = split["photos"]["test"] + split["renders"]["test"]
pairs = make_pairs(test_reals, d["test_decoders"], data / "pairs_test", device, d["max_side"])
rng = np.random.default_rng(0)
for dec in d["test_decoders"]:
    name = f"{dec} ({family(dec)}, {'seen' if family(dec) in {family(x) for x in d['train_decoders']} else 'unseen'})"
    fakes[f"{name} full"] = items([find(p, dec) for p in pairs], 1)
    fakes[f"{name} local"] = []
    for p in pairs:
        real = np.asarray(Image.open(find(p, "real")).convert("RGB"), dtype=np.float32)
        fake = np.asarray(Image.open(find(p, dec)).convert("RGB"), dtype=np.float32)
        mask = random_mask(*real.shape[:2], rng)
        Image.fromarray(blend(real, fake, mask)).save(p / f"local_{dec}.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(p / f"local_{dec}_mask.png")
        fakes[f"{name} local"].append({"path": p / f"local_{dec}.png", "label": 1, "mask": p / f"local_{dec}_mask.png"})

# closed generators
for gen, paths in download_fakes(data / "qwen_image_bench", d["generators"], d["n_per_generator"]).items():
    fakes[gen] = items(paths, 1)

# score under each protocol
tables = []
for protocol in test["protocols"]:
    run = lambda its: score(model, ImageDataset(its, model.patch_size, protocol, d["max_side"]), device)  # noqa: E731
    calib_scores, _ = run(calib)
    real_scores = {name: run(its)[0] for name, its in reals.items()}
    fake_results = {name: run(its) for name, its in fakes.items()}
    thr = threshold_at(calib_scores, test["alpha"])
    fpr = "\n".join(f"measured FPR, {name} ({len(s)}): {(s > thr).mean():.3f}" for name, s in real_scores.items())
    table = report(calib_scores, np.concatenate(list(real_scores.values())), fake_results, test["alpha"])
    tables.append(f"## protocol: {protocol}\n\n{fpr}\n\n{table}\n")
    print(tables[-1])

(out / "report.md").write_text("\n".join(tables), encoding="utf-8")
