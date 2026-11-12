"""python scripts/download_data.py --config configs/train.yaml

Downloads once everything training and testing need: OpenFake train and test, SID_Set validation.
OpenFake files are fetched whole, filtered by the quotas in the config and deleted; the kept images go
straight to the folders training reads:
    data/real/<source>/        laion, docci, ...
    data/fake/<generator>/     gpt-image-1, flux.1-dev, ...
Stop it any time (Ctrl+C) and run it again: finished files are skipped. run_train.py and run_test.py read
these folders and download nothing. Put your own folders with real images into data/real/, or list them
in data.real_dirs. At the end the number of images of every generator is printed.
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(line_buffering=True)  # every line at once, also through `| tee`

from src.data import distribution, load_pool, openfake, sid_set  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train.yaml")
cfg = yaml.safe_load(open(parser.parse_args().config))
d = cfg["data"]
data = Path(d["dir"])
data.mkdir(parents=True, exist_ok=True)

openfake("train", data, workers=d["workers"], **d["train"])
openfake("test", data, workers=d["workers"], **d["test"])
sid_set("validation", data / "sid_validation", **d["sid"])

used = sum(f.stat().st_size for f in data.rglob("*") if f.is_file()) / 1e9
print(f"done: {used:.0f} GB in {data}, {shutil.disk_usage(data).free / 1e9:.0f} GB free")
print(distribution(load_pool(d), cfg["model"]["families"]))  # images per generator
