"""python scripts/download_data.py --config configs/train.yaml

Downloads once everything training and testing need: OpenFake train and test, SID_Set validation.
OpenFake files are fetched whole, filtered by the quotas in the config and deleted, so only the kept
images stay on disk. Stop it any time (Ctrl+C) and run it again: finished files are skipped.
Then the OpenFake images are sorted into folders (a rename, nothing is copied):
    data/real/<source>/        laion, docci, ...
    data/fake/<generator>/     gpt-image-1, flux.1-dev, ...
run_train.py and run_test.py read these folders and download nothing. Put your own folders with real
images into data/real/, or list them in data.real_dirs.
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(line_buffering=True)  # every line at once, also through `| tee`

from src.data import openfake, sid_set, sort_folders  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train.yaml")
d = yaml.safe_load(open(parser.parse_args().config))["data"]
data = Path(d["dir"])
data.mkdir(parents=True, exist_ok=True)

openfake("train", data / "openfake_train", workers=d["workers"], **d["train"])
openfake("test", data / "openfake_test", workers=d["workers"], **d["test"])
sid_set("validation", data / "sid_validation", **d["sid"])
sort_folders(data)

used = sum(f.stat().st_size for f in data.rglob("*") if f.is_file()) / 1e9
print(f"done: {used:.0f} GB in {data}, {shutil.disk_usage(data).free / 1e9:.0f} GB free")
