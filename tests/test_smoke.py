"""CPU check of the whole chain on synthetic images, no downloads. python -m pytest tests -q"""

import io
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from src.data import ImageDataset, load_pool, load_rows, render_folder, save_rows
from src.data.download import _store
from src.evaluate import predict, report, thresholds, verdict
from src.model import Detector, score
from src.train import stats, train, trained

CFG = dict(device="cpu", size=96, batch_size=2, num_workers=0, lr=1e-3, head_lr=1e-2, weight_decay=0.0, epochs=2)


def make_rows(root, n=8, size=200):
    """Reals (label 0) and fakes of two 'generators' (smoother images), labels 1 and 2."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        fake, gen = i % 2, i % 4 // 2
        noise = rng.integers(0, 255, (8 if fake else 32, 8 if fake else 32, 3), dtype=np.uint8)
        path = root / f"{i}.jpg"
        w, h = (size, size) if i < 2 else (size + 24 * (i % 3), size - 16 * (i % 2))  # sizes differ, as in OpenFake
        Image.fromarray(noise).resize((w, h), Image.BICUBIC).save(path, quality=95)
        rows.append({"path": str(path), "label": 1 + gen if fake else 0, "generator": f"gen{gen}" if fake else "",
                     "mask": "", "source": "toy"})
    return rows


def test_score_and_thresholds():
    """One softmax per expert: "generated" is everything but the class "real". Every expert has its own threshold."""
    logits = torch.tensor([[4.0, 0.0, 0.0], [0.0, 1.0, 3.0]])
    prob, index = score(logits.bfloat16())
    assert prob.dtype == torch.float32 and index.tolist() == [0, 1]
    assert prob[0].item() == pytest.approx(2 / (math.e ** 4 + 2), abs=1e-3)

    p = 1 - torch.softmax(logits, dim=1)[:, 0]  # the loss = "generated or real" + "which model" for the generated
    by_hand = (-torch.log(1 - p[0]) - torch.log(p[1]) + F.cross_entropy(logits[1:, 1:], torch.tensor([1]))) / 2
    assert F.cross_entropy(logits, torch.tensor([0, 2])).item() == pytest.approx(by_hand.item(), abs=1e-5)

    four = stats(torch.tensor([[0.0, 3.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 3.0], [3.0, 0.0, 0.0]]), torch.tensor([1, 0, 0, 2]))
    assert {k: four[k] for k in ("acc", "prec", "rec", "fpr", "model_acc")} == dict(acc=0.5, prec=0.5, rec=0.5, fpr=0.5, model_acc=0.5)

    real = np.random.default_rng(0).random((1000, 3)) * [1.0, 0.5, 0.1]  # three experts, different ranges on reals
    thr = thresholds(real, 0.05)
    assert 0.03 < verdict(real, thr)[0].mean() <= 0.05  # together: at most alpha of the reals
    assert np.ptp((real > thr).mean(axis=0)) < 0.005    # and every expert has the same share of them
    hit, winner = verdict(np.array([[0.2, 0.2, 0.2], [0.0, 0.0, 0.0]]), thr)  # 0.2 is high for the third expert only
    assert hit.tolist() == [True, False] and winner[0] == 2
    assert thresholds(np.array([[0.1], [0.5], [0.9], [0.3]]), 0.5).tolist() == [0.5]


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))


def test_pipeline(tmp_path):
    """The general expert with the trunk, then a family expert on the frozen trunk; save, load, predict, report."""
    import copy

    rows = make_rows(tmp_path, n=16)
    save_rows(rows, tmp_path / "rows.csv")
    assert load_rows(tmp_path / "rows.csv")[3]["label"] == 2
    reals, fakes = [r for r in rows if r["label"] == 0], [r for r in rows if r["label"]]
    cfg, run = dict(CFG, out=str(tmp_path / "run")), tmp_path / "run"

    model = Detector("tiny", 4, {"all": ["gen0", "gen1"]})
    image, label = ImageDataset(rows, 96)[1]
    assert image.shape == (3, 96, 96) and label == 1
    assert model(image[None])["all"].shape == (1, 3) and model.features(image[None]).shape == (1, 64)
    whole = Detector("tiny", 0, {"all": ["gen0"]})  # shared 0: nothing in the trunk, the expert is the whole network
    assert not list(whole.trunk.parameters()) and whole(image[None])["all"].shape == (1, 2)

    start = copy.deepcopy(model.trunk)
    train(model, "all", reals[2:], fakes[2:], rows[:4], cfg)
    assert not same(start, model.trunk)  # the general expert is trained together with the trunk
    assert trained(run, "all", 2) and not trained(run, "all", 3) and not trained(run, "one", 2)

    model = Detector.load(run, names=["all"])
    model.add("one", ["gen1"])  # a family expert: a copy of the general one with its own head
    assert same(model.experts["one"].stages, model.experts["all"].stages) and model.experts["one"].head.out_features == 2
    before = copy.deepcopy(model)
    gen1 = [{**r, "label": 1} for r in fakes if r["generator"] == "gen1"]
    train(model, "one", reals[2:], gen1[1:], gen1[:1] + reals[:1], cfg)
    assert same(before.trunk, model.trunk) and same(before.experts["all"], model.experts["all"])  # only the expert learns
    assert not same(before.experts["one"], model.experts["one"])

    assert sorted(p.name for p in (run / "experts").iterdir()) == ["all.pt", "one.pt"] and (run / "trunk.pt").exists()
    loaded = Detector.load(run)
    assert list(loaded.experts) == ["all", "one"] and loaded.experts["one"].models == ["gen1"] and loaded.config == model.config
    log = (run / "metrics" / "one.csv").read_text().splitlines()
    val_loss = [float(line.split(",")[log[0].split(",").index("val_loss")]) for line in log[1:]]
    last = torch.load(run / "state.pt", weights_only=False)
    kept = all(torch.equal(v, last["expert"][k]) for k, v in loaded.experts["one"].state_dict().items())
    assert last["name"] == "one" and kept == (val_loss[1] <= val_loss[0])  # saved = the epoch with the lowest validation loss

    prob, index = predict(loaded, ImageDataset(rows, 96), "cpu", batch_size=4, workers=0)
    assert prob.shape == (16, 2) and ((0 < prob) & (prob < 1)).all() and (index[:, 1] == 0).all() and index[:, 0].max() <= 1
    real = np.array([r["label"] == 0 for r in rows])
    thr = thresholds(prob[real], 0.2)
    table = report(loaded, thr, {"reals": prob[real]}, [("gen0", prob[~real], index[~real]), ("new", prob[~real], index[~real])])
    seen, unseen = ([c.strip() for c in line.split("|")] for line in table.splitlines()[-2:])
    assert seen[1:3] == ["gen0", "8"] and seen[4] in ("all", "one") and seen[5] in ("gen0", "gen1") and seen[6]
    assert unseen[1:3] == ["new", "8"] and unseen[3:6] == seen[3:6] and unseen[6] == ""  # no true name to compare with


def test_train_resume(tmp_path, monkeypatch):
    """A crash in the middle of epoch 2; resume goes on from the state saved after epoch 1."""
    import src.train

    class Crash(ImageDataset):  # fails on the `at`-th training image, like a power cut
        reads, at = 0, 10

        def __getitem__(self, i):
            Crash.reads += self.train
            if self.train and Crash.reads == Crash.at:
                raise RuntimeError("power cut")
            return super().__getitem__(i)

    monkeypatch.setattr(src.train, "ImageDataset", Crash)
    rows = make_rows(tmp_path)
    reals, fakes = [r for r in rows if r["label"] == 0], [r for r in rows if r["label"]]
    cfg, run = dict(CFG, out=str(tmp_path / "run")), tmp_path / "run"
    with pytest.raises(RuntimeError, match="power cut"):  # 8 images of epoch 1, then the 2nd of epoch 2
        train(Detector("tiny", 4, {"all": ["gen0", "gen1"]}), "all", reals, fakes, rows[:4], cfg)
    state = torch.load(run / "state.pt", weights_only=False)
    assert state["name"] == "all" and state["epoch"] == 2 and len(state["log"]) == 1 and state["scheduler"]["last_epoch"] == 4
    assert not trained(run, "all", 2)

    Crash.at = 0
    resumed = Detector("tiny", 4, {"all": ["gen0", "gen1"]})  # as a new process: weights come from the state
    train(resumed, "all", reals, fakes, rows[:4], cfg, resume=True)
    state = torch.load(run / "state.pt", weights_only=False)
    assert state["epoch"] == 3 and state["scheduler"]["last_epoch"] == 8 and trained(run, "all", 2)
    epochs = [line.split(",")[0] for line in (run / "metrics" / "all.csv").read_text().splitlines()[1:]]
    assert epochs == ["1", "2"]
    assert all(torch.equal(v, state["expert"][k]) for k, v in resumed.experts["all"].state_dict().items())


def test_views(tmp_path):
    """Test views are the whole image resized to size x size and never change; training views are random
    parts of the image whose shape does not depend on the shape of the image."""
    from src.data.dataset import random_crop

    rows = make_rows(tmp_path)
    assert all(ImageDataset(rows, 160, train=True)[i][0].shape == (3, 160, 160) for i in range(8))
    val = ImageDataset(rows, 160)
    assert torch.equal(val[2][0], val[2][0]) and val[0][1] == 0 and val[3][1] == 2

    halves = np.zeros((300, 900, 3), np.uint8)
    halves[:, 450:] = 255  # a wide image, dark left half, bright right half
    Image.fromarray(halves).save(tmp_path / "halves.png")
    item = {"path": str(tmp_path / "halves.png"), "label": 0}
    view = ImageDataset([item], 128)[0][0]
    row = view.mean(0)[64]
    assert view.shape == (3, 128, 128) and row[:60].max() < 0.1 and row[68:].min() > 0.9  # squeezed, not cut

    ds, bright = ImageDataset([item], 128, train=True), []
    for ds.epoch in range(20):
        bright.append((ds[0][0].mean(0) > 0.5).float().mean().item())
    assert (abs(np.array(bright) - 0.5) > 0.05).mean() > 0.5  # parts of the image, not the whole of it

    rng = np.random.default_rng(0)
    for size in ((1024, 1024), (1500, 1000)):  # a square generation and a 3:2 photo
        crops = [random_crop(Image.new("RGB", size), rng).size for _ in range(300)]
        aspect, area = np.array([w / h for w, h in crops]), np.array([w * h for w, h in crops]) / (size[0] * size[1])
        assert aspect.min() < 0.7 and aspect.max() > 1.4 and area.min() > 0.35 and area.max() <= 1

    from src.data.dataset import degrade

    photo = Image.fromarray(np.random.default_rng(1).integers(0, 255, (90, 120, 3), dtype=np.uint8))
    changed = []
    for seed in range(60):  # colour, sharpening, censoring, resampling, blur, noise, compression: the size is kept
        out = degrade(photo.copy(), np.random.default_rng(seed))
        assert out.size == photo.size and out.mode == "RGB"
        changed.append(np.abs(np.asarray(out, np.float32) - np.asarray(photo, np.float32)).mean())
    assert min(changed) > 0 and max(changed) > 3 * min(changed)  # always compressed, some views much more changed

    from src.data.dataset import censor

    pixels, share = np.asarray(photo).copy(), []
    for seed in range(100):  # blur, big pixels, a box or a bar over one region; the rest and the input stay as they were
        out = censor(photo, np.random.default_rng(seed))
        assert out.size == photo.size and np.array_equal(np.asarray(photo), pixels)
        share.append((np.asarray(out) != pixels).any(axis=2).mean())
    assert 0 < min(share) and max(share) < 0.3 and 0.03 < np.median(share) < 0.15

    (tmp_path / "broken.jpg").write_bytes(b"not an image")  # reported and replaced by the next image
    ds = ImageDataset([{"path": str(tmp_path / "broken.jpg"), "label": 1}, item], 64)
    image, label = ds[0]
    assert label == 0 and torch.equal(image, ds[1][0])


def test_folders_and_split(tmp_path):
    """data/real/<source>/ and data/fake/<generator>/ and the real folders of the user are read through their
    subfolders; the part of an image depends only on "<folder>/<file>"."""
    from src.data.pool import part

    data, blank = tmp_path / "data", Image.new("RGB", (40, 40))
    for folder, n in (("real/laion", 30), ("real/docci", 10), ("fake/gen-a", 40)):
        (data / folder).mkdir(parents=True)
        for i in range(n):
            blank.save(data / folder / f"train_{i:04d}.png")
    blank.save(data / "fake" / "stray.png")  # not in a generator folder: ignored

    mine = tmp_path / "my_reals"
    (mine / "logos").mkdir(parents=True)
    (mine / "scans" / "2024").mkdir(parents=True)
    for i in range(10):
        blank.save(mine / "logos" / f"{i}.jpg")
        blank.save(mine / "scans" / "2024" / f"{i}.PNG")
    blank.save(mine / "top.bmp")
    (mine / "notes.txt").write_text("not an image")
    d = {"dir": str(data), "real_dirs": [str(mine)], "val_share": 0.2, "test_share": 0.2, "n_render": 4, "render_train": 0.0}
    parts = load_pool(d)
    rows = [r for part_rows in parts.values() for r in part_rows]
    sources = Counter(r["source"] for r in rows if r["label"] == 0)
    assert {s: n for s, n in sources.items() if not s.startswith("render/")} == {
        "laion": 30, "docci": 10, "my_reals/logos": 10, "my_reals/scans": 10, "my_reals": 1}
    assert sum(n for s, n in sources.items() if s.startswith("render/")) == 4  # test fonts: 2 in val, 2 in test
    assert Counter(r["generator"] for r in rows if r["label"]) == {"gen-a": 40}
    assert len({r["path"] for r in rows}) == len(rows) == 105  # every image in exactly one part
    assert 10 <= len(parts["val"]) <= 32 and 10 <= len(parts["test"]) <= 32  # about a fifth each

    where = {r["path"]: name for name, part_rows in parts.items() for r in part_rows}
    for i in range(10, 30):
        blank.save(mine / "logos" / f"{i}.jpg")  # more images later: the old ones stay where they were
    again = {r["path"]: name for name, part_rows in load_pool(d).items() for r in part_rows}
    assert len(again) == 125 and all(again[p] == name for p, name in where.items())
    assert part("/disk1/data/real/laion/7.jpg", 0.2, 0.2) == part("/another/place/laion/7.jpg", 0.2, 0.2)  # data moved


def test_migrate_old_layout(tmp_path):
    """scripts/migrate_data.py (one-off, to be deleted with this test): the flat folders of the old downloader
    become what the new downloader writes, in the csv lists and in the records of finished files alike."""
    import subprocess
    import sys

    import yaml

    data, blank = tmp_path / "data", Image.new("RGB", (40, 40))
    specs = {"train": [(0, "", "openfake/laion")] * 6 + [(1, "gen-a", "openfake")] * 6,
             "test": [(0, "", "openfake/docci")] * 4 + [(1, "gen-a", "openfake")] * 4}
    for split, spec in specs.items():
        folder = data / f"openfake_{split}"
        (folder / "done").mkdir(parents=True)
        rows = []
        for i, (label, generator, source) in enumerate(spec):
            blank.save(folder / f"0000_{i:05d}.png")
            rows.append({"path": str(folder / f"0000_{i:05d}.png"), "label": label, "generator": generator, "mask": "",
                         "source": source, "prompt": f"prompt {i}", "prompt_kind": "", "row_id": f"row-{i}"})
        (folder / "done" / "0000.json").write_text(json.dumps({"read": len(rows), "rows": rows}))
        if split == "train":  # the test split was still downloading: records only, no list yet
            save_rows(rows, data / "openfake_train.csv")
    (tmp_path / "tiny.yaml").write_text(yaml.safe_dump({"data": {"dir": "data"}}))

    def migrate():
        done = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "migrate_data.py"),
                               "--config", "tiny.yaml"], cwd=tmp_path, capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
        return done.stdout

    log = migrate()
    assert "20 images moved" in log and "0 of them not found, 0 files left" in log
    names = lambda folder: sorted(p.name for p in (data / folder).iterdir())  # noqa: E731
    assert names("fake/gen-a") == [f"test_0000_{i:05d}.png" for i in range(4, 8)] + [f"train_0000_{i:05d}.png" for i in range(6, 12)]
    assert names("real/laion") == [f"train_0000_{i:05d}.png" for i in range(6)] and len(names("real/docci")) == 4
    rows = load_rows(data / "openfake_train.csv")
    record = json.loads((data / "openfake_train" / "done" / "0000.json").read_text())
    assert [r["path"] for r in rows] == [r["path"] for r in record["rows"]] and record["read"] == 12
    assert all((tmp_path / r["path"]).exists() for r in rows) and rows[7]["prompt"] == "prompt 7" and rows[7]["row_id"] == "row-7"
    assert Path(rows[7]["path"]).parts[-3:] == ("fake", "gen-a", "train_0000_00007.png")
    record = json.loads((data / "openfake_test" / "done" / "0000.json").read_text())
    assert all((tmp_path / r["path"]).exists() for r in record["rows"]) and not (data / "openfake_test.csv").exists()

    assert "0 images moved" in migrate()  # again: nothing moves, nothing is renamed twice
    assert load_rows(data / "openfake_train.csv") == rows and len(names("fake/gen-a")) == 10


def test_compression_history(tmp_path):
    """A file not stored as JPEG gets one JPEG pass before anything else; a JPEG file is read as it is."""
    from src.data.dataset import load

    pixels = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path / "fake.png")
    Image.fromarray(pixels).save(tmp_path / "real.jpg", quality=90)
    assert not np.array_equal(np.asarray(load(tmp_path / "fake.png")), pixels)  # lossy pass for the PNG
    with Image.open(tmp_path / "real.jpg") as im:
        assert np.array_equal(np.asarray(load(tmp_path / "real.jpg")), np.asarray(im.convert("RGB")))
    a, b = load(tmp_path / "fake.png", np.random.default_rng(1)), load(tmp_path / "fake.png", np.random.default_rng(2))
    assert a.size == b.size == (64, 64)


def test_sid_names_unique(tmp_path, monkeypatch):
    """Rows skipped in one file must not make the next file overwrite stored images."""
    from src.data import download

    def png(color, mode="RGB"):
        buf = io.BytesIO()
        Image.new(mode, (520, 520), color).save(buf, format="PNG")
        return {"bytes": buf.getvalue()}

    colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0)}
    row = lambda label: {"image": png(colors[label]), "label": label, "mask": png(255, "L") if label == 2 else None}  # noqa: E731
    no_mask = dict(row(2), mask=None)
    files = {"data/validation-0": [row(0), row(0), row(0), row(0), no_mask, row(2)], "data/validation-1": [row(1)] * 3 + [row(2)] * 3}
    monkeypatch.setattr(download, "_files", lambda repo, prefix: list(files))
    monkeypatch.setattr(download, "_rows", lambda repo, name, tmp, cols: iter(files[name]))
    rows = download.sid_set("validation", tmp_path / "sid", n_per_class=2)
    assert len(rows) == 6 and len({r["path"] for r in rows}) == 6
    for r in rows:  # every stored image is the one its row says
        kind = {"": 0, "sid-full": 1, "sid-edit": 2}[r["generator"]]
        assert Image.open(r["path"]).convert("RGB").getpixel((5, 5)) == colors[kind]
        assert bool(r["mask"]) == (kind == 2)
    old = [dict(r, path=str(tmp_path / "sid" / "000004.png")) for r in rows]  # names of the old code
    save_rows(old, tmp_path / "sid.csv")
    assert download.sid_set("validation", tmp_path / "sid", n_per_class=2) == rows  # rebuilt, not reused


def test_scripts_end_to_end(tmp_path):
    """download_data.py, run_train.py, run_test.py and the predict and t-SNE notebooks on a tiny model and tiny data,
    as on the server."""
    import subprocess
    import sys

    import yaml

    repo = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    rng = np.random.default_rng(0)

    def table(name, spec):  # spec: (label, generator, source, masked) per image; folders and data/<name>.csv as downloaded
        rows = []
        for i, (label, generator, source, masked) in enumerate(spec):
            folder = data / name if source.startswith("sid") else data / ("fake" if label else "real") / (generator or source.split("/")[-1])
            folder.mkdir(parents=True, exist_ok=True)
            file = folder / f"{name}_{i}.png"
            w, h = 288 + 16 * (i % 3), 256 + 16 * (i % 2)
            cells = 6 if label else 24
            Image.fromarray(rng.integers(0, 255, (cells, cells, 3), dtype=np.uint8)).resize((w, h), Image.BICUBIC).save(file)
            mask = ""
            if masked:
                m = np.zeros((h, w), np.uint8)
                m[60:160, 80:200] = 255
                mask = str(folder / f"{name}_{i}_mask.png")
                Image.fromarray(m).save(mask)
            rows.append({"path": str(file), "label": label, "generator": generator, "mask": mask, "source": source})
        save_rows(rows, data / f"{name}.csv")

    sid = [(0, "", "sid/real", False)] * 3 + [(1, "sid-full", "sid/full", False)] * 3 + [(1, "sid-edit", "sid/edit", True)] * 3
    table("openfake_train", [(0, "", "openfake/laion", False)] * 24 + [(1, "gen-a", "openfake", False)] * 16 + [(1, "gen-b", "openfake", False)] * 16)
    table("openfake_test", [(0, "", "openfake/docci", False)] * 8 + [(1, "gen-a", "openfake", False)] * 4 + [(1, "gen-new", "openfake", False)] * 8)
    (tmp_path / "my_reals" / "logos").mkdir(parents=True)  # a folder of the user with real images
    for i in range(8):
        Image.fromarray(rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)).resize((300, 200), Image.BICUBIC).save(tmp_path / "my_reals" / "logos" / f"{i}.jpg")
    table("sid_validation", sid)

    cfg = {"data": {"dir": "data", "workers": 1, "train": {"n_real": None, "n_per_model": None},
                    "test": {"n_real": 10, "real_sources": ["docci"], "n_per_model": 10},
                    "real_dirs": [str(tmp_path / "my_reals")], "val_share": 0.2, "test_share": 0.2,
                    "sid": {"n_per_class": 3}, "n_render": 8, "render_train": 0.25},
           "model": {"backbone": "tiny", "shared": 4, "families": {"fam-a": ["gen-a"], "fam-b": ["gen-b"]}},
           "train": {"size": 128, "epochs": 2, "batch_size": 4, "val_size": 4, "lr": 1e-3, "head_lr": 1e-2,
                     "weight_decay": 0.0, "num_workers": 0, "device": "cpu", "out": "runs/experts"},
           "test": {"alpha": 0.1, "max_calib": 6}}
    (tmp_path / "tiny.yaml").write_text(yaml.safe_dump(cfg))

    def run(script, *args):  # offline: a call to the Hub fails at once instead of downloading
        env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        done = subprocess.run([sys.executable, str(repo / "scripts" / script), "--config", "tiny.yaml", *args],
                              cwd=tmp_path, capture_output=True, text=True, timeout=300, env=env)
        assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
        return done.stdout

    run("download_data.py")  # everything is cached: nothing is downloaded
    assert len(list((data / "fake" / "gen-a").iterdir())) == 20 and len(list((data / "real" / "docci").iterdir())) == 8
    train_log = run("run_train.py")
    sizes = {name: (int(n_real), int(n_fake)) for name, n_real, n_fake in
             re.findall(r"^(train|val|test): (\d+) reals from \d+ sources, (\d+) generated images", train_log, re.M)}
    assert sum(n for n, _ in sizes.values()) == 24 + 8 + 8 + 8 + round(0.25 * (sizes["train"][0] / 1.25))  # + renders
    assert sum(n for _, n in sizes.values()) == 44 and min(min(pair) for pair in sizes.values()) > 0
    assert "known to the general expert only: gen-new" in train_log
    assert re.search(r"expert all: \d+ generated images of 3 models", train_log)
    assert re.search(r"expert fam-b: \d+ generated images of 1 models", train_log) and "fam-b: best validation loss" in train_log
    run_dir = tmp_path / "runs" / "experts"
    assert sorted(p.name for p in (run_dir / "experts").iterdir()) == ["all.pt", "fam-a.pt", "fam-b.pt"]
    again = run("run_train.py", "--resume")  # nothing is left to train
    assert all(f"{name}: trained, skipped" in again for name in ("all", "fam-a", "fam-b")) and "epoch 1" not in again
    only = run("run_train.py", "--only", "fam-a")  # one expert again, on the same trunk
    assert "expert fam-a:" in only and "expert all:" not in only and "expert fam-b:" not in only

    run("run_test.py")
    report_md = (run_dir / "report.md").read_text()
    thr = json.loads((run_dir / "thresholds.json").read_text())
    assert list(thr) == ["all", "fam-a", "fam-b"]
    assert re.match(r"thresholds for 10% FPR on \d+ validation reals: all ", report_md)
    assert f"Test part: {sizes['test'][0]} reals and {sizes['test'][1]} generated images" in report_md
    assert "| only the general one |" in report_md and "| only the families |" in report_md
    assert "| laion |" in report_md and "| sid/real |" in report_md and "| sid-edit |" in report_md
    experts = [line.split("|")[4].strip() for line in report_md.splitlines() if line.startswith(("| gen-", "| sid-"))]
    assert len(experts) >= 4 and set(experts) <= set(thr)

    import matplotlib  # the predict notebook, cell by cell, with the trained tiny model
    matplotlib.use("Agg")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "train.yaml").write_text(yaml.safe_dump(cfg))
    (tmp_path / "my_images").mkdir()
    sample = sorted((data / "real" / "docci").iterdir())[0]
    Image.open(sample).save(tmp_path / "my_images" / "example.jpg")
    Image.open(sample).save(tmp_path / "my_images" / "example.png")
    def notebook(name):
        nb = json.loads((repo / "notebooks" / name).read_text(encoding="utf-8"))
        cwd = Path.cwd()
        os.chdir(tmp_path)
        try:
            ns = {}
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    exec("".join(cell["source"]), ns)
        finally:
            os.chdir(cwd)
        return ns

    ns = notebook("predict.ipynb")
    assert ns["thresholds"] == thr and set(ns["result"]) == set(thr)
    assert all(0 <= prob <= 1 and name in ("gen-a", "gen-b") for prob, name in ns["result"].values())
    for p in sorted((tmp_path / "my_images").iterdir()):  # the notebook prepares an image exactly as run_test.py
        assert torch.equal(ns["prepare"](p), ImageDataset([{"path": str(p), "label": 0}], cfg["train"]["size"])[0][0])

    ns = notebook("tsne.ipynb")  # every group of the test part and of SID_Set, all its images here (fewer than n)
    assert {"laion", "gen-a", "sid/real", "sid-full", "sid-edit"} <= set(ns["names"])
    assert ns["points"].shape == (len(ns["names"]), 2) and ns["vectors"].shape == (len(ns["names"]), 64)


def test_store_and_renders(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (2000, 1500), "gray").save(buf, format="PNG")
    path, size = _store(buf.getvalue(), tmp_path / "big", 512)
    assert path.suffix == ".png" and size == (2000, 1500) and path.read_bytes() == buf.getvalue()  # untouched
    assert _store(buf.getvalue(), tmp_path / "small", 1600) is None  # too small: skipped
    assert _store(b"not an image", tmp_path / "bad", 512) is None    # broken: skipped
    assert thresholds(np.linspace(0, 1, 101)[:, None], 0.05)[0] >= 0.94
    items = render_folder(tmp_path / "renders", 4, "test")
    assert len(items) == 4 and all(min(Image.open(it["path"]).size) >= 512 for it in items)


class FakeHub:
    """Stands in for the Hub: a split of 3 parquet files, file k holds the k-th third of the rows, grouped by
    kind as in OpenFake test files. `fail[name]` = how many more times that file breaks halfway."""

    def __init__(self):
        buf = io.BytesIO()
        Image.new("RGB", (600, 600), "white").save(buf, format="PNG")
        kinds = [("real", "docci"), ("real", "imagenet"), ("fake", "gen-a"), ("fake", "gen-b"), ("fake", "tiny-random-sana")]
        self.rows = [{"image": {"bytes": buf.getvalue()}, "label": label, "model": model, "prompt": f"prompt {i}",
                      "prompt_kind": "qwen_caption", "row_id": f"row-{i}"} for i, (label, model) in enumerate(k for k in kinds for _ in range(9))]
        self.fail, self.opened = {}, []

    def files(self, repo, prefix):
        return ["f0", "f1", "f2"]

    def part(self, name):
        k = int(name[1])
        return self.rows[k * len(self.rows) // 3:(k + 1) * len(self.rows) // 3]

    def read(self, repo, name, tmp_dir, columns):
        self.opened.append(name)
        part = self.part(name)
        if self.fail.get(name, 0) > 0:
            self.fail[name] -= 1
            return broken(part)
        return iter(part)


def broken(rows):
    yield from rows[: len(rows) // 2]
    raise ConnectionError("connection lost")


def fake_hub(monkeypatch):
    from src.data import download

    hub = FakeHub()
    monkeypatch.setattr(download, "_files", hub.files)
    monkeypatch.setattr(download, "_rows", hub.read)
    monkeypatch.setattr(download, "ATTEMPTS", 3)
    monkeypatch.setattr(download, "RETRY_WAIT", 0)
    return download, hub


def test_openfake_resume(tmp_path, monkeypatch):
    """A file that keeps breaking stops the run but the finished files stay; the rerun reads only that file,
    a single break is retried, and the quotas hold."""
    download, hub = fake_hub(monkeypatch)
    hub.fail = {"f2": 99}
    with pytest.raises(ConnectionError):
        download.openfake("test", tmp_path / "of", n_real=2, n_per_model=2, workers=3)
    assert sorted(p.name for p in (tmp_path / "of" / "openfake_test" / "done").glob("*.json")) == ["0000.json", "0001.json"]
    assert not (tmp_path / "of" / "openfake_test.csv").exists()

    hub.fail, hub.opened = {"f2": 1}, []
    rows = download.openfake("test", tmp_path / "of", n_real=2, n_per_model=2, workers=3)
    assert hub.opened == ["f2", "f2"]
    assert sum(r["label"] == 0 for r in rows) == 2
    assert sorted(r["generator"] for r in rows if r["label"]) == ["gen-a", "gen-a", "gen-b", "gen-b"]
    for r in rows:  # real/<source>/test_<file>_<row>.png or fake/<generator>/...; the prompt of that very row
        kind, folder, name = Path(r["path"]).parts[-3:]
        assert (kind, folder) == (("fake", r["generator"]) if r["label"] else ("real", r["source"].split("/")[1])) and Path(r["path"]).exists()
        split, file, row = Path(name).stem.split("_")
        assert split == "test" and r["prompt"] == f"prompt {int(row) + 15 * int(file)}"
    assert load_rows(tmp_path / "of" / "openfake_test.csv") == rows


def test_openfake_caps(tmp_path, monkeypatch):
    """Quotas shared by threads even when files are grouped by generator, real-source filter, skipped noise model,
    caching; no Hub."""
    download, hub = fake_hub(monkeypatch)

    for workers, max_files, out in ((1, None, "one"), (3, 3, "three")):
        rows = download.openfake("test", tmp_path / out, n_real=2, n_per_model=2, real_sources=["docci"], max_files=max_files, workers=workers)
        reals = [r for r in rows if r["label"] == 0]
        assert len(reals) == 2 and all(r["source"] == "openfake/docci" for r in reals)
        assert sorted(r["generator"] for r in rows if r["label"]) == ["gen-a", "gen-a", "gen-b", "gen-b"]
    assert download.openfake("test", tmp_path / "three", 2, 2) == load_rows(tmp_path / "three" / "openfake_test.csv")  # cached
    assert not [r for r in download.openfake("test", tmp_path / "first", 9, 9, max_files=1) if r["label"]]  # file 0: reals only
    assert len(download.openfake("test", tmp_path / "all", real_sources=["docci"])) == 27  # no limits: 9 docci, 9 gen-a, 9 gen-b

    hub.opened = []  # generator list: only gen-a is kept, and file 2 is never opened once every quota is full
    rows = download.openfake("test", tmp_path / "gen", n_real=2, n_per_model=2, real_sources=["docci"], generators=["gen-a"], workers=1)
    assert sorted(r["generator"] for r in rows if r["label"]) == ["gen-a", "gen-a"] and hub.opened == ["f0", "f1"]
