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

from src.data import ImageDataset, distribution, load_pool, load_rows, render_folder, save_rows
from src.data.download import _store
from src.evaluate import calibrate, compare, predict, report, threshold
from src.model import Detector, score
from src.train import contrastive, stats, train

CFG = dict(device="cpu", size=96, batch_size=2, num_workers=0, lr=1e-3, head_lr=1e-2, weight_decay=0.0, epochs=2,
           save_every=3, balance=0.0, contrast=0.0)


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


def test_score_and_threshold():
    """One softmax: "generated" is everything but the class "real". The threshold comes from reals only."""
    logits = torch.tensor([[4.0, 0.0, 0.0], [0.0, 1.0, 3.0]])
    prob, index = score(logits.bfloat16())
    assert prob.dtype == torch.float32 and index.tolist() == [0, 1]
    assert prob[0].item() == pytest.approx(2 / (math.e ** 4 + 2), abs=1e-3)

    p = 1 - torch.softmax(logits, dim=1)[:, 0]  # the loss = "generated or real" + "which family" for the generated
    by_hand = (-torch.log(1 - p[0]) - torch.log(p[1]) + F.cross_entropy(logits[1:, 1:], torch.tensor([1]))) / 2
    assert F.cross_entropy(logits, torch.tensor([0, 2])).item() == pytest.approx(by_hand.item(), abs=1e-5)

    four = stats(torch.tensor([[0.0, 3.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 3.0], [3.0, 0.0, 0.0]]), torch.tensor([1, 0, 0, 2]))
    assert {k: four[k] for k in ("acc", "prec", "rec", "fpr", "family_acc")} == dict(acc=0.5, prec=0.5, rec=0.5, fpr=0.5, family_acc=0.5)

    real = np.random.default_rng(0).random(1000)
    assert 0.03 < (real > threshold(real, 0.05)).mean() <= 0.05  # at most alpha of the reals
    assert threshold(np.array([0.1, 0.5, 0.9, 0.3]), 0.5) == 0.5


def test_contrastive():
    """Supervised InfoNCE: lower when the classes are apart; one image counted by hand; vectors of earlier steps
    in the bank are positives and negatives too."""
    z = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.2], [0.0, 1.0], [0.2, 1.0]]), dim=1)
    y, mixed = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1])
    assert contrastive(z, y, z, y) < contrastive(z, mixed, z, mixed)
    sim = z[0] @ z[1:].T / 0.1  # image 0 against the others, itself left out; its only positive is image 1
    assert contrastive(z[:1], y[:1], z, y).item() == pytest.approx((sim.logsumexp(0) - sim[0]).item(), abs=1e-5)
    assert contrastive(z[:1], y[:1], z[[0, 2, 3]], y[[0, 2, 3]]).item() == 0  # no positives: zero, not NaN
    old = F.normalize(torch.tensor([[1.0, 0.1]]), dim=1)  # a vector of an earlier step, class 0, close to image 0
    loss = contrastive(z[[0, 2]], y[[0, 2]], torch.cat([z[[0, 2]], old]), torch.tensor([0, 1, 0]))
    assert 0 < loss.item() < 0.01
    z = z.clone().requires_grad_()
    contrastive(z, y, z.detach(), y).backward()
    assert z.grad.abs().sum() > 0


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))


def test_pipeline(tmp_path):
    """Train with both extras on, save, load, predict, calibrate, report, compare."""
    import copy

    rows = make_rows(tmp_path, n=16)
    save_rows(rows, tmp_path / "rows.csv")
    assert load_rows(tmp_path / "rows.csv")[3]["label"] == 2
    reals, fakes = [r for r in rows if r["label"] == 0], [r for r in rows if r["label"]]
    run = tmp_path / "runs" / "one"
    cfg = dict(CFG, out=str(run), balance=0.5, contrast=0.5)

    model = Detector("tiny", ["fam0", "fam1"])
    image, label = ImageDataset(rows, 96)[1]
    assert image.shape == (3, 96, 96) and label == 1
    assert model(image[None]).shape == (1, 3) and model.features(image[None]).shape == (1, 64)

    start = copy.deepcopy(model)
    train(model, reals[2:], fakes[2:], rows[:4], cfg)
    assert not same(start.net, model.net) and not same(start.head, model.head)
    assert sorted(p.name for p in run.iterdir()) == ["metrics.csv", "model.pt", "state.pt"]
    loaded = Detector.load(run / "model.pt")
    assert loaded.classes == ["fam0", "fam1"] and loaded.config == model.config
    log = (run / "metrics.csv").read_text().splitlines()
    val_loss = [float(line.split(",")[log[0].split(",").index("val_loss")]) for line in log[1:]]
    last = torch.load(run / "state.pt", weights_only=False)
    kept = all(torch.equal(v, last["model"][k]) for k, v in loaded.state_dict().items())
    assert kept == (val_loss[1] <= val_loss[0])  # saved = the epoch with the lowest validation loss

    data = ImageDataset(rows, 96)
    odds, index = predict(loaded, data, "cpu", batch_size=4, workers=0)
    assert odds.shape == index.shape == (16,) and set(index.tolist()) <= {0, 1}
    prob = score(loaded(data[0][0][None]))[0].item()  # log-odds of that very probability
    assert 1 / (1 + math.exp(-odds[0])) == pytest.approx(prob, abs=1e-4)

    real = np.array([r["label"] == 0 for r in rows])
    shift = calibrate(loaded, odds[real], 0.2)  # the bias of "real" moves: p = 0.5 is now the threshold
    again = predict(loaded, data, "cpu", batch_size=4, workers=0)[0]
    assert np.allclose(again, odds - shift, atol=1e-4) and (again[real] > 1e-4).mean() <= 0.2
    assert calibrate(loaded, again[real], 0.2) == pytest.approx(0, abs=1e-4)  # a second time changes nothing

    group = np.array([r["generator"] or "toy" for r in rows])
    named = np.array(loaded.classes)[index]
    truth = np.where(real, "", np.where(group == "gen0", "fam0", "?"))  # gen1 here stands for an unknown generator
    text, numbers = report(again > 1e-4, named, group, truth)
    cells = [[c.strip() for c in line.split("|")[1:-1]] for line in text.splitlines()]
    assert cells[2][:2] == ["toy", "8"] and float(cells[2][2]) <= 0.2 and numbers["FPR toy"] <= 0.2
    assert cells[6][:3] == ["gen0", "fam0", "4"] and cells[6][5] != ""
    assert cells[7][:3] == ["gen1", "?", "4"] and cells[7][5] == ""  # no true family to compare with
    assert cells[11][:2] == ["fam0", "4"] and len(cells) == 12 and list(numbers) == ["FPR toy", "TPR gen0", "TPR gen1"]

    (run / "report.json").write_text(json.dumps(numbers))  # every run with a report.json is a column
    (tmp_path / "runs" / "two").mkdir()
    (tmp_path / "runs" / "two" / "report.json").write_text(json.dumps({"FPR toy": 0.5, "new": 1.0}))
    table = compare(tmp_path / "runs").splitlines()
    assert table[0] == "| | one | two |" and table[2] == f"| FPR toy | {numbers['FPR toy']:.3f} | 0.500 |"
    assert table[-1] == "| new |  | 1.000 |" and len(table) == 2 + 4


def test_train_resume(tmp_path, monkeypatch):
    """A crash at the last step of epoch 2; resume goes on from the state saved after its step 3 (save_every = 3)."""
    import src.train

    class Crash(ImageDataset):  # fails on the `at`-th training image, like a power cut
        reads, at = 0, 16

        def __getitem__(self, i):
            Crash.reads += self.train
            if self.train and Crash.reads == Crash.at:
                raise RuntimeError("power cut")
            return super().__getitem__(i)

    monkeypatch.setattr(src.train, "ImageDataset", Crash)
    rows = make_rows(tmp_path)
    reals, fakes = [r for r in rows if r["label"] == 0], [r for r in rows if r["label"]]
    cfg, run = dict(CFG, out=str(tmp_path / "run")), tmp_path / "run"
    with pytest.raises(RuntimeError, match="power cut"):  # 4 steps of 2 images in an epoch: the 4th step of epoch 2
        train(Detector("tiny", ["fam0", "fam1"]), reals, fakes, rows[:4], cfg)
    state = torch.load(run / "state.pt", weights_only=False)
    assert (state["epoch"], state["step"]) == (2, 3)
    assert len(state["log"]) == 1 and state["scheduler"]["last_epoch"] == 7

    Crash.at = 0
    resumed = Detector("tiny", ["fam0", "fam1"])  # as a new process: weights come from the state
    train(resumed, reals, fakes, rows[:4], cfg, resume=True)
    state = torch.load(run / "state.pt", weights_only=False)
    assert (state["epoch"], state["step"]) == (3, 0)
    assert state["scheduler"]["last_epoch"] == 8 and Crash.reads == 16 + 2  # only the one step that was left
    epochs = [line.split(",")[0] for line in (run / "metrics.csv").read_text().splitlines()[1:]]
    assert epochs == ["1", "2"]
    assert all(torch.equal(v, state["model"][k]) for k, v in resumed.state_dict().items())
    train(Detector("tiny", ["fam0", "fam1"]), reals, fakes, rows[:4], cfg, resume=True)  # nothing is left to train
    assert Crash.reads == 16 + 2


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

    assert distribution(parts, {"fam": ["gen-a"]}).split() == [
        "40", "100.0%", "gen-a", "fam", "40", "generated", "images", "of", "1", "generators;", "65", "real", "images"]

    where = {r["path"]: name for name, part_rows in parts.items() for r in part_rows}
    for i in range(10, 30):
        blank.save(mine / "logos" / f"{i}.jpg")  # more images later: the old ones stay where they were
    again = {r["path"]: name for name, part_rows in load_pool(d).items() for r in part_rows}
    assert len(again) == 125 and all(again[p] == name for p, name in where.items())
    assert part("/disk1/data/real/laion/7.jpg", 0.2, 0.2) == part("/another/place/laion/7.jpg", 0.2, 0.2)  # data moved


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


def test_scripts_end_to_end(tmp_path):
    """download_data.py, run_train.py (the plain run and a variant with --set), run_test.py and the predict and
    t-SNE notebooks on a tiny model and tiny data, as on the server."""
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
           "model": {"backbone": "tiny", "families": {"fam-a": ["gen-a"], "fam-b": ["gen-b"]}},
           "train": {"size": 128, "epochs": 2, "batch_size": 4, "val_size": 4, "balance": 0.0, "contrast": 0.0,
                     "lr": 1e-3, "head_lr": 1e-2, "weight_decay": 0.0, "save_every": 2, "num_workers": 0,
                     "device": "cpu", "alpha": 0.1, "max_calib": 6, "out": "runs/base"}}
    (tmp_path / "tiny.yaml").write_text(yaml.safe_dump(cfg))

    def run(script, *args, config="tiny.yaml"):  # offline: a call to the Hub fails at once instead of downloading
        env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        done = subprocess.run([sys.executable, str(repo / "scripts" / script), "--config", config, *args],
                              cwd=tmp_path, capture_output=True, text=True, timeout=300, env=env)
        assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
        return done.stdout

    assert "44  generated images of 3 generators; " in run("download_data.py")  # cached: nothing is downloaded
    assert len(list((data / "fake" / "gen-a").iterdir())) == 20 and len(list((data / "real" / "docci").iterdir())) == 8
    train_log = run("run_train.py")
    sizes = {name: (int(n_real), int(n_fake)) for name, n_real, n_fake in
             re.findall(r"^(train|val|test): (\d+) reals from \d+ sources, (\d+) generated images", train_log, re.M)}
    assert sum(n for n, _ in sizes.values()) == 24 + 8 + 8 + 8 + round(0.25 * (sizes["train"][0] / 1.25))  # + renders
    assert sum(n for _, n in sizes.values()) == 44 and min(min(pair) for pair in sizes.values()) > 0
    assert "classes: real, fam-a, fam-b, gen-new" in train_log  # gen-new is in no family: a class of its own
    assert re.search(r"20 +45\.5% +gen-a +fam-a\n", train_log) and re.search(r"8 +18\.2% +gen-new +-\n", train_log)
    runs = tmp_path / "runs"
    assert sorted(p.name for p in (runs / "base").iterdir()) == ["config.yaml", "metrics.csv", "model.pt", "state.pt"]
    assert yaml.safe_load((runs / "base" / "config.yaml").read_text()) == cfg
    assert re.search(r"calibrated on \d+ validation reals", train_log)
    again = run("run_train.py", "--resume", config="runs/base/config.yaml")  # nothing is left to train
    assert "all epochs are done" in again and "epoch 1 step" not in again and re.search(r"moved by -?0\.000,", again)
    variant = run("run_train.py", "--set", "train.out=runs/both", "train.contrast=0.5", "train.balance=0.5")
    assert "balance 0.5, contrast 0.5" in variant and (runs / "both" / "model.pt").exists()
    assert yaml.safe_load((runs / "both" / "config.yaml").read_text())["train"]["contrast"] == 0.5

    run("run_test.py", config="runs/base/config.yaml")
    report_md = (runs / "base" / "report.md").read_text()
    numbers = json.loads((runs / "base" / "report.json").read_text())
    assert re.match(r"The model as it is: an image is called generated when p\(generated\) > 0\.5\. \S+% of \d+ validation reals", report_md)
    assert f"Test part: {sizes['test'][0]} reals and {sizes['test'][1]} generated images" in report_md
    assert "| p > 0.5, the model as it is |" in report_md and ", target FPR 0.1% |" in report_md
    assert "| laion |" in report_md and "| sid/real |" in report_md and "| sid-edit | ? |" in report_md
    table = [[c.strip() for c in line.split("|")[1:-1]] for line in report_md.splitlines() if line.startswith("|")]
    named = {row[4] for row in table if len(row) == 6 and row[0].startswith(("gen-", "sid-"))}
    assert named and named <= {"fam-a", "fam-b", "gen-new"}
    assert {row[0] for row in table if len(row) == 5} >= {"fam-a", "fam-b", "gen-new"}  # the table of families
    assert list(numbers)[:6] == ["FPR reals", "FPR SID reals", "TPR generators", "TPR sid-full", "precision", "family named right"]
    assert {"FPR laion", "FPR sid/real", "TPR gen-a", "TPR sid-edit"} <= set(numbers)
    assert (runs / "compare.md").read_text().splitlines()[0] == "| | base |"
    run("run_test.py", config="runs/both/config.yaml")  # the second tested run is a second column
    side = (runs / "compare.md").read_text().splitlines()
    assert side[0] == "| | base | both |" and side[2].startswith("| FPR reals | ") and len(side[2].split("|")) == 5

    import matplotlib  # the notebooks, cell by cell, with the trained tiny model of runs/base
    matplotlib.use("Agg")
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
    assert 0 <= ns["score"] <= 1 and list(ns["result"]) == ["fam-a", "fam-b", "gen-new"]
    assert sum(ns["result"].values()) == pytest.approx(1, abs=1e-4)
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
    assert threshold(np.linspace(0, 1, 101), 0.05) >= 0.94
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
    assert sorted(p.name for p in (tmp_path / "of").iterdir()) == ["fake", "openfake_test.csv", "real"]  # no work folder left

    old = tmp_path / "of" / "openfake_test"  # as left by a download with older code: records, and maybe images
    (old / "done").mkdir(parents=True)
    assert download.openfake("test", tmp_path / "of", n_real=2, n_per_model=2) == rows and not old.exists()
    (old / "tmp").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(old / "0000_00001.png")
    download.openfake("test", tmp_path / "of", n_real=2, n_per_model=2)
    assert [p.name for p in old.iterdir()] == ["0000_00001.png"]  # an image is never deleted, only the work files


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
