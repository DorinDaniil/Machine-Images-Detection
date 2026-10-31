"""CPU check of the whole chain on synthetic images, no downloads. python -m pytest tests -q"""

import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import ImageDataset, load_rows, render_folder, save_rows
from src.data.download import _store
from src.evaluate import report, score, threshold_at
from src.losses import patch_loss
from src.model import Detector
from src.train import BinaryMetrics, balanced_weights, train


def make_rows(root, n=8, size=200):
    """Reals, fakes of two 'generators' (smoother images) and one locally edited image with a mask."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        label = i % 2
        noise = rng.integers(0, 255, (8 if label else 32, 8 if label else 32, 3), dtype=np.uint8)
        path = root / f"{i}.jpg"
        w, h = (size, size) if i < 2 else (size + 24 * (i % 3), size - 16 * (i % 2))  # sizes differ, as in OpenFake
        Image.fromarray(noise).resize((w, h), Image.BICUBIC).save(path, quality=95)
        rows.append({"path": str(path), "label": label, "generator": f"gen{i % 4 // 2}" if label else "", "mask": "", "source": "toy"})
    mask = np.zeros((size, size), np.uint8)
    mask[50:120, 60:150] = 255
    Image.fromarray(mask).save(root / "edit_mask.png")
    rows[1]["mask"] = str(root / "edit_mask.png")
    return rows


def test_pipeline(tmp_path):
    rows = make_rows(tmp_path)
    save_rows(rows, tmp_path / "rows.csv")
    assert load_rows(tmp_path / "rows.csv")[1]["label"] == 1

    weights = balanced_weights(rows)
    assert abs(sum(weights) - 1.0) < 1e-6  # half reals, half fakes spread over generators
    small = [{"generator": g} for g in ["a"] * 4 + ["b"]]  # floor 2: b weighs like a 2-image generator
    w = balanced_weights(small, floor=2)
    assert abs(sum(w) - 0.5) < 1e-9 and abs(w[0] - 1 / 12) < 1e-9 and abs(w[4] - 1 / 6) < 1e-9

    model = Detector("tiny", layers=(0, 1), head_dim=16, topk=0.1)
    item = ImageDataset(rows, 160)[1]
    assert item["image"].shape == (3, 160, 160) and 0 < item["target"].mean() < 1  # masked image: partial target
    patch_logits, image_logit = model(item["image"][None])
    assert patch_logits.shape == (1, 1, 10, 10) and image_logit.shape == (1,)
    patch_loss(patch_logits, image_logit, item["target"][None]).backward()

    m = BinaryMetrics(0.5)
    m.update(torch.tensor([2.0, -2.0, 2.0, -2.0]), torch.tensor([1.0, 0.0, 0.0, 1.0]))
    assert m.summary() == {"acc": 0.5, "prec": 0.5, "rec": 0.5, "f1": 0.5, "fpr": 0.5}

    cfg = dict(device="cpu", out=str(tmp_path / "run"), batch_size=2, num_workers=0, lr=1e-3, weight_decay=0.0, epochs=2)
    best = train(model, ImageDataset(rows[2:], 160, train=True), ImageDataset(rows[:4], 160), cfg)
    log = (tmp_path / "run" / "metrics.csv").read_text().splitlines()
    val_loss = [float(line.split(",")[log[0].split(",").index("val_loss")]) for line in log[1:]]
    assert len(val_loss) == 2 and best.exists()
    best_model, last_model = Detector.load(best), Detector.load(tmp_path / "run" / "last.pt")
    same = all(torch.equal(a, b) for a, b in zip(best_model.head.state_dict().values(), last_model.head.state_dict().values()))
    assert same == (val_loss[1] <= val_loss[0])  # best.pt = the epoch with the lowest validation loss

    reals = [r for r in rows if r["label"] == 0]
    real_scores, _ = score(best_model, ImageDataset(reals, 160), "cpu", batch_size=2, workers=0)
    fake_scores, f1 = score(best_model, ImageDataset(rows[1:2], 160), "cpu", batch_size=2, workers=0)
    assert np.isfinite(f1).all() and ((0 < real_scores) & (real_scores < 1)).all()
    thr = threshold_at(real_scores[:2], 0.5)
    assert "| edit |" in report(thr, {"reals": real_scores[2:]}, [("edit", real_scores, fake_scores, f1)])


class Crash(ImageDataset):
    """Fails on its `at`-th image read, like a crash in the middle of an epoch."""

    def __init__(self, rows, at, **kwargs):
        super().__init__(rows, **kwargs)
        self.at, self.reads = at, 0

    def __getitem__(self, i):
        self.reads += 1
        if self.reads == self.at:
            raise RuntimeError("power cut")
        return super().__getitem__(i)


def test_train_resume(tmp_path):
    """A crash in the middle of epoch 2; resume goes on from the last saved state: no step lost or repeated."""
    rows = make_rows(tmp_path)
    cfg = dict(device="cpu", out=str(tmp_path / "run"), batch_size=2, num_workers=0, lr=1e-3, weight_decay=0.0,
               epochs=2, samples_per_epoch=8, save_every=1)
    val = ImageDataset(rows[:4], 160)
    with pytest.raises(RuntimeError, match="power cut"):  # 8 images, then the 5th of epoch 2 (in its 3rd batch)
        train(Detector("tiny", layers=(0, 1), head_dim=16), Crash(rows[2:], at=13, size=160, train=True), val, cfg)
    state = torch.load(tmp_path / "run" / "state.pt", weights_only=False)
    assert state["epoch"] == 2 and state["progress"]["n"] == 4 and len(state["log"]) == 1
    assert state["scheduler"]["last_epoch"] == 6  # 4 steps of epoch 1, 2 of epoch 2

    resumed = Detector("tiny", layers=(0, 1), head_dim=16)  # as a new process: weights come from the state
    train(resumed, ImageDataset(rows[2:], 160, train=True), val, cfg, resume=True)
    state = torch.load(tmp_path / "run" / "state.pt", weights_only=False)
    assert state["epoch"] == 3 and state["progress"] is None and state["scheduler"]["last_epoch"] == 8
    epochs = [line.split(",")[0] for line in (tmp_path / "run" / "metrics.csv").read_text().splitlines()[1:]]
    assert epochs == ["1", "2"]
    assert all(torch.equal(a, b) for a, b in zip(resumed.head.state_dict().values(), state["head"].values()))


def test_train_views(tmp_path):
    """Every view is the whole image resized to size x size, or (augmentation) a large part of it; validation
    views are the whole image and fixed; targets follow the label."""
    rows = make_rows(tmp_path)
    assert all(ImageDataset(rows, 160, train=True)[i]["image"].shape == (3, 160, 160) for i in range(8))
    val = ImageDataset(rows, 160)
    assert torch.equal(val[2]["image"], val[2]["image"])  # validation views do not change
    real, fake = val[0], val[3]
    assert real["target"].max() == 0 and fake["target"].min() == 1 and real["label"] == 0 and fake["label"] == 1

    halves = np.zeros((300, 900, 3), np.uint8)
    halves[:, 450:] = 255  # a wide image, dark left half, bright right half
    Image.fromarray(halves).save(tmp_path / "halves.png")
    item = {"path": str(tmp_path / "halves.png"), "label": 0}
    view = ImageDataset([item], 128)[0]["image"]
    row = view.mean(0)[64]
    assert view.shape == (3, 128, 128) and row[:60].max() < 0.1 and row[68:].min() > 0.9  # squeezed, not cut

    def bright(crop_p):  # bright share of the views: 1/2 for the whole image, flipped or not
        ds, out = ImageDataset([item], 128, train=True, crop_p=crop_p), []
        for ds.epoch in range(20):
            out.append((ds[0]["image"].mean(0) > 0.5).float().mean().item())
        return np.array(out)

    assert (abs(bright(0.0) - 0.5) < 0.03).all()
    assert (abs(bright(1.0) - 0.5) > 0.05).mean() > 0.5  # parts of the image


def test_augment_alignment(tmp_path, monkeypatch):
    """The target map follows the generated region through crop, shrink to MAX_SIDE, degradations, resize and flip."""
    from src.data import dataset

    monkeypatch.setattr(dataset, "MAX_SIDE", 400)  # this 600 px image counts as large
    image = np.full((500, 600, 3), 128, np.uint8)
    image[150:350, 200:450] = (255, 0, 0)
    mask = np.zeros((500, 600), np.uint8)
    mask[150:350, 200:450] = 255
    Image.fromarray(image).save(tmp_path / "edit.png")
    Image.fromarray(mask).save(tmp_path / "edit_mask.png")
    ds = ImageDataset([{"path": str(tmp_path / "edit.png"), "label": 1, "mask": str(tmp_path / "edit_mask.png")}],
                      256, train=True)
    checked = 0
    for ds.epoch in range(30):
        item = ds[0]
        target, red = item["target"][0], item["image"][0] - item["image"][1]  # red minus green: ~1 in the region
        assert target.shape == item["image"].shape[1:]
        inside, outside = target > 0.5, target < 0.5
        if inside.sum() > 100 and outside.sum() > 100:
            assert red[inside].mean() > 0.6 and red[outside].mean() < 0.25, ds.epoch
            checked += 1
    assert checked >= 10


def test_pool():
    """The image logit is the mean of the top 5 % patch logits, in float32."""
    model = Detector("tiny", layers=(0, 1), head_dim=16, topk=0.05)
    hot = torch.zeros(1, 1, 100, 100)
    hot.view(-1)[:50] = 1.0  # 10 000 patches, 50 of them generated
    assert model.pool(hot).item() == pytest.approx(50 / 500)  # 5 % of all patches, as in training
    assert model.pool(hot.bfloat16()).dtype == torch.float32  # no bf16 ties at the threshold


def test_test_views(tmp_path):
    """Test images are resized whole, nothing is cut off; the mask follows the image."""
    image = np.full((500, 700, 3), 128, np.uint8)
    image[:, 650:] = (255, 0, 0)  # a region at the far right edge must stay in view
    mask = np.zeros((500, 700), np.uint8)
    mask[:, 650:] = 255
    Image.fromarray(image).save(tmp_path / "wide.png")
    Image.fromarray(mask).save(tmp_path / "wide_mask.png")
    items = [{"path": str(tmp_path / "wide.png"), "label": 1, "mask": str(tmp_path / "wide_mask.png")}]
    item = ImageDataset(items, 256)[0]
    assert item["image"].shape == (3, 256, 256) and item["target"].shape == (1, 256, 256) and item["has_mask"]
    red, region = item["image"][0] - item["image"][1], item["target"][0] > 0.5
    assert region.any() and red[region].mean() > 0.6 and red[~region].mean() < 0.2


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
    as on the server: whole images resized, SID edits with masks in training."""
    import subprocess
    import sys

    import yaml

    repo = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    rng = np.random.default_rng(0)

    def table(name, spec):  # spec: (label, generator, source, masked) per image; data/<name>.csv as the loaders cache it
        folder = data / name
        folder.mkdir(parents=True)
        rows = []
        for i, (label, generator, source, masked) in enumerate(spec):
            w, h = 288 + 16 * (i % 3), 256 + 16 * (i % 2)
            cells = 6 if label else 24
            Image.fromarray(rng.integers(0, 255, (cells, cells, 3), dtype=np.uint8)).resize((w, h), Image.BICUBIC).save(folder / f"{i}.png")
            mask = ""
            if masked:
                m = np.zeros((h, w), np.uint8)
                m[60:160, 80:200] = 255
                mask = str(folder / f"{i}_mask.png")
                Image.fromarray(m).save(mask)
            rows.append({"path": str(folder / f"{i}.png"), "label": label, "generator": generator, "mask": mask, "source": source})
        save_rows(rows, data / f"{name}.csv")

    sid = [(0, "", "sid/real", False)] * 3 + [(1, "sid-full", "sid/full", False)] * 3 + [(1, "sid-edit", "sid/edit", True)] * 3
    table("openfake_train", [(0, "", "openfake/laion", False)] * 12 + [(1, "gen-a", "openfake", False)] * 6 + [(1, "gen-b", "openfake", False)] * 6)
    table("openfake_test", [(0, "", "openfake/docci", False)] * 6 + [(1, "gen-a", "openfake", False)] * 3 + [(1, "gen-new", "openfake", False)] * 3)
    table("sid_train", sid)
    table("sid_validation", sid)

    cfg = {"data": {"dir": "data", "workers": 1, "train": {"n_real": None, "n_per_model": None},
                    "test": {"n_real": 10, "real_sources": ["docci"], "n_per_model": 10},
                    "sid_train": {"n_per_class": 3}, "sid": {"n_per_class": 3}, "n_render": 8, "render_train": 0.25},
           "model": {"backbone": "tiny", "layers": [0, 1], "head_dim": 16, "topk": 0.05},
           "train": {"size": 128, "crop_p": 0.5, "samples_per_epoch": 16, "epochs": 2,
                     "balance_floor": 5000, "batch_size": 4, "val_size": 4, "lr": 1e-3, "weight_decay": 0.0,
                     "num_workers": 0, "device": "cpu", "out": "runs/pilot"},
           "test": {"alpha": 0.1, "max_calib": 6}}
    (tmp_path / "tiny.yaml").write_text(yaml.safe_dump(cfg))

    def run(script):  # offline: a call to the Hub fails at once instead of downloading
        env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        done = subprocess.run([sys.executable, str(repo / "scripts" / script), "--config", "tiny.yaml"],
                              cwd=tmp_path, capture_output=True, text=True, timeout=300, env=env)
        assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
        return done.stdout

    run("download_data.py")  # everything is cached: nothing is downloaded
    train_log = run("run_train.py")
    assert "with masks" in train_log and "best.pt: epoch" in train_log
    assert "incl. 3 renders" in train_log  # 15 reals, 2 for calibration, a quarter of the other 13 drawn
    assert len(list((data / "renders").glob("train_*"))) == 3  # training fonts, apart from the test renders
    calibration = json.loads((tmp_path / "runs" / "pilot" / "calibration.json").read_text())
    train_paths = {r["path"] for name in ("openfake_train", "sid_train") for r in load_rows(data / f"{name}.csv")}
    assert set(calibration["reals"]) <= train_paths and len(calibration["renders"]) == 4
    assert all(Path(p).name.startswith("test_") for p in calibration["renders"] + calibration["renders_test"])
    run("run_test.py")
    report_md = (tmp_path / "runs" / "pilot" / "report.md").read_text()
    assert report_md.startswith("threshold ") and "protocol" not in report_md
    assert "| gen-a (in train) |" in report_md and "| gen-new |" in report_md and "| sid-edit (in train) |" in report_md
    assert "calibration reals. unseen OpenFake generators: mean TPR" in report_md

    import matplotlib  # the predict notebook, cell by cell, with the trained tiny model
    matplotlib.use("Agg")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "train.yaml").write_text(yaml.safe_dump(cfg))
    (tmp_path / "my_images").mkdir()
    Image.open(data / "openfake_test" / "7.png").save(tmp_path / "my_images" / "example.jpg")
    Image.open(data / "openfake_test" / "7.png").save(tmp_path / "my_images" / "example.png")
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
    assert report_md.startswith(f"threshold {ns['threshold']:.4f}") and 0 <= ns["prob"] <= 1
    for p in sorted((tmp_path / "my_images").iterdir()):  # the notebook prepares an image exactly as run_test.py
        assert torch.equal(ns["prepare"](p), ImageDataset([{"path": str(p), "label": 0}], cfg["train"]["size"])[0]["image"])

    ns = notebook("tsne.ipynb")  # every test group, all its images here (fewer than n)
    assert set(ns["names"]) == {"openfake/docci", "gen-a", "gen-new", "sid/real", "sid-full", "sid-edit", "renders"}
    assert ns["points"].shape == (25, 2) and ns["vectors"].shape == (25, 16)


def test_store_and_renders(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (2000, 1500), "gray").save(buf, format="PNG")
    path, size = _store(buf.getvalue(), tmp_path / "big", 512)
    assert path.suffix == ".png" and size == (2000, 1500) and path.read_bytes() == buf.getvalue()  # untouched
    assert _store(buf.getvalue(), tmp_path / "small", 1600) is None  # too small: skipped
    assert _store(b"not an image", tmp_path / "bad", 512) is None    # broken: skipped
    assert threshold_at(np.linspace(0, 1, 101), 0.05) >= 0.94
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
    assert sorted(p.name for p in (tmp_path / "of" / "done").glob("*.json")) == ["0000.json", "0001.json"]
    assert not (tmp_path / "of.csv").exists()

    hub.fail, hub.opened = {"f2": 1}, []
    rows = download.openfake("test", tmp_path / "of", n_real=2, n_per_model=2, workers=3)
    assert hub.opened == ["f2", "f2"]
    assert sum(r["label"] == 0 for r in rows) == 2
    assert sorted(r["generator"] for r in rows if r["label"]) == ["gen-a", "gen-a", "gen-b", "gen-b"]
    assert all(r["prompt"] == f"prompt {int(Path(r['path']).stem.split('_')[1]) + 15 * int(Path(r['path']).stem[:4])}" for r in rows)


def test_openfake_caps(tmp_path, monkeypatch):
    """Quotas shared by threads even when files are grouped by generator, real-source filter, skipped noise model,
    caching; no Hub."""
    download, hub = fake_hub(monkeypatch)

    for workers, max_files, out in ((1, None, "one"), (3, 3, "three")):
        rows = download.openfake("test", tmp_path / out, n_real=2, n_per_model=2, real_sources=["docci"], max_files=max_files, workers=workers)
        reals = [r for r in rows if r["label"] == 0]
        assert len(reals) == 2 and all(r["source"] == "openfake/docci" for r in reals)
        assert sorted(r["generator"] for r in rows if r["label"]) == ["gen-a", "gen-a", "gen-b", "gen-b"]
    assert download.openfake("test", tmp_path / "three", 2, 2) == load_rows(tmp_path / "three.csv")  # cached
    assert not [r for r in download.openfake("test", tmp_path / "first", 9, 9, max_files=1) if r["label"]]  # file 0: reals only
    assert len(download.openfake("test", tmp_path / "all", real_sources=["docci"])) == 27  # no limits: 9 docci, 9 gen-a, 9 gen-b

    hub.opened = []  # generator list: only gen-a is kept, and file 2 is never opened once every quota is full
    rows = download.openfake("test", tmp_path / "gen", n_real=2, n_per_model=2, real_sources=["docci"], generators=["gen-a"], workers=1)
    assert sorted(r["generator"] for r in rows if r["label"]) == ["gen-a", "gen-a"] and hub.opened == ["f0", "f1"]
