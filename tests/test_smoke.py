"""CPU check of the whole chain on synthetic images, no downloads. python -m pytest tests -q"""

import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import ImageDataset, TrainDataset, load_rows, render_folder, save_rows
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
    item = TrainDataset(rows, 160, train=False)[1]
    assert item["image"].shape == (3, 160, 160) and 0 < item["target"].mean() < 1  # masked image: partial target
    patch_logits, image_logit = model(item["image"][None])
    assert patch_logits.shape == (1, 1, 10, 10) and image_logit.shape == (1,)
    patch_loss(patch_logits, image_logit, item["target"][None]).backward()

    m = BinaryMetrics(0.5)
    m.update(torch.tensor([2.0, -2.0, 2.0, -2.0]), torch.tensor([1.0, 0.0, 0.0, 1.0]))
    assert m.summary() == {"acc": 0.5, "prec": 0.5, "rec": 0.5, "f1": 0.5, "fpr": 0.5}

    cfg = dict(device="cpu", out=str(tmp_path / "run"), batch_size=2, num_workers=0, lr=1e-3, weight_decay=0.0, epochs=2)
    best = train(model, TrainDataset(rows[2:], 160), TrainDataset(rows[:4], 160, seed=1, train=False), cfg)
    log = (tmp_path / "run" / "metrics.csv").read_text().splitlines()
    val_loss = [float(line.split(",")[log[0].split(",").index("val_loss")]) for line in log[1:]]
    assert len(val_loss) == 2 and best.exists()
    best_model, last_model = Detector.load(best), Detector.load(tmp_path / "run" / "last.pt")
    same = all(torch.equal(a, b) for a, b in zip(best_model.head.state_dict().values(), last_model.head.state_dict().values()))
    assert same == (val_loss[1] <= val_loss[0])  # best.pt = the epoch with the lowest validation loss

    reals = [r for r in rows if r["label"] == 0]
    real_scores, _ = score(best_model, ImageDataset(reals, best_model.patch_size), "cpu", tile=64, workers=0)
    fake_scores, f1 = score(best_model, ImageDataset(rows[1:2], best_model.patch_size), "cpu", tile=64, workers=0)
    assert np.isfinite(f1).all() and ((0 < real_scores) & (real_scores < 1)).all()
    thr = threshold_at(real_scores[:2], 0.5)
    assert "| edit |" in report(thr, {"reals": real_scores[2:]}, [("edit", real_scores, fake_scores, f1)])


class Crash(TrainDataset):
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
    val = TrainDataset(rows[:4], 160, seed=1, train=False)
    with pytest.raises(RuntimeError, match="power cut"):  # 8 images, then the 5th of epoch 2 (in its 3rd batch)
        train(Detector("tiny", layers=(0, 1), head_dim=16), Crash(rows[2:], at=13, size=160), val, cfg)
    state = torch.load(tmp_path / "run" / "state.pt", weights_only=False)
    assert state["epoch"] == 2 and state["progress"]["n"] == 4 and len(state["log"]) == 1
    assert state["scheduler"]["last_epoch"] == 6  # 4 steps of epoch 1, 2 of epoch 2

    resumed = Detector("tiny", layers=(0, 1), head_dim=16)  # as a new process: weights come from the state
    train(resumed, TrainDataset(rows[2:], 160), val, cfg, resume=True)
    state = torch.load(tmp_path / "run" / "state.pt", weights_only=False)
    assert state["epoch"] == 3 and state["progress"] is None and state["scheduler"]["last_epoch"] == 8
    epochs = [line.split(",")[0] for line in (tmp_path / "run" / "metrics.csv").read_text().splitlines()[1:]]
    assert epochs == ["1", "2"]
    assert all(torch.equal(a, b) for a, b in zip(resumed.head.state_dict().values(), state["head"].values()))


def test_train_views(tmp_path):
    """Every view is size x size at the image's own resolution: the pixels are not resized unless the downscale
    augmentation says so, and then the short side stays >= size; validation views are fixed; targets follow the label."""
    rows = make_rows(tmp_path)
    assert all(TrainDataset(rows, 160)[i]["image"].shape == (3, 160, 160) for i in range(8))
    assert TrainDataset(rows, 256)[0]["image"].shape == (3, 256, 256)  # smaller than the view: enlarged (a guard only)
    val = TrainDataset(rows, 160, train=False)
    assert torch.equal(val[2]["image"], val[2]["image"])  # validation views do not change
    real, fake = val[0], val[3]
    assert real["target"].max() == 0 and fake["target"].min() == 1 and real["label"] == 0 and fake["label"] == 1

    stripes = np.zeros((600, 900, 3), np.uint8)
    stripes[:, (np.arange(900) // 8) % 2 == 1] = 255  # vertical stripes 8 px wide
    Image.fromarray(stripes).save(tmp_path / "stripes.png")

    def widths(down_p):  # mean stripe width seen in the views: 8 px at the own resolution
        ds, out = TrainDataset([{"path": str(tmp_path / "stripes.png"), "label": 0}], 256, down_p=down_p), []
        for ds.epoch in range(20):
            row = ds[0]["image"][:, 128].mean(0) > 0.5  # one row, bright or dark
            out.append(256 / (1 + (row[1:] != row[:-1]).sum().item()))
        return np.array(out)

    native = widths(0.0)
    assert (abs(native - 8) < 0.6).all(), native  # never resized
    down = widths(1.0)
    assert (down < 7.7).mean() > 0.6 and (down > 3.5).all(), down  # downscaled by 0.5-1 (600 px short side: by >= 0.43)


def test_augment_alignment(tmp_path):
    """The target map follows the generated region through downscale, crop and flip."""
    image = np.full((500, 600, 3), 128, np.uint8)
    image[150:350, 200:450] = (255, 0, 0)
    mask = np.zeros((500, 600), np.uint8)
    mask[150:350, 200:450] = 255
    Image.fromarray(image).save(tmp_path / "edit.png")
    Image.fromarray(mask).save(tmp_path / "edit_mask.png")
    ds = TrainDataset([{"path": str(tmp_path / "edit.png"), "label": 1, "mask": str(tmp_path / "edit_mask.png")}],
                      256, down_p=0.5)
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


def test_tiles_and_pool():
    """An image no larger than a tile is the plain forward pass; larger ones are covered by tiles of exactly the
    tile size, stitched (and averaged where they overlap) into the whole patch map; top-k of all patches."""
    torch.manual_seed(0)
    model = Detector("tiny", layers=(0, 1), head_dim=16, topk=0.05).eval()
    x = torch.rand(1, 3, 160, 208)
    with torch.no_grad():
        plain = model(x)
    whole = model.full(x, tile=512)
    assert torch.allclose(plain[0], whole[0], atol=1e-5) and torch.allclose(plain[1], whole[1], atol=1e-5)

    seen = []
    patches = model.patches
    model.patches = lambda t: (seen.append(tuple(t.shape[2:])), patches(t))[1]  # record every tile
    side_by_side = model.full(x, tile=64)
    tiles_side = list(seen)
    seen.clear()
    overlapping = model.full(x, tile=64, step=32)
    model.patches = patches
    assert set(tiles_side) == set(seen) == {(64, 64)}  # every tile has the training view size
    assert len(tiles_side) == 3 * 4 and len(seen) == 4 * 6  # 10 x 13 patches: starts 0, 4, 6 and 0, 4, 8, 9 / step 2
    for patch_logits, image_logit in (side_by_side, overlapping):
        assert patch_logits.shape == (1, 1, 10, 13) and image_logit.shape == (1,) and torch.isfinite(patch_logits).all()
    with torch.no_grad():  # the top-left 4 x 4 patches lie in the first tile only, without overlap
        corner = model.patches(x[:, :, :64, :64])
    assert torch.allclose(side_by_side[0][:, :, :4, :4], corner, atol=1e-5)

    hot = torch.zeros(1, 1, 100, 100)
    hot.view(-1)[:50] = 1.0  # 10 000 patches, 50 of them generated
    assert model.pool(hot).item() == pytest.approx(50 / 500)  # 5 % of all patches, as in training
    assert model.pool(hot.bfloat16()).dtype == torch.float32  # no bf16 ties at the threshold


def test_test_views(tmp_path):
    """Test images are never cropped beyond the patch grid and never resized; the mask follows the image."""
    image = np.full((500, 700, 3), 128, np.uint8)
    image[:, 650:] = (255, 0, 0)  # a region at the far right edge must stay in view
    mask = np.zeros((500, 700), np.uint8)
    mask[:, 650:] = 255
    Image.fromarray(image).save(tmp_path / "wide.png")
    Image.fromarray(mask).save(tmp_path / "wide_mask.png")
    items = [{"path": str(tmp_path / "wide.png"), "label": 1, "mask": str(tmp_path / "wide_mask.png")}]
    item = ImageDataset(items, 16)[0]
    assert item["image"].shape == (3, 496, 688) and item["mask"].shape == (1, 496, 688)
    red, region = item["image"][0] - item["image"][1], item["mask"][0] > 0.5
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
    """download_data.py, run_train.py, run_test.py and the predict notebook on a tiny model and tiny data,
    as on the server: whole images, tiles at test, SID edits with masks in training."""
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
                    "sid_train": {"n_per_class": 3}, "sid": {"n_per_class": 3}, "n_render": 8},
           "model": {"backbone": "tiny", "layers": [0, 1], "head_dim": 16, "topk": 0.05},
           "train": {"size": 128, "down_p": 0.3, "samples_per_epoch": 16, "epochs": 2,
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
    calibration = json.loads((tmp_path / "runs" / "pilot" / "calibration.json").read_text())
    train_paths = {r["path"] for name in ("openfake_train", "sid_train") for r in load_rows(data / f"{name}.csv")}
    assert set(calibration["reals"]) <= train_paths and len(calibration["renders"]) == 4
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
    nb = json.loads((repo / "notebooks" / "predict.ipynb").read_text(encoding="utf-8"))
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        ns = {}
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                exec("".join(cell["source"]), ns)
    finally:
        os.chdir(cwd)
    assert report_md.startswith(f"threshold {ns['threshold']:.4f}") and 0 <= ns["prob"] <= 1


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

    def columns(self, repo, name, columns):
        return [{c: r[c] for c in columns} for r in self.part(name)]

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
    monkeypatch.setattr(download, "_columns", hub.columns)
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

    # files finished before prompts were kept: the next run fills them in from the text columns only
    for p in (tmp_path / "of" / "done").glob("*.json"):
        part = json.loads(p.read_text())
        for r in part["rows"]:
            for k in ("prompt", "prompt_kind", "row_id"):
                r.pop(k)
        p.write_text(json.dumps(part))
    (tmp_path / "of.csv").unlink()
    hub.opened = []
    again = download.openfake("test", tmp_path / "of", n_real=2, n_per_model=2, workers=3)
    assert hub.opened == [] and again == rows and all(r["row_id"].startswith("row-") for r in again)


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
