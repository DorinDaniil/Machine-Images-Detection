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
from src.evaluate import predict, report, threshold_at
from src.model import Detector, family_loss, score
from src.train import balanced_weights, stats, train


def make_rows(root, n=8, size=200):
    """Reals (label 0) and fakes of two 'generators' (smoother images), each generator its own family (1, 2)."""
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


def test_loss_and_score():
    """A head learns only from reals and from the fakes of its own family; the score is the most confident head."""
    sure = torch.tensor([[9.0, -9.0], [9.0, 9.0]])  # family 1, whatever head 2 says about it
    assert family_loss(sure, torch.tensor([1, 1])).item() < 1e-3
    assert family_loss(sure[1:], torch.tensor([0])).item() > 8  # a real image: both heads are wrong
    logits, label = torch.tensor([[0.3, -1.2], [2.0, 0.5], [-0.7, 1.1]]), torch.tensor([0, 2, 0])
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    by_hand = bce(logits[[0, 2]], torch.zeros(2, 2)) + bce(logits[1, 1], torch.tensor(1.0))  # reals + fakes
    assert family_loss(logits, label).item() == pytest.approx(by_hand.item())

    s, head = score(torch.tensor([[-2.0, 3.0], [1.0, -4.0]]).bfloat16())
    assert s.dtype == torch.float32 and head.tolist() == [1, 0] and s[0].item() == pytest.approx(torch.sigmoid(torch.tensor(3.0)).item(), abs=1e-3)
    four = stats(torch.tensor([[2.0, -2.0], [-2.0, -2.0], [2.0, -2.0], [-2.0, -2.0]]), torch.tensor([1, 0, 0, 2]))
    assert {k: four[k] for k in ("acc", "prec", "rec", "f1", "fpr")} == {"acc": 0.5, "prec": 0.5, "rec": 0.5, "f1": 0.5, "fpr": 0.5}


def test_pipeline(tmp_path):
    rows = make_rows(tmp_path)
    save_rows(rows, tmp_path / "rows.csv")
    assert load_rows(tmp_path / "rows.csv")[3]["label"] == 2

    weights = balanced_weights(rows)
    assert abs(sum(weights) - 1.0) < 1e-6  # half reals, half fakes spread over the families
    small = [{"label": k} for k in [1] * 4 + [2]]  # floor 2: family 2 weighs like a family of 1 image, half of family 1
    w = balanced_weights(small, floor=2)
    assert abs(sum(w) - 0.5) < 1e-9 and abs(w[0] - 1 / 12) < 1e-9 and abs(w[4] - 1 / 6) < 1e-9

    model = Detector("tiny", ["a", "b"], train_blocks=1)
    trained = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all(n.startswith(("vit.blocks.1.", "vit.norm.", "head.")) for n in trained) and "head.weight" in trained
    assert not any(p.requires_grad for p in Detector("tiny", ["a"], train_blocks=0).vit.parameters())  # frozen backbone
    image, label = ImageDataset(rows, 160)[1]
    assert image.shape == (3, 160, 160) and label == 1
    assert model(image[None]).shape == (1, 2) and model.features(image[None]).shape == (1, 64)

    cfg = dict(device="cpu", out=str(tmp_path / "run"), batch_size=2, num_workers=0, lr=1e-3, head_lr=1e-2,
               weight_decay=0.0, epochs=2, samples_per_epoch=4)
    before = [p.clone() for p in model.vit.blocks[0].parameters()]
    best = train(model, ImageDataset(rows[2:], 160, train=True), ImageDataset(rows[:4], 160), cfg)
    assert all(torch.equal(a, b) for a, b in zip(before, model.vit.blocks[0].parameters()))  # frozen block untouched
    log = (tmp_path / "run" / "metrics.csv").read_text().splitlines()
    val_loss = [float(line.split(",")[log[0].split(",").index("val_loss")]) for line in log[1:]]
    assert len(val_loss) == 2 and best.exists()
    best_model, last = Detector.load(best), torch.load(tmp_path / "run" / "state.pt", weights_only=False)["model"]
    same = all(torch.equal(v, last[k]) for k, v in best_model.state_dict().items())
    assert same == (val_loss[1] <= val_loss[0])  # best.pt = the epoch with the lowest validation loss
    assert best_model.config == model.config

    reals = [r for r in rows if r["label"] == 0]
    real_scores, _ = predict(best_model, ImageDataset(reals, 160), "cpu", batch_size=2, workers=0)
    fake_scores, heads = predict(best_model, ImageDataset(rows[1:2], 160), "cpu", batch_size=2, workers=0)
    assert ((0 < real_scores) & (real_scores < 1)).all() and heads.shape == (1,)
    thr = threshold_at(real_scores[:2], 0.5)
    table = report(thr, {"reals": real_scores[2:]}, [("gen0", real_scores, fake_scores, heads)], ["a", "b"])
    assert "| gen0 | 1 |" in table and table.rstrip().endswith(f"| {'ab'[heads[0]]} |")


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
    """A crash in the middle of epoch 2; resume goes on from the state saved after epoch 1."""
    rows = make_rows(tmp_path)
    cfg = dict(device="cpu", out=str(tmp_path / "run"), batch_size=2, num_workers=0, lr=1e-3, head_lr=1e-2,
               weight_decay=0.0, epochs=2, samples_per_epoch=4)
    val = ImageDataset(rows[:4], 160)
    with pytest.raises(RuntimeError, match="power cut"):  # 4 images of epoch 1, then the 2nd of epoch 2
        train(Detector("tiny", ["a", "b"], train_blocks=1), Crash(rows[2:], at=6, size=160, train=True), val, cfg)
    state = torch.load(tmp_path / "run" / "state.pt", weights_only=False)
    assert state["epoch"] == 2 and len(state["log"]) == 1 and state["scheduler"]["last_epoch"] == 2

    resumed = Detector("tiny", ["a", "b"], train_blocks=1)  # as a new process: weights come from the state
    train(resumed, ImageDataset(rows[2:], 160, train=True), val, cfg, resume=True)
    state = torch.load(tmp_path / "run" / "state.pt", weights_only=False)
    assert state["epoch"] == 3 and state["scheduler"]["last_epoch"] == 4
    epochs = [line.split(",")[0] for line in (tmp_path / "run" / "metrics.csv").read_text().splitlines()[1:]]
    assert epochs == ["1", "2"]
    assert all(torch.equal(v, state["model"][k]) for k, v in resumed.state_dict().items())


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
    table("sid_validation", sid)

    cfg = {"data": {"dir": "data", "workers": 1, "train": {"n_real": None, "n_per_model": None},
                    "test": {"n_real": 10, "real_sources": ["docci"], "n_per_model": 10},
                    "sid": {"n_per_class": 3}, "n_render": 8, "render_train": 0.25},
           "model": {"backbone": "tiny", "train_blocks": 1, "families": {"fam-a": ["gen-a"], "fam-b": ["gen-b"]}},
           "train": {"size": 128, "samples_per_epoch": 16, "epochs": 2, "balance_floor": 10000, "batch_size": 4,
                     "val_size": 4, "lr": 1e-3, "head_lr": 1e-2, "weight_decay": 0.0,
                     "num_workers": 0, "device": "cpu", "out": "runs/families"},
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
    assert "images per family: fam-a 6, fam-b 6" in train_log and "best.pt: validation loss" in train_log
    assert "incl. 3 renders" in train_log  # 12 reals, 1 for calibration, a quarter of the other 11 drawn
    assert len(list((data / "renders").glob("train_*"))) == 3  # training fonts, apart from the test renders
    calibration = json.loads((tmp_path / "runs" / "families" / "calibration.json").read_text())
    train_paths = {r["path"] for r in load_rows(data / "openfake_train.csv")}
    assert set(calibration["reals"]) <= train_paths and len(calibration["renders"]) == 4
    assert all(Path(p).name.startswith("test_") for p in calibration["renders"] + calibration["renders_test"])
    run("run_test.py")
    report_md = (tmp_path / "runs" / "families" / "report.md").read_text()
    assert report_md.startswith("threshold ")
    assert "| gen-a (in train) |" in report_md and "| gen-new |" in report_md and "| sid-edit |" in report_md
    assert all(line.rstrip(" |").endswith(("fam-a", "fam-b")) for line in report_md.splitlines() if line.startswith(("| gen-", "| sid-")))
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
        assert torch.equal(ns["prepare"](p), ImageDataset([{"path": str(p), "label": 0}], cfg["train"]["size"])[0][0])

    ns = notebook("tsne.ipynb")  # every test group, all its images here (fewer than n)
    assert set(ns["names"]) == {"openfake/docci", "gen-a", "gen-new", "sid/real", "sid-full", "sid-edit", "renders"}
    assert ns["points"].shape == (25, 2) and ns["vectors"].shape == (25, 64)


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
