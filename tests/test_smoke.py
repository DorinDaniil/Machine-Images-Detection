"""CPU check of the whole chain on synthetic images, no downloads. python -m pytest tests -q"""

import io

import numpy as np
import pytest
from PIL import Image

from src.data import ImageDataset, TrainDataset, load_rows, render_folder, save_rows
from src.data.download import _store
from src.evaluate import report, score, threshold_at
from src.losses import patch_loss
from src.model import Detector
from src.train import balanced_weights, train


def make_rows(root, n=8, size=200):
    """Reals, fakes of two 'generators' (smoother images) and one locally edited image with a mask."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        label = i % 2
        noise = rng.integers(0, 255, (8 if label else 32, 8 if label else 32, 3), dtype=np.uint8)
        path = root / f"{i}.jpg"
        Image.fromarray(noise).resize((size, size), Image.BICUBIC).save(path, quality=95)
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

    model = Detector("tiny", layers=(0, 1), head_dim=16, topk=0.1)
    item = TrainDataset(rows, crop=160)[1]
    assert item["image"].shape == (3, 160, 160) and 0 < item["target"].mean() < 1  # masked image: partial target

    patch_logits, image_logit = model(item["image"][None])
    assert patch_logits.shape == (1, 1, 10, 10) and image_logit.shape == (1,)
    patch_loss(patch_logits, image_logit, item["target"][None]).backward()

    cfg = dict(device="cpu", out=str(tmp_path / "run"), batch_size=2, num_workers=0, lr=1e-3, weight_decay=0.0, epochs=1)
    best = train(model, TrainDataset(rows[2:], 160), TrainDataset(rows[:4], 160, seed=1), cfg)
    model = Detector.load(best)

    for protocol in ("native", "equalized"):
        reals = [r for r in rows if r["label"] == 0]
        real_scores, _ = score(model, ImageDataset(reals, model.patch_size, protocol), "cpu")
        fake_scores, f1 = score(model, ImageDataset(rows[1:2], model.patch_size, protocol), "cpu")
        assert np.isfinite(f1).all()
        thr = threshold_at(real_scores[:2], 0.5)
        assert "| edit |" in report(thr, {"reals": real_scores[2:]}, [("edit", real_scores, fake_scores, f1)])


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
        self.rows = [{"image": {"bytes": buf.getvalue()}, "label": label, "model": model} for label, model in kinds for _ in range(9)]
        self.fail, self.opened = {}, []

    def files(self, repo, prefix):
        return ["f0", "f1", "f2"]

    def read(self, repo, name, tmp_dir, columns):
        self.opened.append(name)
        k = int(name[1])
        part = self.rows[k * len(self.rows) // 3:(k + 1) * len(self.rows) // 3]
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
