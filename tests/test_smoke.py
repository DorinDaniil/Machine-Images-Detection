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
    assert _store(buf.getvalue(), tmp_path / "big.jpg", 1024, 512) == (1024, 768)  # shrunk and re-saved as JPEG
    assert _store(buf.getvalue(), tmp_path / "small.jpg", 1024, 1600) is None       # too small: skipped
    assert threshold_at(np.linspace(0, 1, 101), 0.05) >= 0.94
    items = render_folder(tmp_path / "renders", 4, "test")
    assert len(items) == 4 and all(min(Image.open(it["path"]).size) >= 512 for it in items)


class FakeStream(list):
    """Stands in for a streamed Hub split of 3 files: file k is the k-th contiguous third.
    File `fail` loses its connection halfway; `opened` lists the files that were read."""
    n_shards = 3

    def __init__(self, rows, fail=None):
        super().__init__(rows)
        self.fail, self.opened = fail, []

    def shard(self, n, k):
        self.opened.append(k)
        part = self[k * len(self) // n:(k + 1) * len(self) // n]
        return broken(part) if k == self.fail else part


def broken(rows):
    yield from rows[: len(rows) // 2]
    raise ConnectionError("connection lost")


def fake_openfake_rows():
    """9 rows of each kind, grouped by kind as in OpenFake test files."""
    buf = io.BytesIO()
    Image.new("RGB", (600, 600), "white").save(buf, format="PNG")
    kinds = [("real", "docci"), ("real", "imagenet"), ("fake", "gen-a"), ("fake", "gen-b"), ("fake", "tiny-random-sana")]
    return [{"image": {"bytes": buf.getvalue()}, "label": label, "model": model} for label, model in kinds for _ in range(9)]


def test_openfake_resume(tmp_path, monkeypatch):
    """A crash in one file keeps the finished files; the rerun reads only the broken file and respects the quotas."""
    from src.data import download

    stream = FakeStream(fake_openfake_rows(), fail=2)
    monkeypatch.setattr(download, "_stream", lambda *a, **k: stream)
    with pytest.raises(ConnectionError):
        download.openfake("test", tmp_path / "of", n_real=2, n_per_model=2, workers=3)
    assert sorted(p.name for p in (tmp_path / "of" / "done").glob("*.json")) == ["0000.json", "0001.json"]
    assert not (tmp_path / "of.csv").exists()

    stream.fail, stream.opened = None, []
    rows = download.openfake("test", tmp_path / "of", n_real=2, n_per_model=2, workers=3)
    assert stream.opened == [2]
    assert sum(r["label"] == 0 for r in rows) == 2
    assert sorted(r["generator"] for r in rows if r["label"]) == ["gen-a", "gen-a", "gen-b", "gen-b"]


def test_openfake_caps(tmp_path, monkeypatch):
    """Quotas shared by threads even when files are grouped by generator, real-source filter, skipped noise model,
    caching; no Hub."""
    from src.data import download

    stream = FakeStream(fake_openfake_rows())
    monkeypatch.setattr(download, "_stream", lambda *a, **k: stream)

    for workers, max_rows, out in ((1, None, "one"), (3, 100, "three")):
        rows = download.openfake("test", tmp_path / out, n_real=2, n_per_model=2, max_rows=max_rows, real_sources=["docci"], workers=workers)
        reals = [r for r in rows if r["label"] == 0]
        assert len(reals) == 2 and all(r["source"] == "openfake/docci" for r in reals)
        assert sorted(r["generator"] for r in rows if r["label"]) == ["gen-a", "gen-a", "gen-b", "gen-b"]
    assert download.openfake("test", tmp_path / "three", 2, 2, 100) == load_rows(tmp_path / "three.csv")  # cached
