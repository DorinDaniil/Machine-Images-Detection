"""CPU check of the whole chain on synthetic images, no downloads. python -m pytest tests -q"""

import numpy as np
from PIL import Image

from src.data import ImageDataset, PairDataset, blend, random_mask, render_folder
from src.data.decoders import find, save_like
from src.evaluate import report, score, threshold_at
from src.losses import trace_loss
from src.model import TraceDetector
from src.train import train


def make_pairs(root, n=6, size=200):
    """Real JPEGs and a stand-in "decoder" copy saved with the same JPEG history."""
    rng = np.random.default_rng(0)
    pairs = []
    for i in range(n):
        real = np.asarray(Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).resize((size, size), Image.BICUBIC))
        fake = np.clip(real.astype(float) + rng.normal(0, 4, real.shape), 0, 255).astype(np.uint8)
        pair = root / f"{i:03d}"
        pair.mkdir()
        Image.fromarray(real).save(pair / "real.jpg", quality=90)
        with Image.open(pair / "real.jpg") as src:
            save_like(Image.fromarray(fake), src, pair / "sd")
        pairs.append(pair)
    return pairs


def test_pipeline(tmp_path):
    pairs = make_pairs(tmp_path)
    assert find(pairs[0], "sd").suffix == ".jpg"
    model = TraceDetector("tiny", layers=(0, 1), head_dim=16, topk=0.1)
    data = PairDataset(pairs, ["sd"], crop=160)
    item = data[0]
    assert item["image"].shape == (3, 160, 160) and item["target"].shape == (1, 160, 160)

    patch_logits, image_logit = model(item["image"][None])
    assert patch_logits.shape == (1, 1, 10, 10) and image_logit.shape == (1,)
    trace_loss(patch_logits, image_logit, item["target"][None]).backward()

    cfg = dict(device="cpu", out=str(tmp_path / "run"), batch_size=2, num_workers=0, lr=1e-3, weight_decay=0.0, epochs=1)
    best = train(model, PairDataset(pairs[2:], ["sd"], 160), PairDataset(pairs[:2], ["sd"], 160, seed=1), cfg)
    model = TraceDetector.load(best)

    real = np.asarray(Image.open(find(pairs[0], "real")), dtype=np.float32)
    fake = np.asarray(Image.open(find(pairs[0], "sd")), dtype=np.float32)
    mask = random_mask(*real.shape[:2], np.random.default_rng(1))
    Image.fromarray(blend(real, fake, mask)).save(tmp_path / "local.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(tmp_path / "local_mask.png")

    for protocol in ("native", "equalized"):
        reals = ImageDataset([{"path": find(p, "real"), "label": 0} for p in pairs], model.patch_size, protocol, size=160)
        fakes = ImageDataset([{"path": tmp_path / "local.png", "label": 1, "mask": tmp_path / "local_mask.png"}],
                             model.patch_size, protocol, size=160)
        real_scores, _ = score(model, reals, "cpu")
        fake_scores, f1 = score(model, fakes, "cpu")
        assert np.isfinite(f1).all()
        assert "| local |" in report(real_scores[:3], real_scores[3:], {"local": (fake_scores, f1)})


def test_threshold_and_renders(tmp_path):
    scores = np.linspace(0, 1, 101)
    assert threshold_at(scores, 0.05) >= 0.94
    items = render_folder(tmp_path, 4, "test")
    assert len(items) == 4 and all(min(Image.open(it["path"]).size) >= 512 for it in items)
