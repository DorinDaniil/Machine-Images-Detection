"""CPU check of the whole chain on synthetic images, no downloads. python -m pytest tests -q"""

import numpy as np
from PIL import Image

from src.data import ImageDataset, PairDataset, blend, random_mask
from src.evaluate import report, score
from src.losses import trace_loss
from src.model import TraceDetector
from src.train import train


def make_pairs(root, n=6, size=200):
    rng = np.random.default_rng(0)
    pairs = []
    for i in range(n):
        real = np.asarray(Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).resize((size, size), Image.BICUBIC))
        fake = np.clip(real.astype(float) + rng.normal(0, 4, real.shape), 0, 255).astype(np.uint8)  # stand-in for a VAE
        pair = root / f"{i:03d}"
        pair.mkdir()
        Image.fromarray(real).save(pair / "real.png")
        Image.fromarray(fake).save(pair / "sd15.png")
        pairs.append(pair)
    return pairs


def test_pipeline(tmp_path):
    pairs = make_pairs(tmp_path)
    model = TraceDetector("tiny", head_dim=16, topk=0.1)
    data = PairDataset(pairs, ["sd15"], crop=168)
    item = data[0]
    assert item["image"].shape == (3, 168, 168) and item["target"].shape == (1, 168, 168)

    patch_logits, image_logit = model(item["image"][None])
    assert patch_logits.shape == (1, 1, 12, 12) and image_logit.shape == (1,)
    trace_loss(patch_logits, image_logit, item["target"][None]).backward()

    cfg = dict(device="cpu", out=str(tmp_path / "run"), batch_size=2, num_workers=0, lr=1e-3, weight_decay=0.0, epochs=1)
    best = train(model, PairDataset(pairs[2:], ["sd15"], 168), PairDataset(pairs[:2], ["sd15"], 168, seed=1), cfg)
    model = TraceDetector.load(best)

    real = np.asarray(Image.open(pairs[0] / "real.png"), dtype=np.float32)
    fake = np.asarray(Image.open(pairs[0] / "sd15.png"), dtype=np.float32)
    mask = random_mask(*real.shape[:2], np.random.default_rng(1))
    Image.fromarray(blend(real, fake, mask)).save(tmp_path / "local.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(tmp_path / "local_mask.png")

    reals = ImageDataset([{"path": p / "real.png", "label": 0} for p in pairs], model.patch_size)
    fakes = ImageDataset([{"path": tmp_path / "local.png", "label": 1, "mask": tmp_path / "local_mask.png"}], model.patch_size)
    real_scores, _ = score(model, reals, "cpu")
    fake_scores, f1 = score(model, fakes, "cpu")
    assert np.isfinite(f1).all()
    assert "| local |" in report(real_scores, {"local": (fake_scores, f1)})
