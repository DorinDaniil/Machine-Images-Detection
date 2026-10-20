# Machine-Images-Detection

Detection of AI-generated and AI-edited images with patch-level localisation.
A frozen DINOv3 / DINOv2 backbone, a small head on top, trained on real images and
their copies re-decoded through generator VAEs (no generator outputs in training).

## Setup

```bash
pip install -r requirements.txt
# GPU build of torch for RTX 50xx:
# pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pytest tests -q
```

DINOv3 weights are gated on Hugging Face: accept the license and run `hf auth login`,
or set `model.backbone: facebook/dinov2-with-registers-base` in `configs/train.yaml`.

## Layout

```
src/            backbone.py, model.py, losses.py, train.py, evaluate.py, inference.py
src/data/       manifest.py, dataset.py, synth.py, augment.py, decoders.py,
                prepare_pairs.py, generate_partial.py, sources/
scripts/        run_train.py, run_test.py
configs/        train.yaml, test.yaml, smoke.yaml, data_sources_*.yaml
notebooks/      quickstart.ipynb
tests/          CPU smoke test
docs/           research notes
```

## Run

```bash
python -m src.data.prepare_pairs --manifest data/manifests/train_real.csv --out data/aligned
python scripts/run_train.py --config configs/train.yaml
python scripts/run_test.py --config configs/test.yaml
```
