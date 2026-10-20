# Machine-Images-Detection

Detection of AI-generated and AI-edited images with patch-level localisation.
A frozen DINOv3 / DINOv2 backbone with a small head, trained on real photos and their
copies re-decoded through generator VAEs. No generator outputs are used in training.

## Setup

```bash
pip install -r requirements.txt
# GPU build of torch for RTX 50xx:
# pip install torch --index-url https://download.pytorch.org/whl/cu128
hf auth login          # DINOv3 and the FLUX VAE are gated: accept their licenses on Hugging Face
python -m pytest tests -q
```

Without DINOv3 access set `model.backbone: facebook/dinov2-with-registers-base` in `configs/train.yaml`.

## Pilot experiment

One command. Question: does the decoder trace learned on one VAE transfer to unseen decoder
families and to real generator outputs?

```bash
python scripts/run_pilot.py --config configs/train.yaml
```

What it does: 2000 COCO photos, 80/20 split; training pairs re-decoded with `sd15`; held-out
pairs re-decoded with `sdxl` and `flux` (never seen in training), turned into full re-decodes
and local pastes with masks; training; a threshold with FPR <= 1 % fitted on the held-out
reals; metrics on the held-out fakes and on DailyBench (FLUX.2, Qwen-Image, Nano Banana 2,
GPT-Image 2 and edits by the same models). Results land in `runs/pilot/summary.md`.

CPU dry run with a tiny random backbone, about ten minutes:

```bash
python scripts/run_pilot.py --config configs/smoke.yaml --n-real 30 --test-decoders sdxl --bench-items 40 --max-side 512 --epochs 1 --out runs/pilot_smoke
```

## Full run

```bash
python -m src.data.build_manifest --role train_real --out data/manifests/train_real.csv
python -m src.data.prepare_pairs --manifest data/manifests/train_real.csv --out data/aligned --decoders sd15,sdxl,flux
python scripts/run_train.py --config configs/train.yaml
python scripts/run_test.py --config configs/test.yaml
```

Datasets are listed in `configs/datasets.yaml`; a Hugging Face dataset is added with a few
YAML lines (`kind: hf_parquet` or `hf_files`). The full registry with about 75 sources
(photos, art, graphics, documents, biomedical, 30 benchmarks) is kept in branch `data-registry`.

## Layout

```
src/            backbone.py, model.py, losses.py, train.py, evaluate.py, inference.py
src/data/       manifest.py, dataset.py, synth.py, augment.py, decoders.py, prepare_pairs.py,
                registry.py, build_manifest.py, sources/ (generic loaders, DailyBench, local folder)
scripts/        run_pilot.py, run_train.py, run_test.py
configs/        train.yaml, test.yaml, smoke.yaml, datasets.yaml
notebooks/      quickstart.ipynb
tests/          CPU smoke test
docs/           research notes
```
