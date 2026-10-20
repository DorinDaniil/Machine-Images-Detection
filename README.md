# Machine-Images-Detection

Detection of generated and partly generated images. A frozen DINO backbone with a small head
predicts, for every patch, whether it came out of a generator's decoder.

Training uses no generated images at all: fakes are real photos passed through the VAE of a
generator (`decode(encode(x))`), fully or inside a random region.

## Setup

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU
hf auth login
python -m pytest tests -q
```

On Hugging Face accept the licenses of `facebook/dinov3-vitb16-pretrain-lvd1689m` and
`black-forest-labs/FLUX.1-schnell`, or set `backbone: facebook/dinov2-with-registers-base`.

## Pilot

```bash
python scripts/run_train.py --config configs/train.yaml
python scripts/run_test.py --config configs/train.yaml
```

- train: 1600 COCO photos and their copies re-decoded by the Stable Diffusion 1.x VAE (`sd15`);
- test: 400 held-out COCO photos; the same photos re-decoded by the SDXL and FLUX VAEs, fully
  and inside a region with a mask; 200 images of each of nine generators from
  Qwen-Image-Bench (GPT-Image-2, Nano Banana Pro and 2, FLUX.2 Pro, Seedream 4.5, Imagen 4,
  Qwen-Image, Hunyuan Image 3, Kling);
- result: `runs/pilot/report.md`, per group AUROC, TPR at 1 % FPR and pixel F1.

Downloads about 4 GB; takes well under an hour on one GPU.

## Layout

```
src/backbone.py   frozen DINO -> patch features
src/model.py      head, top-k pooling, save / load
src/losses.py     patch and image BCE
src/train.py      training loop
src/evaluate.py   scores and the report table
src/data/         download.py (COCO, Qwen-Image-Bench), vae.py (re-decoding), dataset.py
scripts/          run_train.py, run_test.py
configs/          train.yaml
notebooks/        quickstart.ipynb
```
