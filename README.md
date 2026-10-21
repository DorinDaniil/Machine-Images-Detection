# Machine-Images-Detection

Detection of generated and partly generated images. A frozen DINOv3 with a small head scores every
patch; the image score is the mean of the top 5 % patches, so a small edited region still counts.

Training data is simply real images and images of many generators (OpenFake: SD 1.x to FLUX.2,
GPT-Image-1, Imagen 4, Midjourney 6 and ~70 others). Batches are balanced: half reals, half fakes
spread evenly over generators. Every image is stored and augmented by the same rules whatever its
label, so file format and size cannot give the answer.

## Setup

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU
python -m pytest tests -q
```

Weights and data download from Hugging Face without access requests. DINOv3 comes from the timm copy
(Meta DINOv3 license: commercial use allowed, cite DINOv3 in papers). OpenFake is CC BY-NC 4.0: research use.
For long streams log in once (`hf auth login`, a read token): anonymous downloads are rate-limited.

## Pilot

```bash
python scripts/run_train.py --config configs/train.yaml
python scripts/run_test.py --config configs/train.yaml
```

- train: first 400k rows of OpenFake core/train, 60k reals (LAION, Pexels) and up to 5000 images
  per generator; `tiny-random-sana` (noise from a random-weight test model) is skipped;
  15 % of the reals are kept apart for calibration;
- threshold: 1 % false positives on the calibration reals and on rendered text, UI, charts, logos;
- test, against reals of the same source:
  - OpenFake core/test: newer generators (GPT-Image-1.5 and 2, Nano Banana Pro, Seedream 5,
    Midjourney 7, FLUX.2-klein-9B, Z-Image-Turbo, video frames of Veo 3 and Sora 2) vs DOCCI photos;
    generators that also appear in training are marked;
  - SID_Set: fully synthetic and locally edited images (pixel F1 from masks) vs SID reals;
  - rendered non-photo images: measured false positives;
- protocols: `equalized` (every image resized, cropped and JPEG-compressed the same way) and `native`;
- result: `runs/pilot/report.md`.

Data: 8 threads stream ~570 GB of train and 68 GB of test; only kept images are written (~80 GB).
Training: one GPU, a few hours. Generators are mixed in every train file and very unequal in size,
so `max_rows` decides how many of them reach 5000 (estimate from the first file):

| max_rows | streamed | generators with 5000 | with ≥ 1000 |
|---|---|---|---|
| 200k | 0.3 TB | 3 | 24 |
| 400k | 0.6 TB | 13 | 34 |
| 800k | 1.1 TB | 21 | 48 |
| 2.3M (all) | 3.3 TB | 35 | 64 |

## Layout

```
src/backbone.py   frozen DINOv3 (timm) -> patch features of chosen blocks
src/model.py      head, top-k pooling, save / load
src/losses.py     patch and image BCE
src/train.py      training loop, generator-balanced sampling
src/evaluate.py   scores, threshold from calibration reals, report table
src/data/         download.py (OpenFake, SID_Set), render.py (non-photo reals), dataset.py
scripts/          run_train.py, run_test.py
configs/          train.yaml
notebooks/        quickstart.ipynb
```
