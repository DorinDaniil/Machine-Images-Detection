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

All weights and data are on Hugging Face without access requests (DINOv3 through timm).
OpenFake is CC BY-NC 4.0: research use.

## Pilot

```bash
python scripts/run_train.py --config configs/train.yaml
python scripts/run_test.py --config configs/train.yaml
```

- train: OpenFake core/train, 5000 reals (LAION, Pexels) and up to 150 images per generator;
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

Streams about 30 shards of 5 GB (only kept images are written, ~5 GB on disk); one GPU, a few hours.
Fewer rows: lower `max_rows` in the config.

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
