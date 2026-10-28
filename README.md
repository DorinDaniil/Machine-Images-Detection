# Machine-Images-Detection

Detection of generated and partly generated images. A frozen DINOv3 with a small head scores every
patch; the image score is the mean of the top 5 % patches, so a small edited region still counts.

Training data is simply real images and images of many generators (OpenFake: SD 1.x to FLUX.2,
GPT-Image-1, Imagen 4, Midjourney 6 and ~70 others). Batches are balanced: half reals, half fakes
spread over generators (equally, small generators in proportion to their size). Images are stored as
untouched originals; every training image is JPEG-compressed at a random quality, so the file format
cannot give the answer.

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
python scripts/download_data.py --config configs/train.yaml   # once; stop and rerun any time
python scripts/run_train.py --config configs/train.yaml
python scripts/run_test.py --config configs/train.yaml
```

- train: the whole of OpenFake core/train, 300k reals (LAION, Pexels) and up to 10k images of each
  of 36 generators: API models, open flagships, the most used community fine-tunes, frames of video
  models (list in the config); images are used at their own resolution, never resized (sides above
  1024 are randomly cropped); 15 % of the reals are kept apart for calibration; every row of the
  data CSV keeps the OpenFake prompt, prompt kind and row id, so real / fake pairs with the same
  caption can be formed;
- training log: accuracy, precision, recall, F1 and FPR of every epoch (train and validation) in
  `runs/pilot/metrics.csv`; `best.pt` is the epoch with the best validation F1;
- threshold: 1 % false positives on the calibration reals and on rendered text, UI, charts, logos;
- test, against reals of the same source:
  - OpenFake core/test: newer generators (GPT-Image-1.5 and 2, Nano Banana Pro, Seedream 5,
    Midjourney 7, FLUX.2-klein-9B, Z-Image-Turbo, video frames of Veo 3 and Sora 2) vs DOCCI photos;
    generators that also appear in training are marked;
  - SID_Set: fully synthetic and locally edited images (pixel F1 from masks) vs SID reals;
  - rendered non-photo images: measured false positives;
- protocols: `native` (own resolution, as in training) and `equalized` (every image resized and cropped
  to 512, a check for size shortcuts); both JPEG q90, like training;
- result: `runs/pilot/report.md`.

Downloads 3.3 TB (608 train files, each deleted once its images are taken) and keeps ~0.6 TB. Training:
one GPU, epochs of 100k images, about an hour each. Generators, caps, files: in the config.

## Layout

```
src/backbone.py   frozen DINOv3 (timm) -> patch features of chosen blocks
src/model.py      head, top-k pooling, save / load
src/losses.py     patch and image BCE
src/train.py      training loop, generator-balanced sampling
src/evaluate.py   scores, threshold from calibration reals, report table
src/data/         download.py (OpenFake, SID_Set), render.py (non-photo reals), dataset.py
scripts/          download_data.py, run_train.py, run_test.py
configs/          train.yaml
notebooks/        quickstart.ipynb (model, data), predict.ipynb (verdict and heat map on your images)
```
