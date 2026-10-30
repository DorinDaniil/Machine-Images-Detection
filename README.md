# Machine-Images-Detection

Detection of generated and partly generated images. A frozen DINOv3 with a small head scores every
patch; the image score is the mean of the top 5 % patches, so a small edited region still counts.
Training views are 512 x 512 crops at the images' own resolution; the test scores whole images in 512
tiles, so nothing is resized to fit the model and no part of a test image is left out.

Training data is simply real images and images of many generators (OpenFake: SD 1.x to FLUX.2,
GPT-Image-1, Imagen 4, Midjourney 6 and ~70 others); locally edited images with masks (SID_Set) can be added.
Batches are balanced: half reals, half fakes spread over generators (equally, small generators in
proportion to their size). Images are stored as untouched originals; every training image is compressed
at a random quality, so the file format cannot give the answer.

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
python scripts/run_train.py --config configs/train.yaml            # after a crash: add --resume
python scripts/run_test.py --config configs/train.yaml
```

- train: the whole of OpenFake core/train, 300k reals (LAION, Pexels) and up to 10k images of each
  of 36 generators: API models, open flagships, the most used community fine-tunes, frames of video
  models (list in the config); SID_Set train with locally edited images and their masks is off by
  default (`sid_train.n_per_class`); 15 % of the reals are kept apart for calibration; every row
  of the data CSV keeps the OpenFake prompt, prompt kind and row id;
- views: an image not stored as JPEG is JPEG-compressed once first, so both classes have a compression
  history; a random downscale (augmentation), then a random 512 x 512 crop at that resolution, then a
  flip, resampling, blur, noise, JPEG or WebP compression, on both classes alike (probabilities in the
  config); batches of 32;
- training log: accuracy, precision, recall, F1 and FPR of every epoch (train and validation) in
  `runs/pilot/metrics.csv`; `best.pt` is the epoch with the lowest validation loss;
- threshold: 1 % false positives on the calibration reals and on rendered text, UI, charts, logos;
- test, whole images, against reals of the same source:
  - OpenFake core/test: newer generators (GPT-Image-1.5 and 2, Nano Banana Pro, Seedream 5,
    Midjourney 7, FLUX.2-klein-9B, Z-Image-Turbo, video frames of Veo 3 and Sora 2) vs DOCCI photos;
    generators that also appear in training are marked;
  - SID_Set validation: fully synthetic and locally edited images (pixel F1 from masks) vs SID reals;
  - rendered non-photo images: measured false positives;
- test images: whole, at their own resolution, in 512 tiles, JPEG q90 like training;
- result: `runs/pilot/report.md`.

Downloads 3.3 TB (each file deleted once its images are taken) and keeps ~0.6 TB. Training: one GPU,
5 epochs over the whole train set. Generators, caps, files, augmentations: in the config.

## Layout

```
src/backbone.py   frozen DINOv3 (timm) -> patch features of chosen blocks
src/model.py      head, top-k pooling, tiled scoring of whole images, save / load
src/losses.py     patch and image BCE
src/train.py      training loop, generator-balanced sampling
src/evaluate.py   scores, threshold from calibration reals, report table
src/data/         download.py (OpenFake, SID_Set), render.py (non-photo reals), dataset.py
scripts/          download_data.py, run_train.py, run_test.py
configs/          train.yaml
notebooks/        quickstart.ipynb (model, data), predict.ipynb (verdict and heat map on your images)
```
