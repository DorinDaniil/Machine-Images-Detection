# Machine-Images-Detection

Detection of generated and partly generated images. A frozen DINOv3 with a small head scores every
patch; the image score is the mean of the top 5 % patches, so a small edited region still counts.

Training data is simply real images and images of many generators (OpenFake: SD 1.x to FLUX.2,
GPT-Image-1, Imagen 4, Midjourney 6 and ~70 others). Batches are balanced: half reals, half fakes
spread evenly over generators. Images are stored as untouched originals; every training image is
JPEG-compressed at a random quality, so the file format cannot give the answer.

## Setup

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU
python -m pytest tests -q
```

Weights and data download from Hugging Face without access requests. DINOv3 comes from the timm copy
(Meta DINOv3 license: commercial use allowed, cite DINOv3 in papers). OpenFake is CC BY-NC 4.0: research use.
Log in once on the download machine (`hf auth login`, a read token): anonymous downloads are rate-limited.

## Pilot

```bash
python scripts/download_data.py --config configs/train.yaml   # once; stop and rerun any time
python scripts/run_train.py --config configs/train.yaml
python scripts/run_test.py --config configs/train.yaml
```

- train: 100k reals (LAION, Pexels) and up to 5000 images of each of 21 popular and recent
  generators (GPT-Image-1, DALL·E 3, Imagen 3/4, Midjourney 6, Ideogram 3, FLUX 1 and 2, SD 1.5 to
  3.5, HiDream, Qwen-Image, ...) from the first 200 of 608 files; the generator list, the caps and
  the number of files are in the config (`null` = everything); 15 % of the reals are kept apart for
  calibration;
- threshold: 1 % false positives on the calibration reals and on rendered text, UI, charts, logos;
- test, against reals of the same source:
  - OpenFake core/test: newer generators (GPT-Image-1.5 and 2, Nano Banana Pro, Seedream 5,
    Midjourney 7, FLUX.2-klein-9B, Z-Image-Turbo, video frames of Veo 3 and Sora 2) vs DOCCI photos;
    generators that also appear in training are marked;
  - SID_Set: fully synthetic and locally edited images (pixel F1 from masks) vs SID reals;
  - rendered non-photo images: measured false positives;
- protocols: `equalized` (every image resized, cropped and JPEG-compressed the same way) and `native`;
- result: `runs/pilot/report.md`.

Data: `download_data.py` fetches train files (5.4 GB each) and the 13 test files whole, 8 at a time,
keeps the original bytes of the images the limits allow and deletes each file (~45 GB of temporary
disk). The pilot downloads ~1.1 TB and keeps ~0.25 TB; the whole set is 3.3 TB and ~3 TB on disk.
A broken file is retried; a stopped run resumes from the finished files. Training: one GPU, under
an hour per epoch for the pilot. Generators are very unequal in size; `max_files` sets the budget
(estimate for all 72 generators, cap 5000, from the first file):

| max_files | downloaded | generators with 5000 | with ≥ 1000 |
|---|---|---|---|
| 100 | 0.5 TB | 13 | 33 |
| 200 | 1.1 TB | 21 | 47 |
| null (all 608) | 3.3 TB | 34 | 63 |

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
notebooks/        quickstart.ipynb
```
