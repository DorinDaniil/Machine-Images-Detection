# Machine-Images-Detection

Detection of generated and partly generated images. A frozen DINOv3 with a small head predicts,
for every patch, whether its pixels came out of a generator's decoder.

Training uses no generated images: fakes are real images passed through a generator's
autoencoder, `decode(encode(x))`, fully or inside a random region, saved with the same JPEG
history as the original, so only the decoder trace differs.

## Setup

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU
python -m pytest tests -q
```

All weights and data are on Hugging Face without access requests (DINOv3 through timm).

## Pilot

```bash
python scripts/run_train.py --config configs/train.yaml
python scripts/run_test.py --config configs/train.yaml
```

- reals: 2000 COCO photos and 400 rendered text pages, UI screens, charts and logos, split 70/15/15
  into train, calibration and test;
- training fakes: the train reals re-decoded by the SD and FLUX.1 autoencoders;
- test fakes: the test reals re-decoded by SDXL (seen family) and FLUX.2, Qwen-Image and DC-AE
  (unseen families), fully and in one region with a mask; 200 images of each of nine closed
  generators from Qwen-Image-Bench (GPT-Image-2, Nano Banana Pro and 2, FLUX.2 Pro, Seedream 4.5,
  Imagen 4, Qwen-Image, Hunyuan Image 3, Kling);
- threshold: 1 % false positives on the calibration reals; false positives are then measured on
  the test photos and test renders;
- protocols: `equalized` (every image resized, cropped and JPEG-compressed the same way, so size and
  file format cannot give the answer) and `native`;
- result: `runs/pilot/report.md` with AUROC, TPR at the threshold and pixel F1 per group.

Downloads about 6 GB; one GPU, about an hour.

## Layout

```
src/backbone.py   frozen DINOv3 (timm) -> patch features of chosen blocks
src/model.py      head, top-k pooling, save / load
src/losses.py     patch and image BCE
src/train.py      training loop
src/evaluate.py   scores, threshold from calibration reals, report table
src/data/         download.py (COCO, Qwen-Image-Bench), decoders.py (8 autoencoders, JPEG history),
                  render.py (text, UI, chart, logo reals), dataset.py
scripts/          run_train.py, run_test.py
configs/          train.yaml
notebooks/        quickstart.ipynb
```
