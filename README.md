# Machine-Images-Detection

Detector of AI-generated and AI-edited images that is built to generalise: to unseen
generators, to unseen visual domains (photos, art, logos, text, scientific figures) and
to fragment generations (inpainting, object edits), with a false-positive guarantee.

## Idea

Products change every month, decoder families change every few years. Every modern
generator, including autoregressive ones like GPT-Image and Nano Banana, ends with the
same operation: a decoder turns a compact latent into pixels and stamps its trace on every
pixel it produces. We teach the model to see that trace at **patch level**, and we never
show it a "real fake" during training:

1. **Aligned negatives.** A fake is a real image passed through a generator decoder,
   `decode(encode(x))`, using a small bank of VAEs (SD 1.x, SDXL, FLUX / SD3).
   Same content, same size, same JPEG history; only the decoder trace differs. The
   detector cannot learn the dataset, the domain or the prompt style, there is nothing
   else to learn.
2. **Local pastes without a global trace.** Re-decoded pixels are pasted into a random
   region while the pixels outside stay exactly original. This is the INP-X setting that
   breaks most inpainting detectors, which secretly rely on the global VAE blur.
3. **Soft targets.** Pasted pixels are blended with ratio `alpha`; the target map is
   `alpha` inside the region and 0 outside. The model learns a continuous axis
   "original -> decoder output" instead of a border between two piles.
4. **Frozen foundation backbone.** DINOv3 (or DINOv2 with registers) patch tokens at
   native resolution; only a small convolutional head is trained. Image score =
   mean of the top 5 % patch logits, so a small edit moves the image score.
5. **Degradations in training.** JPEG, down-scaling, blur and "laundering"
   (down-scale, JPEG, up-scale, JPEG) on every sample.
6. **Conformal thresholds on real images only.** The decision threshold is a quantile
   of scores on a held-out set of real images per content type, which gives
   `FPR <= alpha` with a finite-sample guarantee and no fakes in calibration.

Output per image: a probability, a heat map of generated regions and, with thresholds,
a verdict.

## Models

| Name | Backbone | Patch | Trainable | Config |
|---|---|---|---|---|
| trace_dinov3_b16 | `facebook/dinov3-vitb16-pretrain-lvd1689m` (gated, accept the license, `hf auth login`) | 16 | head only | [train_config_detector.yaml](trace_detector/configs/train_config_detector.yaml) |
| trace_dinov2_reg_b14 | `facebook/dinov2-with-registers-base` (Apache-2.0) | 14 | head only | same file, swap `model.backbone` |
| smoke | random 2-block ViT, CPU | 14 | head | [train_config_smoke.yaml](trace_detector/configs/train_config_smoke.yaml) |

Set `model.train_last_blocks: 2` to also fine-tune the last backbone blocks.

## Repository structure

```
trace_detector/                  self-contained detector package
├── configs/                     train / eval / smoke configs
├── dino_trace/                  the model, described plainly
│   ├── backbone.py              frozen DINOv3 / DINOv2 -> patch feature grid
│   ├── head.py                  patch logits + top-k pooling
│   ├── losses.py                soft BCE on patches and image, Dice
│   └── model.py                 TraceDetector
├── data/
│   ├── manifest.py              one CSV schema for every dataset
│   ├── decoders.py              VAE decoder bank (sd15, sdxl, flux)
│   ├── synth.py                 aligned negatives: full re-decode, local paste, mixing
│   ├── augment.py               JPEG / resize / blur / laundering
│   └── dataset.py               AlignedPairsDataset (train), ManifestDataset (eval)
├── model_detector_torch.py      inference wrapper: image -> score + heat map
├── train_detector.py            training loop
├── run_train_detector.py        CLI
├── evaluate_detector.py         AUROC, AP, Precision@Recall, pixel F1 / IoU per group
├── run_eval_detector.py         CLI
└── calibrate_detector.py        conformal thresholds on real images
data_tools/                      dataset collection
├── build_manifest.py            download sources, merge into one manifest
├── prepare_aligned_negatives.py re-decode real images through the VAE bank
├── generate_partial_fakes.py    diffusers inpainting / img2img with ground-truth masks
└── sources/                     one small adapter per dataset (see configs/data_sources_*.yaml)
configs/                         dataset documentation (ids, licenses, roles)
tests/                           CPU smoke test of the whole pipeline
```

## Usage

```bash
pip install -r requirements.txt
# GPU: pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pytest tests -q                          # CPU smoke test, ~10 s
```

1. Collect data into one manifest (`path,label,mask_path,generator,content_type,source,split`):

```bash
python -m data_tools.build_manifest --list
python -m data_tools.build_manifest --sources coco_val,wikiart,rvl_cdip,rocov2 --max-per-source 5000 --out data/manifests/train_real.csv
python -m data_tools.build_manifest --sources dailybench,mllm_bench,chameleon,wildrf,textfake,impostor --max-per-source 2000 --out data/manifests/test.csv
```

2. Re-decode the real images through the decoder bank (GPU, once):

```bash
python -m data_tools.prepare_aligned_negatives --manifest data/manifests/train_real.csv --out data/aligned --decoders sd15,sdxl,flux
```

3. Optionally generate partial fakes with masks for validation and testing:

```bash
python -m data_tools.generate_partial_fakes --manifest data/manifests/train_real.csv --out data/partial --pipeline sdxl-inpaint --limit 2000
```

4. Train, calibrate, evaluate:

```bash
python -m trace_detector.run_train_detector --config trace_detector/configs/train_config_detector.yaml
python -m trace_detector.calibrate_detector --weights checkpoints/trace_dinov3_b16/best.pt --manifest data/manifests/calib_real.csv --alpha 0.01 --out checkpoints/trace_dinov3_b16/thresholds.json
python -m trace_detector.run_eval_detector --config trace_detector/configs/eval_config_detector.yaml
```

5. Inference:

```python
from PIL import Image
from trace_detector.model_detector_torch import TraceDetectorInference

detector = TraceDetectorInference("checkpoints/trace_dinov3_b16/best.pt", device="cuda",
                                  thresholds="checkpoints/trace_dinov3_b16/thresholds.json")
result = detector.predict(Image.open("photo.jpg"), content_type="photo")
result["score"], result["verdict"], result["map"].shape
```

## Data plan

Training never uses images from a generator; it uses real images and their re-decoded
copies. Real images must cover every domain the detector will see, otherwise the
calibration and the content-type thresholds are meaningless.

| Role | Content | Sources (adapters in `data_tools/sources/`) |
|---|---|---|
| train real | photos | coco_val, open_images_sample, raise_raw |
| train real | art, drawings | wikiart, imagenet_sketch, tu_berlin_sketches |
| train real | graphics, logos, icons | logos, icons |
| train real | text, documents, screenshots | rvl_cdip, screenshots |
| train real | biomedical, scientific figures | medmnist, patchcamelyon, rocov2, bbbc, chest_xray, sci_figures |
| calib real | all of the above, held out | split `calib` |
| test fake, full | GPT-Image 1/2, Nano Banana 2, FLUX.2, Qwen-Image, Seedream, Midjourney, ... | dailybench, mllm_bench, community_forensics, so_fake, chameleon, wildrf, itw_sm, openfake, genimage |
| test fake, partial | edits with masks | dailybench (ManipulationBench), impostor, opensdid, inpx, so_fake, generate_partial_fakes |
| test, text-rich | screenshots, documents | textfake, local_folder |
| legacy | the previous project's data | legacy_artifact, legacy_defactify |

Licenses, ids and download modes are listed in `configs/data_sources_fake.yaml` and
`configs/data_sources_real.yaml`. Benchmarks are test-only: nothing from them is used
for training.

## Evaluation protocol

- Leave-one-decoder-family-out on the aligned pairs; never train on benchmark fakes.
- Report per source, per generator and per content type: AUROC, AP, balanced accuracy at
  the calibrated threshold, Precision at Recall 0.3 / 0.5 / 0.7, FPR on real images, and
  pixel F1 / IoU where masks exist.
- Keep a real-only `calib` split per content type; the FPR on it is the product metric.
