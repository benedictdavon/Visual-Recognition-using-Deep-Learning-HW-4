# NYCU Visual Recognition HW4 - PromptIR Image Restoration

> Canonical portfolio entry point: [NYCU Visual Recognition using Deep Learning 2026](https://github.com/benedictdavon/nycu-visual-recognition-using-deep-learning-2026/tree/master/hw4-promptir-image-restoration). This repository is the historical standalone project snapshot.

## 周恭麟

## Introduction

This repository implements a PromptIR-based image restoration pipeline for
Visual Recognition using Deep Learning, 2026 Spring, Homework 4.

The model restores rain and snow degraded images. The goal is to maximize PSNR
on the CodaBench leaderboard while following the official restrictions:

- use a single PromptIR-based model;
- train from scratch using only the official HW4 training images;
- do not use external data;
- do not use pretrained weights;
- submit restored test images as `pred.npz`.

## Task Overview

- **Input:** degraded rain/snow images.
- **Output:** restored clean images.
- **Metric:** PSNR.
- **Test set:** 100 images named `0.png` to `99.png`.
- **Submission file:** `pred.npz`.

The official `pred.npz` format is dictionary-like:

- keys are original test filenames, for example `0.png`, `1.png`, ..., `99.png`;
- values are restored RGB images in CHW format;
- each value has shape `(3, H, W)`;
- each value is `uint8` in the range `[0, 255]`.

## Method Overview

The final method uses one PromptIR-based restoration model trained only on the
official paired HW4 data.

Pipeline summary:

1. Build an HW4-native paired dataset loader for `train/degraded` and
   `train/clean`.
2. Use a fixed balanced validation split with 160 rain and 160 snow validation
   images.
3. Train a single PromptIR-based model from scratch.
4. Use a deeper/wider PromptIR variant for higher rain/snow restoration
   capacity.
5. Fine-tune progressively with larger patch sizes to expose the model to more
   spatial context.
6. Select checkpoints using validation PSNR instead of training loss alone.
7. Restore test images with shape-preserving full-image inference and x8
   test-time augmentation.
8. Validate and save predictions in the official `pred.npz` format.

Gaussian-overlap patch inference is implemented and was tested, but it is not
the final recommended setting. The best public result so far came from
`epoch108-psnr30.190.ckpt` with x8 TTA only.

## Repository Structure

Important files and folders:

```text
net/
  model.py                         PromptIR model implementation

utils/
  dataset_utils.py                 HW4 train/val/test dataset helpers
  submission_utils.py              padding, TTA, Gaussian patching, checkpoint loading, pred.npz helpers
  color_postprocess.py             optional RGB/YCbCr postprocessing helpers
  image_utils.py, image_io.py      image utility code
  schedulers.py, loss_utils.py     training helpers

src/
  train.py                         training entry point
  validate_hw4.py                  validation on the fixed HW4 split
  infer_hw4.py                     canonical pred.npz inference script
  average_checkpoints.py           model soup / checkpoint averaging utility

tools/
  create_submission.py             compatibility wrapper around src/infer_hw4.py
  check_submission_npz.py          validates pred.npz structure and image shapes
  blend_npz.py                     blends two pred.npz files
  rgb_calibration.py               optional RGB calibration sweep/apply helper
  chroma_residual_sweep.py         optional chroma residual sweep helper
  residual_strength_sweep.py       optional residual-strength sweep helper

splits/
  hw4_split_seed42.json            fixed balanced validation split

notebooks/
  *.ipynb                          Kaggle/notebook experiment entry points

output/runs/                       local experiment outputs, checkpoints, metrics, and submissions
data/                              expected local dataset location; data is not included in git
```

Generated datasets, checkpoints, logs, prediction files, and zip submissions are
kept out of git.

## Environment Setup

The repo includes `requirements.yaml` for the current local Conda environment:

```bash
conda env create -f requirements.yaml
conda activate prompt-ir
```

If you need the older environment file:

```bash
conda env create -f env.yml
conda activate promptir
```

The `requirements.yaml` environment uses Python 3.10, PyTorch 2.2, CUDA 11.8,
Lightning, TorchMetrics, Pillow, NumPy, SciPy, OpenCV, scikit-image, tqdm, and
einops. Adjust the CUDA package if your local GPU driver requires a different
runtime.

## Dataset Layout

Place the official HW4 dataset under `data/`:

```text
data/
  train/
    degraded/
      rain-1.png
      ...
      rain-1600.png
      snow-1.png
      ...
      snow-1600.png
    clean/
      rain_clean-1.png
      ...
      rain_clean-1600.png
      snow_clean-1.png
      ...
      snow_clean-1600.png
  test/
    degraded/
      0.png
      ...
      99.png
```

If your data is stored elsewhere, pass the path with `--hw4_data_root` for
training/validation and `--test_dir` for inference.

## Training

Example local training command:

```bash
python src/train.py \
  --hw4_data_root data \
  --hw4_split_file splits/hw4_split_seed42.json \
  --epochs 120 \
  --batch_size 8 \
  --patch_size 128 \
  --num_gpus 1 \
  --ckpt_dir train_ckpt_hw4 \
  --run_name promptir_hw4
```

The best public result was produced by the longer notebook/Kaggle experiment
family, not by this short example command. The key final run was:

- run: `exp37_e029_long224_safe_structural`;
- architecture: `MODEL_DIM=56`, `NUM_BLOCKS=[4, 6, 8, 12]`,
  `NUM_REFINEMENT_BLOCKS=6`, `PROMPT_LEN=12`;
- final selected checkpoint: `epoch108-psnr30.190.ckpt`;
- inference: x8 TTA only.

## Validation

Validate the selected checkpoint on the fixed split:

```bash
python src/validate_hw4.py \
  --hw4_data_root data \
  --hw4_split_file splits/hw4_split_seed42.json \
  --ckpt_path output/runs/exp37_e029_long224_safe_structural/epoch108-psnr30.190.ckpt \
  --model_dim 56 \
  --num_blocks 4 6 8 12 \
  --num_refinement_blocks 6 \
  --prompt_len 12 \
  --disable_soft_prompt_routing
```

Validation with x8 TTA:

```bash
python src/validate_hw4.py \
  --hw4_data_root data \
  --hw4_split_file splits/hw4_split_seed42.json \
  --ckpt_path output/runs/exp37_e029_long224_safe_structural/epoch108-psnr30.190.ckpt \
  --model_dim 56 \
  --num_blocks 4 6 8 12 \
  --num_refinement_blocks 6 \
  --prompt_len 12 \
  --disable_soft_prompt_routing \
  --tta
```

## Inference And Submission

The recommended final inference setting is `epoch108 + x8 TTA only`.

Create `pred.npz`:

```bash
python tools/create_submission.py \
  --test_dir data/test/degraded \
  --ckpt_path output/runs/exp37_e029_long224_safe_structural/epoch108-psnr30.190.ckpt \
  --output_npz output/runs/exp37_e029_long224_safe_structural/e108_tta_only/pred.npz \
  --model_dim 56 \
  --num_blocks 4 6 8 12 \
  --num_refinement_blocks 6 \
  --prompt_len 12 \
  --disable_soft_prompt_routing \
  --tta
```

Validate the generated file:

```bash
python tools/check_submission_npz.py \
  output/runs/exp37_e029_long224_safe_structural/e108_tta_only/pred.npz \
  --test_dir data/test/degraded \
  --expected_count 100
```

Create an upload zip containing `pred.npz` at the archive root:

```powershell
Compress-Archive `
  -Path output/runs/exp37_e029_long224_safe_structural/e108_tta_only/pred.npz `
  -DestinationPath output/runs/exp37_e029_long224_safe_structural/e108_tta_only.zip `
  -Force
```

The inference script also supports Gaussian patch inference:

```bash
python tools/create_submission.py \
  --test_dir data/test/degraded \
  --ckpt_path output/runs/exp37_e029_long224_safe_structural/epoch108-psnr30.190.ckpt \
  --output_npz output/runs/exp37_e029_long224_safe_structural/e108_gaussian_x8/pred.npz \
  --model_dim 56 \
  --num_blocks 4 6 8 12 \
  --num_refinement_blocks 6 \
  --prompt_len 12 \
  --disable_soft_prompt_routing \
  --gaussian_patch \
  --tile_size 128 \
  --patch_grid 6 \
  --patch_batch_size 1 \
  --tta
```

This Gaussian-overlap setting was tested but not selected because it scored
around `30.65`, below TTA-only.

## Model Soup Utility

Checkpoint averaging is supported by `src/average_checkpoints.py`. The
`105-110` soup was tested, but did not beat the single epoch-108 checkpoint.

```bash
python src/average_checkpoints.py \
  --ckpts \
  output/runs/exp37_e029_long224_safe_structural/epoch105-psnr30.189.ckpt \
  output/runs/exp37_e029_long224_safe_structural/epoch106-psnr30.189.ckpt \
  output/runs/exp37_e029_long224_safe_structural/epoch107-psnr30.189.ckpt \
  output/runs/exp37_e029_long224_safe_structural/epoch108-psnr30.190.ckpt \
  output/runs/exp37_e029_long224_safe_structural/epoch109-psnr30.189.ckpt \
  output/runs/exp37_e029_long224_safe_structural/last_e110.ckpt \
  --output output/runs/exp37_e029_long224_safe_structural/soup_e105_e110.ckpt
```

## Performance Snapshot

Public CodaBench results from the latest notes:

![Public leaderboard row for the selected final submission](figures/leaderboard.png)

| Submission | Inference / Variant | Public PSNR |
|---|---|---:|
| `e108_tta_only..zip` | epoch 108 checkpoint + x8 TTA only | 30.80 |
| `soup_e105-e110_tta_only.zip` | checkpoint soup 105-110 + x8 TTA only | 30.79 |
| `exp37_e029_long224_safe_structural.zip` | E034 long 224 safe structural run with Gaussian/x8 export | 30.65 |
| `ensemble_e030_e033_alpha050.zip` | earlier ensemble baseline | 30.60 |
| `exp30_e29_ft224_lr1e5_e30.zip` | earlier E030 baseline | 30.60 |

Conclusion: the current best public setting is the single epoch-108 checkpoint
with x8 TTA only. Gaussian-overlap inference and the 105-110 model soup were
useful ablations but were not selected as the final submission method.

| Checkpoint / Epoch | Overall Val PSNR | Rain Val PSNR | Snow Val PSNR | Notes |
|---|---:|---:|---:|---|
| epoch 108 | 30.190 | 29.503 | 30.876 | selected final checkpoint |
| epoch 110 | 30.188 | 29.502 | 30.874 | late checkpoint, slightly below epoch 108 |

Additional report figures are saved under `figures/`:

- `training_curve.png`: combined E034 validation PSNR curve from `metrics.csv`
  through `metrics4.csv`;
- `confusion_matrix.png`: rain/snow degradation-routing probe on the fixed
  validation split, `[[157, 3], [0, 160]]`, accuracy `99.1%`;
- `qualitative_results.png`: validation examples restored by epoch 108 with
  x8 TTA;
- `failure_cases.png`: representative difficult examples with restored output,
  clean target, and error-map observations.

## Report Notes

The report should mention:

- single PromptIR-based model;
- trained from scratch;
- official training data only;
- fixed balanced validation split;
- deeper/wider PromptIR variant;
- progressive larger-patch fine-tuning;
- validation PSNR checkpoint selection;
- final epoch-108 + x8 TTA-only inference;
