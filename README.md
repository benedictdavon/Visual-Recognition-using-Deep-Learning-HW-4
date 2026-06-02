# HW4 PromptIR Image Restoration

PromptIR-based image restoration repository for Visual Recognition using Deep
Learning, Homework 4. The task restores rain and snow degraded images and
exports an official `pred.npz` submission file.

The current best public submission is:

- `e108_tta_only..zip`
- public score: `30.80`
- checkpoint: `output/runs/exp37_e029_long224_safe_structural/epoch108-psnr30.190.ckpt`
- inference: x8 TTA only, no Gaussian patch inference

## Project Overview

This repo adapts PromptIR for the HW4 rain/snow restoration dataset:

- `net/` contains model code.
- `src/train.py` trains the local Lightning-based pipeline.
- `src/validate_hw4.py` validates checkpoints on the fixed split.
- `src/infer_hw4.py` creates `pred.npz`.
- `tools/create_submission.py` is a compatibility wrapper around `src/infer_hw4.py`.
- `src/average_checkpoints.py` averages compatible checkpoints for model soup tests.
- `utils/` contains dataset, padding, TTA, checkpoint loading, and submission helpers.

Generated artifacts such as datasets, checkpoints, outputs, logs, predictions,
and zip submissions are intentionally ignored by git.

## Environment Setup

Recommended local environment:

```powershell
conda env create -f requirements.yaml
conda activate prompt-ir
```

If that environment is unavailable, use a PyTorch environment with the usual
PromptIR dependencies:

```powershell
conda env create -f env.yml
conda activate promptir
pip install lightning tqdm pillow numpy torchvision einops
```

No new dependencies are required for the cleanup utilities.

## Dataset Layout

The default local data root is `data/`:

```text
data/
  train/
    degraded/
      rain-1.png
      snow-1.png
      ...
    clean/
      rain_clean-1.png
      snow_clean-1.png
      ...
  test/
    degraded/
      0.png
      1.png
      ...
      99.png
```

Use `--hw4_data_root` for training/validation and `--test_dir` for inference if
your data lives elsewhere.

## Training

Example local training command:

```powershell
python src/train.py `
  --hw4_data_root data `
  --epochs 120 `
  --batch_size 8 `
  --patch_size 128 `
  --num_gpus 1 `
  --ckpt_dir train_ckpt_hw4 `
  --run_name promptir_hw4
```

The fixed validation split is stored at:

```text
splits/hw4_split_seed42.json
```

Do not run expensive training during repository hygiene checks.

## Validation

Simple validation:

```powershell
python src/validate_hw4.py `
  --hw4_data_root data `
  --hw4_split_file splits/hw4_split_seed42.json `
  --ckpt_path output/runs/exp37_e029_long224_safe_structural/epoch108-psnr30.190.ckpt `
  --model_dim 56 `
  --num_blocks 4 6 8 12 `
  --num_refinement_blocks 6 `
  --prompt_len 12 `
  --disable_soft_prompt_routing
```

Validation with x8 TTA:

```powershell
python src/validate_hw4.py `
  --hw4_data_root data `
  --hw4_split_file splits/hw4_split_seed42.json `
  --ckpt_path output/runs/exp37_e029_long224_safe_structural/epoch108-psnr30.190.ckpt `
  --model_dim 56 `
  --num_blocks 4 6 8 12 `
  --num_refinement_blocks 6 `
  --prompt_len 12 `
  --disable_soft_prompt_routing `
  --tta
```

## Inference And Submission

Current best public setting: epoch 108 with x8 TTA only.

```powershell
python tools/create_submission.py `
  --test_dir data/test/degraded `
  --ckpt_path output/runs/exp37_e029_long224_safe_structural/epoch108-psnr30.190.ckpt `
  --output_npz output/runs/exp37_e029_long224_safe_structural/pred_e108_tta_only.npz `
  --model_dim 56 `
  --num_blocks 4 6 8 12 `
  --num_refinement_blocks 6 `
  --prompt_len 12 `
  --disable_soft_prompt_routing `
  --tta
```

Create the upload zip:

```powershell
Compress-Archive `
  -Path output/runs/exp37_e029_long224_safe_structural/pred_e108_tta_only.npz `
  -DestinationPath output/runs/exp37_e029_long224_safe_structural/submission_e108_tta_only.zip `
  -Force
```

Gaussian patch inference is still supported, but the current best run performed
worse with Gaussian patch + x8 TTA than with TTA only.

```powershell
python tools/create_submission.py `
  --test_dir data/test/degraded `
  --ckpt_path output/runs/exp37_e029_long224_safe_structural/epoch108-psnr30.190.ckpt `
  --output_npz output/runs/exp37_e029_long224_safe_structural/pred_e108_gauss36_x8.npz `
  --model_dim 56 `
  --num_blocks 4 6 8 12 `
  --num_refinement_blocks 6 `
  --prompt_len 12 `
  --disable_soft_prompt_routing `
  --gaussian_patch `
  --tile_size 128 `
  --patch_grid 6 `
  --patch_batch_size 1 `
  --tta
```

The official submission file must contain:

- filename keys such as `0.png`, `1.png`, ..., `99.png`;
- one `uint8` array per image;
- array shape `(3, H, W)`;
- values in `[0, 255]`.

Check a generated prediction file before zipping or uploading:

```powershell
python tools/check_submission_npz.py `
  output/runs/exp37_e029_long224_safe_structural/pred_e108_tta_only.npz `
  --test_dir data/test/degraded `
  --expected_count 100
```

## Model Soup

Average compatible checkpoints:

```powershell
python src/average_checkpoints.py `
  --ckpts `
  output/runs/exp37_e029_long224_safe_structural/epoch105-psnr30.189.ckpt `
  output/runs/exp37_e029_long224_safe_structural/epoch106-psnr30.189.ckpt `
  output/runs/exp37_e029_long224_safe_structural/epoch107-psnr30.189.ckpt `
  output/runs/exp37_e029_long224_safe_structural/epoch108-psnr30.190.ckpt `
  output/runs/exp37_e029_long224_safe_structural/epoch109-psnr30.189.ckpt `
  output/runs/exp37_e029_long224_safe_structural/last_e110.ckpt `
  --output output/runs/exp37_e029_long224_safe_structural/soup_e105_e110.ckpt
```

The `105-110` soup with TTA-only scored `30.79`, slightly below single
epoch 108 TTA-only at `30.80`.

## Best Public Score Snapshot

Submission budget at the latest snapshot: 3 of 5 daily submissions used,
15 of 99 total submissions used.

| Submission ID | File | Date | Public LB |
|---:|---|---|---:|
| 775607 | e108_tta_only..zip | 2026-06-02 17:32 | 30.80 |
| 775610 | soup_e105-e110_tta_only.zip | 2026-06-02 17:35 | 30.79 |
| 775083 | exp37_e029_long224_safe_structural.zip | 2026-06-02 12:12 | 30.65 |
| 759723 | ensemble_e030_e033_alpha050.zip | 2026-05-27 22:30 | 30.60 |
| 748696 | exp30_e29_ft224_lr1e5_e30.zip | 2026-05-23 01:37 | 30.60 |

Main inference note: TTA-only is currently stronger than Gaussian patch
inference for the best E034 checkpoint.

## Submission Checklist

- Confirm `pred.npz` has exactly 100 keys.
- Confirm keys match numeric test filenames.
- Confirm arrays are `uint8` and CHW.
- Run `tools/check_submission_npz.py` on the generated file.
- Zip only `pred.npz` at the archive root.
- Do not change checkpoint loading, EMA handling, TTA, Gaussian patch logic, or
  output format during cleanup.

## Report And GitHub Reminders

- Include the best public score and method name in the report.
- Mention that `epoch108 + TTA-only` is the current best public setting.
- Keep generated checkpoints, predictions, logs, datasets, and submissions out
  of GitHub.
- Submit source code plus report assets required by the homework instructions.
