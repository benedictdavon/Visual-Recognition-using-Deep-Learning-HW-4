"""Fit and apply tiny per-channel RGB calibration.

Calibration form:

    final_rgb = gain_rgb * pred_rgb + bias_rgb

The fit is constrained to small postprocessing changes, by default:

    gain_rgb in [0.98, 1.02]
    bias_rgb in [-2/255, +2/255]

The fitting pass runs the selected checkpoint on the fixed HW4 validation split,
fits channel-wise MSE-optimal gain/bias under the constraints, and reports raw
vs calibrated PSNR breakdowns.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from net.model import PromptIR
from utils.dataset_utils import HW4RestorationDataset
from utils.submission_utils import (
    detect_lwr_variant_from_checkpoint,
    forward_with_padding,
    load_promptir_checkpoint,
    predict_with_tta,
    save_npz_submission,
)


DEFAULT_E030_CKPT = (
    "output/runs/exp30_e029_ft224_lr1e5_30e/epoch012-psnr29.433.ckpt"
)
DEFAULT_E033_CKPT = (
    "output/runs/exp33_promptir_lwr_from_best_ft224/output/runs/"
    "E033_deep_promptir_lwr_from_best224/ft224_lwr/"
    "lwr_epoch008-psnr29.612.ckpt"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hw4_data_root", type=str, default="data")
    parser.add_argument("--hw4_split_file", type=str, default="splits/hw4_split_seed42.json")
    parser.add_argument("--hw4_val_per_task", type=int, default=160)
    parser.add_argument("--hw4_seed", type=int, default=42)
    parser.add_argument("--patch_size", type=int, default=224)

    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_E033_CKPT)
    parser.add_argument("--run_name", type=str, default="E033")
    parser.add_argument("--model_dim", type=int, default=56)
    parser.add_argument("--num_blocks", type=int, nargs=4, default=[4, 6, 8, 12])
    parser.add_argument("--num_refinement_blocks", type=int, default=6)
    parser.add_argument("--prompt_len", type=int, default=12)
    parser.add_argument("--disable_soft_prompt_routing", action="store_true", default=True)

    parser.add_argument("--use_local_weather_refine", action="store_true")
    parser.add_argument(
        "--lwr_variant",
        choices=["auto", "local", "notebook", "kaggle"],
        default="auto",
    )
    parser.add_argument("--lwr_expansion", type=float, default=1.0)
    parser.add_argument("--lwr_residual_scale_init", type=float, default=0.05)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable_ema", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--tile", action="store_true")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--tile_overlap", type=int, default=32)
    parser.add_argument("--gaussian_patch", action="store_true")
    parser.add_argument("--patch_grid", type=int, default=6)
    parser.add_argument("--gaussian_sigma_scale", type=float, default=0.25)
    parser.add_argument("--patch_batch_size", type=int, default=1)
    parser.add_argument("--tta", action="store_true")

    parser.add_argument("--gain_min", type=float, default=0.98)
    parser.add_argument("--gain_max", type=float, default=1.02)
    parser.add_argument(
        "--gain_steps",
        type=int,
        default=81,
        help="Per-channel gain grid count. 81 gives 0.0005 spacing over [0.98, 1.02].",
    )
    parser.add_argument("--bias_min_pixel", type=float, default=-2.0)
    parser.add_argument("--bias_max_pixel", type=float, default=2.0)

    parser.add_argument(
        "--smoke_max_samples_per_task",
        type=int,
        default=0,
        help="Use 1 or 2 for a quick correctness test. Keep 0 for full validation.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="output/rgb_calibration/e033_rgb_calibration.json",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="output/rgb_calibration/e033_rgb_calibration_metrics.csv",
    )
    parser.add_argument(
        "--input_npz",
        type=str,
        default="",
        help="Optional pred.npz to calibrate after fitting.",
    )
    parser.add_argument(
        "--output_npz",
        type=str,
        default="",
        help="Optional calibrated pred.npz output path.",
    )
    return parser.parse_args()


def resolve_device(device_name):
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def resolve_lwr_variant(ckpt_path, use_lwr, requested_variant, use_ema):
    if not use_lwr:
        return "local"
    if requested_variant != "auto":
        return requested_variant
    detected = detect_lwr_variant_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        use_ema=use_ema,
    )
    return detected or "local"


def build_model(args):
    lwr_variant = resolve_lwr_variant(
        args.ckpt_path,
        use_lwr=args.use_local_weather_refine,
        requested_variant=args.lwr_variant,
        use_ema=not args.disable_ema,
    )
    model = PromptIR(
        decoder=True,
        dim=args.model_dim,
        num_blocks=args.num_blocks,
        num_refinement_blocks=args.num_refinement_blocks,
        prompt_len=args.prompt_len,
        use_task_prompt_routing=not args.disable_soft_prompt_routing,
        num_tasks=2,
        use_local_weather_refine=args.use_local_weather_refine,
        lwr_expansion=args.lwr_expansion,
        lwr_residual_scale_init=args.lwr_residual_scale_init,
        lwr_variant=lwr_variant,
    )
    load_promptir_checkpoint(
        model,
        args.ckpt_path,
        map_location="cpu",
        use_ema=not args.disable_ema,
    )
    return model, lwr_variant


@torch.no_grad()
def predict_one(model, degraded, args):
    if args.tta:
        return predict_with_tta(
            model,
            degraded,
            multiple=8,
            use_tile=args.tile,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            use_gaussian_patch=args.gaussian_patch,
            patch_grid=args.patch_grid,
            gaussian_sigma_scale=args.gaussian_sigma_scale,
            patch_batch_size=args.patch_batch_size,
        )
    return forward_with_padding(
        model,
        degraded,
        multiple=8,
        use_tile=args.tile,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        use_gaussian_patch=args.gaussian_patch,
        patch_grid=args.patch_grid,
        gaussian_sigma_scale=args.gaussian_sigma_scale,
        patch_batch_size=args.patch_batch_size,
    )


def metadata_tasks(metadata):
    tasks = metadata["task"]
    if isinstance(tasks, str):
        return [tasks]
    return list(tasks)


def psnr_per_image(prediction, target, data_range=1.0):
    prediction = prediction.clamp(0.0, data_range)
    target = target.clamp(0.0, data_range)
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    return torch.where(
        mse <= 1e-12,
        torch.full_like(mse, 100.0),
        10.0 * torch.log10((data_range ** 2) / mse),
    )


def init_channel_stats():
    return {
        "n": torch.zeros(3, dtype=torch.float64),
        "sum_p": torch.zeros(3, dtype=torch.float64),
        "sum_y": torch.zeros(3, dtype=torch.float64),
        "sum_p2": torch.zeros(3, dtype=torch.float64),
        "sum_y2": torch.zeros(3, dtype=torch.float64),
        "sum_py": torch.zeros(3, dtype=torch.float64),
    }


def update_channel_stats(stats, pred, target):
    pred = pred.detach().double().cpu()
    target = target.detach().double().cpu()
    for channel in range(3):
        p = pred[:, channel].reshape(-1)
        y = target[:, channel].reshape(-1)
        stats["n"][channel] += p.numel()
        stats["sum_p"][channel] += p.sum()
        stats["sum_y"][channel] += y.sum()
        stats["sum_p2"][channel] += (p * p).sum()
        stats["sum_y2"][channel] += (y * y).sum()
        stats["sum_py"][channel] += (p * y).sum()


def fit_channel_calibration(stats, args):
    gains = []
    biases = []
    bias_min = float(args.bias_min_pixel) / 255.0
    bias_max = float(args.bias_max_pixel) / 255.0
    gain_grid = torch.linspace(
        float(args.gain_min),
        float(args.gain_max),
        int(args.gain_steps),
        dtype=torch.float64,
    )

    for channel in range(3):
        n = stats["n"][channel].clamp_min(1.0)
        sum_p = stats["sum_p"][channel]
        sum_y = stats["sum_y"][channel]
        sum_p2 = stats["sum_p2"][channel]
        sum_y2 = stats["sum_y2"][channel]
        sum_py = stats["sum_py"][channel]

        raw_bias = (sum_y - gain_grid * sum_p) / n
        bias_grid = raw_bias.clamp(bias_min, bias_max)
        sse = (
            gain_grid.square() * sum_p2
            + 2.0 * gain_grid * bias_grid * sum_p
            - 2.0 * gain_grid * sum_py
            + bias_grid.square() * n
            - 2.0 * bias_grid * sum_y
            + sum_y2
        )
        best_idx = int(torch.argmin(sse).item())
        gains.append(float(gain_grid[best_idx].item()))
        biases.append(float(bias_grid[best_idx].item()))

    return torch.tensor(gains, dtype=torch.float32), torch.tensor(biases, dtype=torch.float32)


def apply_calibration(tensor, gains, biases):
    gains = gains.to(device=tensor.device, dtype=tensor.dtype).view(1, 3, 1, 1)
    biases = biases.to(device=tensor.device, dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor * gains + biases).clamp(0.0, 1.0)


def summarize(values):
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def worst20(values):
    if not values:
        return float("nan")
    selected = sorted(values)[: min(20, len(values))]
    return float(sum(selected) / len(selected))


def evaluate_cached(cached_batches, gains, biases):
    raw = {"all": [], "rain": [], "snow": []}
    calibrated = {"all": [], "rain": [], "snow": []}
    for pred, target, tasks in cached_batches:
        raw_scores = psnr_per_image(pred, target)
        calibrated_pred = apply_calibration(pred, gains, biases)
        calibrated_scores = psnr_per_image(calibrated_pred, target)
        for idx, task in enumerate(tasks):
            raw_value = float(raw_scores[idx].item())
            calibrated_value = float(calibrated_scores[idx].item())
            raw["all"].append(raw_value)
            raw[task].append(raw_value)
            calibrated["all"].append(calibrated_value)
            calibrated[task].append(calibrated_value)

    return {
        "raw": {
            "overall_psnr": summarize(raw["all"]),
            "rain_psnr": summarize(raw["rain"]),
            "snow_psnr": summarize(raw["snow"]),
            "worst20_psnr": worst20(raw["all"]),
            "num_images": len(raw["all"]),
        },
        "calibrated": {
            "overall_psnr": summarize(calibrated["all"]),
            "rain_psnr": summarize(calibrated["rain"]),
            "snow_psnr": summarize(calibrated["snow"]),
            "worst20_psnr": worst20(calibrated["all"]),
            "num_images": len(calibrated["all"]),
        },
    }


def apply_calibration_to_npz(input_npz, output_npz, gains, biases):
    data = np.load(input_npz)
    predictions = {}
    gain_np = gains.numpy().astype(np.float32).reshape(3, 1, 1)
    bias_np = biases.numpy().astype(np.float32).reshape(3, 1, 1)
    for key in data.files:
        array = data[key]
        if array.shape[0] != 3:
            raise ValueError(f"{key}: expected CHW array with 3 channels, got {array.shape}")
        calibrated = (array.astype(np.float32) / 255.0) * gain_np + bias_np
        calibrated = np.rint(np.clip(calibrated, 0.0, 1.0) * 255.0).astype(np.uint8)
        predictions[key] = calibrated
    save_npz_submission(predictions, output_path=output_npz, expected_count=len(predictions))


def main():
    args = parse_args()
    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"

    print(f"Loading {args.run_name}: {args.ckpt_path}")
    model, lwr_variant = build_model(args)
    print(
        "Model settings:",
        f"use_lwr={args.use_local_weather_refine}",
        f"lwr_variant={lwr_variant}",
    )
    model.to(device).eval()

    dataset = HW4RestorationDataset(args, split="val")
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
    )
    stats = init_channel_stats()
    cached_batches = []
    seen_by_task = {"rain": 0, "snow": 0}

    for metadata, degraded, clean in tqdm(loader, desc="RGB calibration fit"):
        tasks = metadata_tasks(metadata)
        keep_indices = []
        for idx, task in enumerate(tasks):
            if (
                args.smoke_max_samples_per_task > 0
                and seen_by_task[task] >= args.smoke_max_samples_per_task
            ):
                continue
            keep_indices.append(idx)
        if not keep_indices:
            if args.smoke_max_samples_per_task > 0 and all(
                count >= args.smoke_max_samples_per_task
                for count in seen_by_task.values()
            ):
                break
            continue

        degraded = degraded.to(device)
        clean = clean.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = predict_one(model, degraded, args).float().clamp(0.0, 1.0)

        if len(keep_indices) != pred.shape[0]:
            keep_tensor = torch.tensor(keep_indices, device=pred.device, dtype=torch.long)
            pred = pred.index_select(0, keep_tensor)
            clean = clean.index_select(0, keep_tensor)
            tasks = [tasks[idx] for idx in keep_indices]

        update_channel_stats(stats, pred, clean)
        cached_batches.append((pred.detach().cpu(), clean.detach().cpu(), tasks))
        for task in tasks:
            seen_by_task[task] += 1

    gains, biases = fit_channel_calibration(stats, args)
    metrics = evaluate_cached(cached_batches, gains, biases)

    output = {
        "run_name": args.run_name,
        "ckpt_path": args.ckpt_path,
        "gain_rgb": [float(value) for value in gains.tolist()],
        "bias_rgb": [float(value) for value in biases.tolist()],
        "bias_rgb_pixel": [float(value * 255.0) for value in biases.tolist()],
        "constraints": {
            "gain_min": args.gain_min,
            "gain_max": args.gain_max,
            "gain_steps": args.gain_steps,
            "bias_min_pixel": args.bias_min_pixel,
            "bias_max_pixel": args.bias_max_pixel,
        },
        "metrics": metrics,
    }

    print()
    print("RGB calibration:")
    print("gain_rgb:", ", ".join(f"{value:.6f}" for value in output["gain_rgb"]))
    print("bias_rgb_pixel:", ", ".join(f"{value:.4f}" for value in output["bias_rgb_pixel"]))
    print("split,overall_psnr,rain_psnr,snow_psnr,worst20_psnr,num_images")
    for split_name in ["raw", "calibrated"]:
        row = metrics[split_name]
        print(
            f"{split_name},"
            f"{row['overall_psnr']:.4f},"
            f"{row['rain_psnr']:.4f},"
            f"{row['snow_psnr']:.4f},"
            f"{row['worst20_psnr']:.4f},"
            f"{row['num_images']}"
        )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved calibration JSON: {output_json}")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "overall_psnr",
                "rain_psnr",
                "snow_psnr",
                "worst20_psnr",
                "num_images",
            ],
        )
        writer.writeheader()
        for split_name in ["raw", "calibrated"]:
            writer.writerow({"split": split_name, **metrics[split_name]})
    print(f"Saved metrics CSV: {output_csv}")

    if args.input_npz or args.output_npz:
        if not args.input_npz or not args.output_npz:
            raise ValueError("--input_npz and --output_npz must be provided together.")
        apply_calibration_to_npz(args.input_npz, args.output_npz, gains, biases)
        print(f"Saved calibrated NPZ: {args.output_npz}")


if __name__ == "__main__":
    main()
