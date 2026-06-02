"""Sweep YCbCr chroma residual shrinkage on HW4 validation.

For each prediction and degraded input:

    Y_final  = Y_pred
    Cb_final = Cb_degraded + gamma * (Cb_pred - Cb_degraded)
    Cr_final = Cr_degraded + gamma * (Cr_pred - Cr_degraded)

This keeps the model's luminance/structure correction while shrinking color
changes that can hurt PSNR if they are noisy.
"""

import argparse
import csv
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from net.model import PromptIR
from utils.color_postprocess import shrink_chroma_residual
from utils.dataset_utils import HW4RestorationDataset
from utils.submission_utils import (
    detect_lwr_variant_from_checkpoint,
    forward_with_padding,
    load_promptir_checkpoint,
    predict_with_tta,
)


DEFAULT_E030_CKPT = (
    "output/runs/exp30_e029_ft224_lr1e5_30e/epoch012-psnr29.433.ckpt"
)
DEFAULT_E033_CKPT = (
    "output/runs/exp33_promptir_lwr_from_best_ft224/output/runs/"
    "E033_deep_promptir_lwr_from_best224/ft224_lwr/"
    "lwr_epoch008-psnr29.612.ckpt"
)
DEFAULT_GAMMAS = [0.25, 0.50, 0.75, 1.00]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hw4_data_root", type=str, default="data")
    parser.add_argument("--hw4_split_file", type=str, default="splits/hw4_split_seed42.json")
    parser.add_argument("--hw4_val_per_task", type=int, default=160)
    parser.add_argument("--hw4_seed", type=int, default=42)
    parser.add_argument("--patch_size", type=int, default=224)

    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_E033_CKPT)
    parser.add_argument("--run_name", type=str, default="E033")
    parser.add_argument("--gammas", type=float, nargs="+", default=DEFAULT_GAMMAS)

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
    parser.add_argument(
        "--smoke_max_samples_per_task",
        type=int,
        default=0,
        help="Use 1 or 2 for a quick correctness test. Keep 0 for full validation.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="output/chroma_sweeps/e033_chroma_gamma_val_sweep.csv",
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


def psnr_per_image(prediction, target, data_range=1.0):
    prediction = prediction.clamp(0.0, data_range)
    target = target.clamp(0.0, data_range)
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    return torch.where(
        mse <= 1e-12,
        torch.full_like(mse, 100.0),
        10.0 * torch.log10((data_range ** 2) / mse),
    )


def metadata_tasks(metadata):
    tasks = metadata["task"]
    if isinstance(tasks, str):
        return [tasks]
    return list(tasks)


def mean_or_nan(values):
    if not values:
        return float("nan")
    return sum(values) / len(values)


def worst_k_mean(values, k=20):
    if not values:
        return float("nan")
    worst = sorted(values)[: min(k, len(values))]
    return sum(worst) / len(worst)


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
    gammas = [float(gamma) for gamma in args.gammas]
    scores = {
        gamma: {"all": [], "rain": [], "snow": []}
        for gamma in gammas
    }
    seen_by_task = {"rain": 0, "snow": 0}

    for metadata, degraded, clean in tqdm(loader, desc="Chroma gamma sweep"):
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
            pred = predict_one(model, degraded, args).float()

        for gamma in gammas:
            final = shrink_chroma_residual(pred, degraded.float(), gamma=gamma)
            psnr_values = psnr_per_image(final, clean)
            for idx in keep_indices:
                task = tasks[idx]
                psnr = psnr_values[idx].item()
                scores[gamma]["all"].append(psnr)
                scores[gamma][task].append(psnr)

        for idx in keep_indices:
            seen_by_task[tasks[idx]] += 1

    rows = []
    for gamma in gammas:
        values = scores[gamma]
        rows.append({
            "gamma": gamma,
            "overall_psnr": mean_or_nan(values["all"]),
            "rain_psnr": mean_or_nan(values["rain"]),
            "snow_psnr": mean_or_nan(values["snow"]),
            "worst20_psnr": worst_k_mean(values["all"], k=20),
            "num_images": len(values["all"]),
        })

    rows.sort(key=lambda row: row["overall_psnr"], reverse=True)
    print()
    print("chroma residual shrinkage:")
    print("Y=Y_pred, Cb/Cr=degraded + gamma * (pred - degraded)")
    print(f"run={args.run_name}")
    print("gamma,overall_psnr,rain_psnr,snow_psnr,worst20_psnr,num_images")
    for row in rows:
        print(
            f"{row['gamma']:.3f},"
            f"{row['overall_psnr']:.4f},"
            f"{row['rain_psnr']:.4f},"
            f"{row['snow_psnr']:.4f},"
            f"{row['worst20_psnr']:.4f},"
            f"{row['num_images']}"
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "gamma",
                "overall_psnr",
                "rain_psnr",
                "snow_psnr",
                "worst20_psnr",
                "num_images",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved sweep CSV: {output_csv}")


if __name__ == "__main__":
    main()
