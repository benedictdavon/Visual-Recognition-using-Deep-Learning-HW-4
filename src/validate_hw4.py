"""Evaluate a PromptIR checkpoint on the fixed HW4 validation split."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from net.model import PromptIR
from utils.dataset_utils import HW4RestorationDataset
from utils.submission_utils import (
    calculate_psnr_torch,
    detect_lwr_variant_from_checkpoint,
    forward_with_padding,
    load_promptir_checkpoint,
    predict_with_tta,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hw4_data_root", type=str, default="data")
    parser.add_argument("--hw4_split_file", type=str, default="splits/hw4_split_seed42.json")
    parser.add_argument("--hw4_val_per_task", type=int, default=160)
    parser.add_argument("--hw4_seed", type=int, default=42)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_dim", type=int, default=48)
    parser.add_argument("--num_blocks", type=int, nargs=4, default=[4, 6, 6, 8])
    parser.add_argument("--num_refinement_blocks", type=int, default=4)
    parser.add_argument("--prompt_len", type=int, default=5)
    parser.add_argument("--disable_soft_prompt_routing", action="store_true")
    parser.add_argument("--use_local_weather_refine", action="store_true")
    parser.add_argument("--lwr_expansion", type=float, default=1.0)
    parser.add_argument(
        "--lwr_variant",
        choices=["auto", "local", "notebook", "kaggle"],
        default="auto",
        help="LWR implementation variant. Use auto for checkpoint-key detection.",
    )
    parser.add_argument(
        "--lwr_use_directional",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--lwr_use_dilated",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lwr_residual_scale_init", type=float, default=1.0)
    parser.add_argument("--use_hf_refine", action="store_true")
    parser.add_argument("--hf_refine_blocks", type=int, default=2)
    parser.add_argument("--hf_residual_scale_init", type=float, default=0.1)
    parser.add_argument("--allow_partial_load_for_hf_refine", action="store_true")
    parser.add_argument("--use_multiscale_context_fusion", action="store_true")
    parser.add_argument("--mscf_hidden_ratio", type=float, default=0.5)
    parser.add_argument("--mscf_residual_scale_init", type=float, default=0.05)
    parser.add_argument("--allow_partial_load_for_mscf", action="store_true")
    parser.add_argument("--use_intermediate_supervision", action="store_true")
    parser.add_argument(
        "--allow_partial_load_for_intermediate_supervision",
        action="store_true",
    )
    parser.add_argument("--use_soft_degradation_mask", action="store_true")
    parser.add_argument("--mask_hidden_dim", type=int, default=32)
    parser.add_argument("--mask_guidance_alpha", type=float, default=0.5)
    parser.add_argument(
        "--allow_partial_load_for_soft_degradation_mask",
        action="store_true",
    )
    parser.add_argument("--disable_ema", action="store_true")
    parser.add_argument("--tile", action="store_true")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--tile_overlap", type=int, default=32)
    parser.add_argument(
        "--gaussian_patch",
        action="store_true",
        help="Use Gaussian-weighted patch testing. For 256x256 HW4 images, "
             "--tile_size 128 --patch_grid 6 gives 36 overlapping patches.",
    )
    parser.add_argument("--patch_grid", type=int, default=6)
    parser.add_argument("--gaussian_sigma_scale", type=float, default=0.25)
    parser.add_argument(
        "--patch_batch_size",
        type=int,
        default=1,
        help="Number of Gaussian spatial patches to run per model forward.",
    )
    parser.add_argument("--tta", action="store_true")
    parser.add_argument(
        "--smoke_max_samples_per_task",
        type=int,
        default=0,
        help="Smoke-test limiter only. Keep 0 for real validation metrics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lwr_variant = args.lwr_variant
    if args.use_local_weather_refine and lwr_variant == "auto":
        detected_variant = detect_lwr_variant_from_checkpoint(
            args.ckpt_path,
            map_location="cpu",
            use_ema=not args.disable_ema,
        )
        lwr_variant = detected_variant or "local"
        print(f"Resolved LWR variant: {lwr_variant}")

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
        lwr_use_directional=args.lwr_use_directional,
        lwr_use_dilated=args.lwr_use_dilated,
        lwr_residual_scale_init=args.lwr_residual_scale_init,
        lwr_variant=lwr_variant,
        use_hf_refine=args.use_hf_refine,
        hf_refine_blocks=args.hf_refine_blocks,
        hf_residual_scale_init=args.hf_residual_scale_init,
        use_multiscale_context_fusion=args.use_multiscale_context_fusion,
        mscf_hidden_ratio=args.mscf_hidden_ratio,
        mscf_residual_scale_init=args.mscf_residual_scale_init,
        use_intermediate_supervision=args.use_intermediate_supervision,
        use_soft_degradation_mask=args.use_soft_degradation_mask,
        mask_hidden_dim=args.mask_hidden_dim,
        mask_guidance_alpha=args.mask_guidance_alpha,
    )
    total_params = sum(param.numel() for param in model.parameters())
    print(
        "Model initialization: "
        f"use_hf_refine={args.use_hf_refine}, "
        f"hf_refine_blocks={args.hf_refine_blocks}, "
        f"hf_residual_scale_init={args.hf_residual_scale_init}, "
        f"use_intermediate_supervision={args.use_intermediate_supervision}, "
        f"use_local_weather_refine={args.use_local_weather_refine}, "
        f"lwr_variant={lwr_variant}, "
        f"use_multiscale_context_fusion={args.use_multiscale_context_fusion}, "
        f"use_soft_degradation_mask={args.use_soft_degradation_mask}, "
        f"total_params={total_params:,}"
    )
    model = load_promptir_checkpoint(
        model,
        args.ckpt_path,
        map_location=device,
        use_ema=not args.disable_ema,
        allow_partial_load_for_hf_refine=args.allow_partial_load_for_hf_refine,
        allow_partial_load_for_mscf=args.allow_partial_load_for_mscf,
        allow_partial_load_for_intermediate_supervision=(
            args.allow_partial_load_for_intermediate_supervision
        ),
        allow_partial_load_for_soft_degradation_mask=(
            args.allow_partial_load_for_soft_degradation_mask
        ),
    )
    model.to(device)
    model.eval()

    dataset = HW4RestorationDataset(args, split="val")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    psnr_all = []
    psnr_by_task = {"rain": [], "snow": []}

    with torch.no_grad():
        for metadata, degraded, clean in tqdm(loader, desc="HW4 validation"):
            task = metadata["task"][0]
            if (
                args.smoke_max_samples_per_task > 0
                and len(psnr_by_task[task]) >= args.smoke_max_samples_per_task
            ):
                if all(
                    len(values) >= args.smoke_max_samples_per_task
                    for values in psnr_by_task.values()
                ):
                    break
                continue

            degraded = degraded.to(device)
            clean = clean.to(device)
            if args.tta:
                restored = predict_with_tta(
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
            else:
                restored = forward_with_padding(
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
            psnr = calculate_psnr_torch(restored, clean).item()
            psnr_all.append(psnr)
            psnr_by_task[task].append(psnr)
            if (
                args.smoke_max_samples_per_task > 0
                and all(
                    len(values) >= args.smoke_max_samples_per_task
                    for values in psnr_by_task.values()
                )
            ):
                break

    mean_all = sum(psnr_all) / len(psnr_all)
    print(f"val_psnr: {mean_all:.4f}")
    for task, values in psnr_by_task.items():
        if values:
            print(f"val_psnr_{task}: {sum(values) / len(values):.4f}")


if __name__ == "__main__":
    main()
