"""Smoke checks for Deep PromptIR + Local Weather Refinement Head."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from net.model import PromptIR
from utils.submission_utils import predict_with_tta, strip_lightning_prefix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=(
            "output/runs/exp7_deeper_promptir_scratch128/"
            "epoch059-psnr28.855.ckpt"
        ),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dummy_size", type=int, default=128)
    parser.add_argument("--tta_size", type=int, default=64)
    parser.add_argument("--max_identity_diff", type=float, default=1e-5)
    parser.add_argument("--skip_checkpoint", action="store_true")
    parser.add_argument("--skip_tta", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg if torch.cuda.is_available() or device_arg == "cpu" else "cpu")


def build_deep_promptir(use_lwr):
    return PromptIR(
        decoder=True,
        dim=48,
        num_blocks=[4, 6, 8, 10],
        num_refinement_blocks=6,
        prompt_len=8,
        use_task_prompt_routing=False,
        num_tasks=2,
        use_local_weather_refine=use_lwr,
        lwr_expansion=1.0,
        lwr_use_directional=True,
        lwr_use_dilated=True,
        lwr_residual_scale_init=1.0,
    )


def load_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "ema_state_dict" in checkpoint:
        state_dict = checkpoint["ema_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")
    return strip_lightning_prefix(state_dict)


def assert_dummy_forward(model, x, name):
    with torch.no_grad():
        y = model(x)
    if y.shape != x.shape:
        raise AssertionError(f"{name} output shape {tuple(y.shape)} != {tuple(x.shape)}")
    print(f"{name} dummy forward shape: {tuple(y.shape)}")


def assert_lwr_partial_keys(missing_keys, unexpected_keys):
    bad_missing = [
        key for key in missing_keys
        if not key.startswith("local_weather_refine.")
    ]
    if bad_missing:
        raise AssertionError(f"Non-LWR missing keys: {bad_missing}")
    if unexpected_keys:
        raise AssertionError(f"Unexpected keys while loading LWR model: {unexpected_keys}")
    print("LWR missing keys:")
    for key in missing_keys:
        print(f"  {key}")
    print("LWR unexpected keys: <none>")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    print(f"Using device: {device}")

    base_model = build_deep_promptir(use_lwr=False).to(device).eval()
    lwr_model = build_deep_promptir(use_lwr=True).to(device).eval()

    x = torch.randn(1, 3, args.dummy_size, args.dummy_size, device=device)
    assert_dummy_forward(base_model, x, "Deep PromptIR")
    assert_dummy_forward(lwr_model, x, "Deep PromptIR + LWR")

    if not args.skip_checkpoint:
        checkpoint_path = Path(args.ckpt_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        base_model = base_model.cpu()
        lwr_model = lwr_model.cpu()
        state_dict = load_state_dict(checkpoint_path)
        base_model.load_state_dict(state_dict, strict=True)
        load_result = lwr_model.load_state_dict(state_dict, strict=False)
        missing_keys = list(load_result.missing_keys)
        unexpected_keys = list(load_result.unexpected_keys)
        assert_lwr_partial_keys(missing_keys, unexpected_keys)

        base_model = base_model.to(device).eval()
        lwr_model = lwr_model.to(device).eval()
        identity_x = torch.randn(1, 3, args.dummy_size, args.dummy_size, device=device)
        base_y = base_model(identity_x)
        lwr_y = lwr_model(identity_x)
        max_abs_diff = (base_y - lwr_y).abs().max().item()
        print(f"Identity max_abs_diff: {max_abs_diff:.8g}")
        if max_abs_diff > args.max_identity_diff:
            raise AssertionError(
                f"LWR identity diff {max_abs_diff:.8g} exceeds "
                f"{args.max_identity_diff:.8g}"
            )

    if not args.skip_tta:
        tta_x = torch.randn(1, 3, args.tta_size, args.tta_size, device=device)
        tta_y = predict_with_tta(
            lwr_model,
            tta_x,
            multiple=8,
            use_tile=True,
            tile_size=args.tta_size,
            tile_overlap=16,
        )
        if tta_y.shape != tta_x.shape:
            raise AssertionError(
                f"Tile+x8 TTA output shape {tuple(tta_y.shape)} != {tuple(tta_x.shape)}"
            )
        print(f"Tile+x8 TTA shape: {tuple(tta_y.shape)}")

    print("LWR smoke checks: OK")


if __name__ == "__main__":
    main()
