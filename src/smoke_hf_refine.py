"""Smoke checks for Deep PromptIR + High-Frequency Refinement Head."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from net.model import PromptIR
from utils.submission_utils import strip_lightning_prefix


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
    parser.add_argument("--hf_refine_blocks", type=int, default=2)
    parser.add_argument("--hf_residual_scale_init", type=float, default=0.1)
    parser.add_argument("--max_param_increase_ratio", type=float, default=0.005)
    parser.add_argument("--skip_checkpoint", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def build_deep_promptir(use_hf_refine, args):
    return PromptIR(
        decoder=True,
        dim=48,
        num_blocks=[4, 6, 8, 10],
        num_refinement_blocks=6,
        prompt_len=8,
        use_task_prompt_routing=False,
        num_tasks=2,
        use_hf_refine=use_hf_refine,
        hf_refine_blocks=args.hf_refine_blocks,
        hf_residual_scale_init=args.hf_residual_scale_init,
    )


def count_params(model):
    return sum(param.numel() for param in model.parameters())


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


def assert_hf_partial_keys(missing_keys, unexpected_keys):
    bad_missing = [
        key for key in missing_keys
        if not key.startswith("hf_refine.")
    ]
    if bad_missing:
        raise AssertionError(f"Non-HF missing keys: {bad_missing}")
    if unexpected_keys:
        raise AssertionError(f"Unexpected keys while loading HF model: {unexpected_keys}")
    print("HF missing keys:")
    for key in missing_keys:
        print(f"  {key}")
    print("HF unexpected keys: <none>")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(0)
    print(f"Using device: {device}")

    base_model = build_deep_promptir(False, args)
    hf_model = build_deep_promptir(True, args)
    base_params = count_params(base_model)
    hf_params = count_params(hf_model)
    added_params = hf_params - base_params
    ratio = added_params / base_params
    print(f"Base params: {base_params:,}")
    print(f"HF params: {hf_params:,}")
    print(f"Added params: {added_params:,} ({ratio:.6%})")
    if ratio > args.max_param_increase_ratio:
        raise AssertionError(
            f"HF param increase ratio {ratio:.6%} exceeds "
            f"{args.max_param_increase_ratio:.6%}"
        )

    hf_model = hf_model.to(device).train()
    x = torch.randn(1, 3, args.dummy_size, args.dummy_size, device=device)
    y = hf_model(x)
    if y.shape != x.shape:
        raise AssertionError(f"HF output shape {tuple(y.shape)} != {tuple(x.shape)}")
    y.mean().backward()
    if not any(
        param.grad is not None
        for name, param in hf_model.named_parameters()
        if name.startswith("hf_refine.")
    ):
        raise AssertionError("No HF refinement parameter received gradients.")
    print(f"HF dummy forward/backward shape: {tuple(y.shape)}")

    if not args.skip_checkpoint:
        checkpoint_path = Path(args.ckpt_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        state_dict = load_state_dict(checkpoint_path)
        base_model.load_state_dict(state_dict, strict=True)
        load_result = build_deep_promptir(True, args).load_state_dict(
            state_dict,
            strict=False,
        )
        assert_hf_partial_keys(
            list(load_result.missing_keys),
            list(load_result.unexpected_keys),
        )

    print("HF refinement smoke checks: OK")


if __name__ == "__main__":
    main()
