"""Smoke checks for PromptIR intermediate decoder supervision."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from net.model import PromptIR
from utils.submission_utils import load_promptir_checkpoint


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
    parser.add_argument("--dummy_size", type=int, default=64)
    parser.add_argument("--skip_checkpoint", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def build_deep_promptir(use_intermediate_supervision):
    return PromptIR(
        decoder=True,
        dim=48,
        num_blocks=[4, 6, 8, 10],
        num_refinement_blocks=6,
        prompt_len=8,
        use_task_prompt_routing=False,
        num_tasks=2,
        use_intermediate_supervision=use_intermediate_supervision,
    )


def assert_aux_outputs(model, x):
    pred = model(x)
    if not isinstance(pred, torch.Tensor):
        raise AssertionError(f"model(x) returned {type(pred)}, expected Tensor")
    if pred.shape != x.shape:
        raise AssertionError(f"final shape {tuple(pred.shape)} != {tuple(x.shape)}")

    outputs = model(x, return_aux=True)
    expected_keys = {"final", "aux_l1", "aux_l2"}
    missing = expected_keys.difference(outputs)
    if missing:
        raise AssertionError(f"Missing aux output keys: {sorted(missing)}")
    for key in expected_keys:
        if outputs[key].shape != x.shape:
            raise AssertionError(
                f"{key} shape {tuple(outputs[key].shape)} != {tuple(x.shape)}"
            )
    print("Forward checks: OK")


def assert_training_loss(model, x, target):
    outputs = model(x, return_aux=True)
    loss_final = F.l1_loss(outputs["final"], target)
    loss_aux_l1 = F.l1_loss(outputs["aux_l1"], target)
    loss_aux_l2 = F.l1_loss(outputs["aux_l2"], target)
    loss = loss_final + 0.10 * loss_aux_l1 + 0.05 * loss_aux_l2
    if not torch.isfinite(loss):
        raise AssertionError("Intermediate-supervision loss is non-finite")
    loss.backward()
    aux_has_grad = any(
        param.grad is not None
        for name, param in model.named_parameters()
        if name.startswith(("aux_head_l1.", "aux_head_l2."))
    )
    if not aux_has_grad:
        raise AssertionError("Auxiliary heads did not receive gradients")
    print("Training loss/backward checks: OK")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(0)
    print(f"Using device: {device}")

    model = build_deep_promptir(True).to(device).train()
    x = torch.rand(1, 3, args.dummy_size, args.dummy_size, device=device)
    target = torch.rand_like(x)
    assert_aux_outputs(model, x)
    assert_training_loss(model, x, target)

    model.eval()
    with torch.no_grad():
        val_pred = model(x).clamp(0.0, 1.0)
    if val_pred.shape != x.shape:
        raise AssertionError("Validation-style final output has wrong shape")
    print("Validation final-output check: OK")

    if not args.skip_checkpoint:
        checkpoint_path = Path(args.ckpt_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        checkpoint_model = build_deep_promptir(True)
        load_promptir_checkpoint(
            checkpoint_model,
            str(checkpoint_path),
            map_location="cpu",
            use_ema=True,
            allow_partial_load_for_intermediate_supervision=True,
        )
        print("Checkpoint partial-load check: OK")

    print("Intermediate supervision smoke checks: OK")


if __name__ == "__main__":
    main()
