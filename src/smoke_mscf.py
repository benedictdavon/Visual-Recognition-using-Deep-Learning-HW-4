"""Smoke-test zero-init multi-scale context fusion compatibility."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from net.model import PromptIR  # noqa: E402


def main():
    torch.manual_seed(42)
    common_kwargs = {
        "decoder": True,
        "dim": 16,
        "num_blocks": [1, 1, 1, 1],
        "num_refinement_blocks": 1,
        "prompt_len": 2,
        "use_task_prompt_routing": False,
    }
    baseline = PromptIR(**common_kwargs)
    mscf_model = PromptIR(
        **common_kwargs,
        use_multiscale_context_fusion=True,
        mscf_hidden_ratio=0.5,
        mscf_residual_scale_init=0.05,
    )

    load_result = mscf_model.load_state_dict(baseline.state_dict(), strict=False)
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    bad_missing = [
        key for key in missing
        if not key.startswith("multiscale_context_fusion.")
    ]
    if bad_missing or unexpected:
        raise RuntimeError(
            "Unexpected checkpoint compatibility result: "
            f"bad_missing={bad_missing}, unexpected={unexpected}"
        )

    baseline.eval()
    mscf_model.eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        baseline_out = baseline(x)
        mscf_out = mscf_model(x)
    max_abs_diff = (baseline_out - mscf_out).abs().max().item()
    print(f"Missing MSCF keys: {len(missing)}")
    print(f"Max abs output diff after partial load: {max_abs_diff:.8f}")
    if max_abs_diff > 1e-6:
        raise RuntimeError("MSCF zero-init path changed baseline output.")


if __name__ == "__main__":
    main()
