"""Average multiple PromptIR checkpoints for optional HW4 inference."""

import argparse
from collections import OrderedDict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from utils.submission_utils import strip_lightning_prefix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--output", type=str, default="averaged_promptir.ckpt")
    parser.add_argument("--disable_ema", action="store_true")
    return parser.parse_args()


def extract_state(path, use_ema=True):
    checkpoint = torch.load(path, map_location="cpu")
    if use_ema and isinstance(checkpoint, dict) and "ema_state_dict" in checkpoint:
        state = checkpoint["ema_state_dict"]
        source = "ema_state_dict"
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state = strip_lightning_prefix(checkpoint["model"])
        source = "model"
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = strip_lightning_prefix(checkpoint["model_state_dict"])
        source = "model_state_dict"
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = strip_lightning_prefix(checkpoint["state_dict"])
        source = "state_dict"
    elif isinstance(checkpoint, dict):
        state = strip_lightning_prefix(checkpoint)
        source = "raw_state_dict"
    else:
        raise TypeError(f"Unsupported checkpoint type in {path}: {type(checkpoint)}")
    return state, source


def validate_compatible(states, paths):
    reference_keys = list(states[0].keys())
    reference_key_set = set(reference_keys)
    for path, state in zip(paths[1:], states[1:]):
        key_set = set(state.keys())
        missing = sorted(reference_key_set - key_set)
        extra = sorted(key_set - reference_key_set)
        if missing or extra:
            raise ValueError(
                f"Checkpoint key mismatch in {path}. "
                f"Missing: {missing[:5]} Extra: {extra[:5]}"
            )
        for key in reference_keys:
            if state[key].shape != states[0][key].shape:
                raise ValueError(
                    f"Shape mismatch for {key} in {path}: "
                    f"expected {tuple(states[0][key].shape)}, got {tuple(state[key].shape)}"
                )
    return reference_keys


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    extracted = [extract_state(path, use_ema=not args.disable_ema) for path in args.ckpts]
    states = [state for state, _ in extracted]
    sources = [source for _, source in extracted]
    keys = validate_compatible(states, args.ckpts)

    averaged = OrderedDict()
    for key in keys:
        values = [state[key] for state in states]
        if torch.is_floating_point(values[0]):
            averaged[key] = torch.stack(values, dim=0).mean(dim=0)
        else:
            averaged[key] = values[0]
    torch.save(averaged, output)
    print(f"Averaged {len(args.ckpts)} checkpoints")
    print(f"Sources: {', '.join(sources)}")
    print(f"Saved averaged checkpoint to {output}")


if __name__ == "__main__":
    main()
