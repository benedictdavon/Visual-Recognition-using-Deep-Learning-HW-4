"""Blend two HW4 pred.npz files in float space and save a new pred.npz."""

import argparse
from pathlib import Path

import numpy as np


def natural_key(name):
    stem = Path(name).stem
    return int(stem) if stem.isdigit() else stem


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_a", type=str, required=True)
    parser.add_argument("--pred_b", type=str, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output_npz", type=str, required=True)
    parser.add_argument("--expected_count", type=int, default=100)
    return parser.parse_args()


def load_npz(path):
    data = np.load(path)
    return {key: data[key] for key in data.files}


def main():
    args = parse_args()
    pred_a = load_npz(args.pred_a)
    pred_b = load_npz(args.pred_b)
    keys_a = set(pred_a.keys())
    keys_b = set(pred_b.keys())
    if keys_a != keys_b:
        missing_a = sorted(keys_b - keys_a, key=natural_key)[:5]
        missing_b = sorted(keys_a - keys_b, key=natural_key)[:5]
        raise RuntimeError(
            f"NPZ keys differ. Missing in A: {missing_a}; missing in B: {missing_b}"
        )
    if args.expected_count > 0 and len(keys_a) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} keys, got {len(keys_a)}")

    alpha = float(args.alpha)
    output = {}
    for key in sorted(keys_a, key=natural_key):
        a = pred_a[key].astype(np.float32)
        b = pred_b[key].astype(np.float32)
        if a.shape != b.shape:
            raise RuntimeError(f"{key}: shape mismatch {a.shape} vs {b.shape}")
        blended = alpha * a + (1.0 - alpha) * b
        output[key] = np.rint(np.clip(blended, 0.0, 255.0)).astype(np.uint8)

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **output)
    print(f"Saved blended NPZ: {output_path.resolve()}")
    print(f"alpha={alpha:g}, pred_a={args.pred_a}, pred_b={args.pred_b}")


if __name__ == "__main__":
    main()
