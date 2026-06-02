"""Validate an HW4 ``pred.npz`` submission file.

This checks the public submission format only. It does not need clean labels and
does not compute PSNR.
"""

import argparse
from pathlib import Path
import re

import numpy as np
from PIL import Image


def natural_key(text):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(text))
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=str, help="Path to pred.npz.")
    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/test/degraded",
        help="Optional test image directory for key and shape checks.",
    )
    parser.add_argument("--expected_count", type=int, default=100)
    return parser.parse_args()


def expected_from_test_dir(test_dir):
    root = Path(test_dir)
    if not root.is_dir():
        return None

    expected = {}
    for path in sorted(root.glob("*.png"), key=natural_key):
        with Image.open(path) as image:
            width, height = image.size
        expected[path.name] = (3, height, width)
    return expected


def expected_numeric_keys(count):
    return {f"{idx}.png" for idx in range(count)}


def validate_array(key, array, expected_shape=None):
    if array.dtype != np.uint8:
        raise AssertionError(f"{key}: expected uint8, got {array.dtype}")
    if array.ndim != 3 or array.shape[0] != 3:
        raise AssertionError(f"{key}: expected CHW with 3 channels, got {array.shape}")
    if expected_shape is not None and tuple(array.shape) != tuple(expected_shape):
        raise AssertionError(
            f"{key}: expected shape {expected_shape}, got {tuple(array.shape)}"
        )
    if array.size == 0:
        raise AssertionError(f"{key}: empty array")
    if array.min() < 0 or array.max() > 255:
        raise AssertionError(f"{key}: values outside [0, 255]")


def main():
    args = parse_args()
    path = Path(args.npz_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    expected = expected_from_test_dir(args.test_dir)
    expected_keys = set(expected) if expected is not None else expected_numeric_keys(
        args.expected_count
    )

    with np.load(path) as payload:
        keys = sorted(payload.files, key=natural_key)
        key_set = set(keys)
        missing = sorted(expected_keys - key_set, key=natural_key)
        extra = sorted(key_set - expected_keys, key=natural_key)
        if missing or extra:
            raise AssertionError(f"Key mismatch. Missing={missing[:5]} Extra={extra[:5]}")
        if len(keys) != args.expected_count:
            raise AssertionError(
                f"Expected {args.expected_count} predictions, got {len(keys)}"
            )

        for key in keys:
            expected_shape = expected[key] if expected is not None else None
            validate_array(key, payload[key], expected_shape=expected_shape)

    print(f"pred.npz format OK: {path}")
    print(f"num_images: {len(keys)}")
    print(f"first_keys: {keys[:5]}")
    if expected is not None:
        print(f"checked_shapes_against: {Path(args.test_dir)}")


if __name__ == "__main__":
    main()
