"""Benchmark HW4 DataLoader throughput for different worker counts."""

import argparse
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dataset_utils import HW4RestorationDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hw4_data_root", type=str, default="data")
    parser.add_argument("--hw4_split_file", type=str, default="splits/hw4_split_seed42.json")
    parser.add_argument("--hw4_val_per_task", type=int, default=160)
    parser.add_argument("--hw4_seed", type=int, default=42)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--warmup_batches", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_args = SimpleNamespace(
        hw4_data_root=args.hw4_data_root,
        hw4_split_file=args.hw4_split_file,
        hw4_val_per_task=args.hw4_val_per_task,
        hw4_seed=args.hw4_seed,
        patch_size=args.patch_size,
    )
    dataset = HW4RestorationDataset(dataset_args, split="train")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    iterator = iter(loader)
    for _ in range(args.warmup_batches):
        next(iterator)

    count = 0
    start = time.perf_counter()
    for _ in range(args.num_batches):
        next(iterator)
        count += 1
    elapsed = time.perf_counter() - start

    print(f"workers: {args.num_workers}")
    print(f"batch_size: {args.batch_size}")
    print(f"patch_size: {args.patch_size}")
    print(f"batches: {count}")
    print(f"elapsed_sec: {elapsed:.3f}")
    print(f"batches_per_sec: {count / elapsed:.3f}")
    print(f"sec_per_batch: {elapsed / count:.3f}")


if __name__ == "__main__":
    main()
