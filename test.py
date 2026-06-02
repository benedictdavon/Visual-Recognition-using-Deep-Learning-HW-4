import numpy as np
from pathlib import Path

def load_npz(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}

p30 = load_npz("path/to/e030_pred.npz")
p31 = load_npz("path/to/e031_pred.npz")
p33 = load_npz("path/to/e033_pred.npz")

def compare(a, b, name):
    keys = sorted(a.keys())
    exact_same = 0
    total_mae = 0.0
    total_max = 0.0

    for k in keys:
        x = a[k].astype(np.float32)
        y = b[k].astype(np.float32)
        diff = np.abs(x - y)
        if np.array_equal(a[k], b[k]):
            exact_same += 1
        total_mae += diff.mean()
        total_max = max(total_max, diff.max())

    print(name)
    print("exact same images:", exact_same, "/", len(keys))
    print("mean abs diff:", total_mae / len(keys))
    print("max abs diff:", total_max)
    print()

compare(p31, p33, "E031 vs E033")
compare(p30, p33, "E030 vs E033")
compare(p30, p31, "E030 vs E031")