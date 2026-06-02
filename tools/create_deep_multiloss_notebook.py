"""Create a self-contained Deep PromptIR multi-loss scratch notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(True),
    }


def notebook_model_source() -> str:
    source = (ROOT / "net" / "model.py").read_text(encoding="utf-8")
    source = source.replace("from einops import rearrange\n", "")
    source = source.replace("from einops.layers.torch import Rearrange\n", "")
    source = source.replace("from pdb import set_trace as stx\n", "")
    source = source.replace("import time\n", "")
    replacement = r'''
# Minimal local replacement for the small subset of einops.rearrange used here.
def rearrange(x, pattern, **kwargs):
    pattern = " ".join(pattern.split())
    if pattern == "b c h w -> b (h w) c":
        b, c, h, w = x.shape
        return x.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()
    if pattern == "b (h w) c -> b c h w":
        h = kwargs["h"]
        w = kwargs["w"]
        b, hw, c = x.shape
        if hw != h * w:
            raise RuntimeError(f"Token count {hw} does not match h*w={h * w}")
        return x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    if pattern == "b (head c) h w -> b head c (h w)":
        head = kwargs["head"]
        b, hc, h, w = x.shape
        if hc % head != 0:
            raise RuntimeError(f"Channels {hc} are not divisible by heads {head}")
        c = hc // head
        return x.reshape(b, head, c, h * w).contiguous()
    if pattern == "b head c (h w) -> b (head c) h w":
        h = kwargs["h"]
        w = kwargs["w"]
        b, head, c, hw = x.shape
        if hw != h * w:
            raise RuntimeError(f"Token count {hw} does not match h*w={h * w}")
        return x.reshape(b, head * c, h, w).contiguous()
    raise NotImplementedError(f"Unsupported rearrange pattern: {pattern}")
'''
    marker = "import numbers\n"
    source = source.replace(marker, marker + replacement + "\n", 1)
    return source


TITLE = r"""
# HW4 Deep PromptIR 128 Multi-Loss Scratch

This notebook runs the E018 experiment: train the same **Deep PromptIR 128** architecture from scratch with a 2026-safe multi-loss objective.

It is intentionally self-contained:

- no `from net.model import ...`
- no local helper library such as `hw4_lib.py`
- dataset, model, loss, training, validation, and inference functions are normal notebook cells
- train/val PNGs are cached in CPU RAM as uint8 tensors by default to avoid repeated PIL decode on Kaggle

The experiment uses **patch size 128** so it can be compared fairly against the Deep PromptIR 128 L1 scratch baseline:

```text
output/runs/exp7_deeper_promptir_scratch128/epoch059-psnr28.855.ckpt
simple validation: overall 28.855, rain 27.720, snow 29.990
```

Do not use VGG/perceptual loss here unless the TA explicitly allows pretrained loss networks. This notebook uses only L1, MS-SSIM, Charbonnier, and Sobel gradient losses.

Important: this notebook does **not** initialize from E015. It trains from random initialization.
"""


IMPORTS = r"""
import os
import sys
import csv
import json
import math
import time
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
if torch.cuda.is_available():
    print("CUDA capability:", torch.cuda.get_device_capability(0))
"""


CONFIG = r"""
# =========================
# CAPSLOCKED CONFIG
# =========================

DATA_ROOT = "/kaggle/input/YOUR_DATASET_NAME"
AUTO_FIND_DATA_ROOT = True

# Optional only: attach/upload E015 if you want to revalidate the reference checkpoint.
# Training below is from scratch and does not use this checkpoint.
REFERENCE_CKPT_PATH = None
AUTO_FIND_REFERENCE_CKPT = False
REFERENCE_CKPT_FILENAME_HINT = "epoch059-psnr28.855.ckpt"

OUTPUT_DIR = "/kaggle/working/output"
RUN_NAME = "exp18_deep128_multiloss_scratch"
SEED = 42

VAL_PER_TASK = 160
USE_FIXED_SPLIT = True
SPLIT_JSON_PATH = "/kaggle/working/hw4_split_seed42.json"

MODEL_DIM = 48
NUM_BLOCKS = [4, 6, 8, 10]
NUM_REFINEMENT_BLOCKS = 6
PROMPT_LEN = 8
DISABLE_SOFT_PROMPT_ROUTING = True

USE_SOFT_DEGRADATION_MASK = False
USE_MASK_LOSS = True
USE_LOCAL_WEIGHTED_LOSS = True
MASK_LOSS_WEIGHT = 0.05
LOCAL_LOSS_WEIGHT = 0.15
MASK_HIDDEN_DIM = 32
MASK_GUIDANCE_ALPHA = 0.5
MASK_LOSS_TYPE = "l1"  # "l1" or "mse"
PSEUDO_MASK_SMOOTH_KERNEL_SIZE = 3

PATCH_SIZE = 128
EPOCHS = 60
BATCH_SIZE = 4
LR = 2e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 8
GRAD_CLIP = 1.0
NUM_WORKERS = 2
PRECISION = "AMP"  # "AMP" or "FP32"
MATMUL_PRECISION = "medium"
USE_DATA_PARALLEL = True
# On Kaggle this should fit comfortably in 30 GB RAM and avoids repeated PNG/PIL decode.
CACHE_TRAIN_VAL_IMAGES_IN_RAM = True
CACHE_TEST_IMAGES_IN_RAM = False

# Configurable restoration loss. Keep these editable in this hyperparameter cell.
USE_MULTI_LOSS = True
USE_MS_SSIM = True
L1_WEIGHT = 1.0
SSIM_WEIGHT = 0.2
CHARBONNIER_WEIGHT = 0.05
GRADIENT_WEIGHT = 0.03
CHARBONNIER_EPS = 1e-3
MS_SSIM_LEVELS = 4
SSIM_WINDOW_SIZE = 7

TRAIN_LOG_EVERY_N_STEPS = 0  # 0 keeps component logging to once per epoch.
VALIDATE_REFERENCE_CKPT_BEFORE_TRAIN = False
VALIDATE_EVERY_EPOCH = True
SAVE_TOP_K = 3

USE_TILED_VALIDATION = True
TILE_SIZE = 256
TILE_OVERLAP = 32
USE_X8_TTA = True

MAKE_SUBMISSION = False
TEST_DIR = None  # None means DATA_ROOT/test/degraded
SUBMISSION_NPZ_PATH = "/kaggle/working/pred.npz"
SUBMISSION_ZIP_PATH = "/kaggle/working/submission.zip"

KNOWN_BASELINE = {
    "overall": 28.855,
    "rain": 27.720,
    "snow": 29.990,
    "name": "E015 Deep PromptIR 128 L1 baseline",
}

RUNNING_ON_KAGGLE = (
    "KAGGLE_URL_BASE" in os.environ
    or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    or (Path("/kaggle/input").exists() and Path("/kaggle/working").exists())
)


def localize_kaggle_working_path(path_value):
    if RUNNING_ON_KAGGLE:
        return path_value
    prefix = "/kaggle/working/"
    if isinstance(path_value, str) and path_value.startswith(prefix):
        return str(Path("kaggle_working") / path_value[len(prefix):])
    return path_value


OUTPUT_DIR = localize_kaggle_working_path(OUTPUT_DIR)
SPLIT_JSON_PATH = localize_kaggle_working_path(SPLIT_JSON_PATH)
SUBMISSION_NPZ_PATH = localize_kaggle_working_path(SUBMISSION_NPZ_PATH)
SUBMISSION_ZIP_PATH = localize_kaggle_working_path(SUBMISSION_ZIP_PATH)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
torch.set_float32_matmul_precision(MATMUL_PRECISION)

print("RUN_NAME:", RUN_NAME)
print("RUNNING_ON_KAGGLE:", RUNNING_ON_KAGGLE)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("PATCH_SIZE:", PATCH_SIZE)
print("CACHE_TRAIN_VAL_IMAGES_IN_RAM:", CACHE_TRAIN_VAL_IMAGES_IN_RAM)
print("CACHE_TEST_IMAGES_IN_RAM:", CACHE_TEST_IMAGES_IN_RAM)
print("Loss weights:", {
    "use_multi_loss": USE_MULTI_LOSS,
    "use_ms_ssim": USE_MS_SSIM,
    "l1": L1_WEIGHT,
    "ssim": SSIM_WEIGHT,
    "charbonnier": CHARBONNIER_WEIGHT,
    "gradient": GRADIENT_WEIGHT,
})
print("Soft degradation mask:", {
    "enabled": USE_SOFT_DEGRADATION_MASK,
    "mask_loss": USE_MASK_LOSS,
    "local_weighted_loss": USE_LOCAL_WEIGHTED_LOSS,
    "mask_loss_weight": MASK_LOSS_WEIGHT,
    "local_loss_weight": LOCAL_LOSS_WEIGHT,
    "mask_guidance_alpha": MASK_GUIDANCE_ALPHA,
})
"""


DATA = r"""
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def has_hw4_layout(path):
    path = Path(path)
    return (
        (path / "train" / "degraded").is_dir()
        and (path / "train" / "clean").is_dir()
        and (path / "test" / "degraded").is_dir()
    )


def find_hw4_root(preferred, auto=True):
    preferred = Path(preferred)
    if preferred.exists() and has_hw4_layout(preferred):
        return preferred
    if not auto:
        raise FileNotFoundError(f"DATA_ROOT does not match expected HW4 layout: {preferred}")

    search_roots = [Path("/kaggle/input"), Path.cwd()]
    candidates = []
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for root, _, _ in os.walk(search_root):
            root_path = Path(root)
            if has_hw4_layout(root_path):
                candidates.append(root_path)
    if not candidates:
        raise FileNotFoundError("Could not auto-find train/degraded, train/clean, test/degraded")
    return sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0]


def find_reference_checkpoint(preferred, auto=True):
    if preferred is None:
        return None
    preferred = Path(preferred)
    if preferred.exists():
        return str(preferred)
    if not auto:
        raise FileNotFoundError(preferred)

    search_roots = [Path("/kaggle/input"), Path.cwd(), Path("/kaggle/working")]
    matches = []
    for root in search_roots:
        if root.exists():
            matches.extend(root.rglob(REFERENCE_CKPT_FILENAME_HINT))
    if not matches:
        fallback = Path("output/runs/exp7_deeper_promptir_scratch128") / REFERENCE_CKPT_FILENAME_HINT
        if fallback.exists():
            return str(fallback)
        raise FileNotFoundError(
            f"Could not auto-find {REFERENCE_CKPT_FILENAME_HINT}. "
            "Attach the checkpoint or set REFERENCE_CKPT_PATH."
        )
    return str(sorted(matches, key=lambda p: (len(str(p)), str(p)))[0])


def parse_task_index(filename):
    stem = Path(filename).stem
    task, idx = stem.split("-", 1)
    return task, int(idx)


def clean_name_for(degraded_name):
    task, idx = parse_task_index(degraded_name)
    return f"{task}_clean-{idx}.png"


def natural_image_key(path):
    p = Path(path)
    if p.stem.isdigit():
        return (0, int(p.stem))
    task, idx = parse_task_index(p.name)
    return (1 if task == "rain" else 2, idx)


def build_or_load_split(data_root):
    split_path = Path(SPLIT_JSON_PATH)
    if USE_FIXED_SPLIT and split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        print("Loaded split:", split_path)
        return split

    degraded_dir = Path(data_root) / "train" / "degraded"
    rain = sorted(degraded_dir.glob("rain-*.png"), key=lambda p: parse_task_index(p.name)[1])
    snow = sorted(degraded_dir.glob("snow-*.png"), key=lambda p: parse_task_index(p.name)[1])
    rng = random.Random(SEED)
    rain_names = [p.name for p in rain]
    snow_names = [p.name for p in snow]
    rng.shuffle(rain_names)
    rng.shuffle(snow_names)
    split = {
        "seed": SEED,
        "val_per_task": VAL_PER_TASK,
        "val": {
            "rain": sorted(rain_names[:VAL_PER_TASK]),
            "snow": sorted(snow_names[:VAL_PER_TASK]),
        },
        "train": {
            "rain": sorted(rain_names[VAL_PER_TASK:]),
            "snow": sorted(snow_names[VAL_PER_TASK:]),
        },
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print("Saved split:", split_path)
    return split


def pil_to_tensor(path):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def pil_to_uint8_tensor(path):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def image_to_float_tensor(tensor):
    if torch.is_floating_point(tensor):
        return tensor.float().contiguous()
    return tensor.float().div_(255.0).contiguous()


class HW4RestorationDataset(Dataset):
    def __init__(self, data_root, split, split_name, patch_size=128, augment=True, cache_images=True):
        self.data_root = Path(data_root)
        self.split_name = split_name
        self.patch_size = patch_size
        self.augment = augment and split_name == "train"
        self.cache_images = cache_images
        self.degraded_dir = self.data_root / "train" / "degraded"
        self.clean_dir = self.data_root / "train" / "clean"
        self.items = []
        for task in ["rain", "snow"]:
            for degraded_name in split[split_name][task]:
                self.items.append({
                    "degraded": degraded_name,
                    "clean": clean_name_for(degraded_name),
                    "task": task,
                    "task_id": 0 if task == "rain" else 1,
                })
        if split_name == "train":
            random.Random(SEED + 123).shuffle(self.items)
        self.cache = {}
        if self.cache_images:
            self._build_cache()

    def _build_cache(self):
        unique_paths = []
        for item in self.items:
            unique_paths.append(self.degraded_dir / item["degraded"])
            unique_paths.append(self.clean_dir / item["clean"])
        iterator = tqdm(
            unique_paths,
            desc=f"cache {self.split_name} images",
            dynamic_ncols=True,
        ) if tqdm is not None else unique_paths
        start = time.time()
        total_bytes = 0
        for path in iterator:
            key = str(path)
            if key in self.cache:
                continue
            tensor = pil_to_uint8_tensor(path)
            self.cache[key] = tensor
            total_bytes += tensor.numel() * tensor.element_size()
        elapsed = time.time() - start
        print(
            f"Cached {len(self.cache)} {self.split_name} tensors "
            f"({total_bytes / 1024 ** 2:.1f} MiB) in {elapsed:.1f}s"
        )

    def load_image(self, path):
        if self.cache_images:
            return self.cache[str(path)]
        return pil_to_uint8_tensor(path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        degraded = self.load_image(self.degraded_dir / item["degraded"])
        clean = self.load_image(self.clean_dir / item["clean"])
        if self.split_name == "train":
            degraded, clean = self.random_crop_pair(degraded, clean, self.patch_size)
            if self.augment:
                if random.random() < 0.5:
                    degraded = degraded.flip(-1)
                    clean = clean.flip(-1)
                if random.random() < 0.5:
                    degraded = degraded.flip(-2)
                    clean = clean.flip(-2)
        degraded = image_to_float_tensor(degraded)
        clean = image_to_float_tensor(clean)
        meta = {
            "filename": item["degraded"],
            "task": item["task"],
            "task_id": item["task_id"],
        }
        return meta, degraded, clean

    @staticmethod
    def random_crop_pair(degraded, clean, patch_size):
        _, h, w = degraded.shape
        if h < patch_size or w < patch_size:
            pad_h = max(0, patch_size - h)
            pad_w = max(0, patch_size - w)
            mode = "reflect" if h > pad_h and w > pad_w else "replicate"
            degraded = F.pad(degraded.unsqueeze(0), (0, pad_w, 0, pad_h), mode=mode).squeeze(0)
            clean = F.pad(clean.unsqueeze(0), (0, pad_w, 0, pad_h), mode=mode).squeeze(0)
            _, h, w = degraded.shape
        top = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)
        return degraded[:, top:top + patch_size, left:left + patch_size], clean[:, top:top + patch_size, left:left + patch_size]


class HW4TestDataset(Dataset):
    def __init__(self, test_dir, cache_images=False):
        self.test_dir = Path(test_dir)
        self.files = sorted(self.test_dir.glob("*.png"), key=natural_image_key)
        if not self.files:
            raise FileNotFoundError(f"No PNG files found in {self.test_dir}")
        self.cache_images = cache_images
        self.cache = {}
        if self.cache_images:
            iterator = tqdm(
                self.files,
                desc="cache test images",
                dynamic_ncols=True,
            ) if tqdm is not None else self.files
            for path in iterator:
                self.cache[str(path)] = pil_to_uint8_tensor(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        if self.cache_images:
            tensor = self.cache[str(path)]
        else:
            tensor = pil_to_uint8_tensor(path)
        tensor = image_to_float_tensor(tensor)
        _, h, w = tensor.shape
        return {"filename": path.name, "height": h, "width": w}, tensor


seed_everything(SEED)
DATA_ROOT = str(find_hw4_root(DATA_ROOT, AUTO_FIND_DATA_ROOT))
REFERENCE_CKPT_PATH = find_reference_checkpoint(
    REFERENCE_CKPT_PATH,
    AUTO_FIND_REFERENCE_CKPT,
)
SPLIT = build_or_load_split(DATA_ROOT)

print("Selected DATA_ROOT:", DATA_ROOT)
print("REFERENCE_CKPT_PATH:", REFERENCE_CKPT_PATH)
print("Training initialization: scratch / random weights")
print("Train rain/snow:", len(SPLIT["train"]["rain"]), len(SPLIT["train"]["snow"]))
print("Val rain/snow:", len(SPLIT["val"]["rain"]), len(SPLIT["val"]["snow"]))

for required in [
    Path(DATA_ROOT) / "train" / "degraded",
    Path(DATA_ROOT) / "train" / "clean",
    Path(DATA_ROOT) / "test" / "degraded",
]:
    assert required.is_dir(), f"Missing required folder: {required}"

print("First train rain files:", SPLIT["train"]["rain"][:3])
print("First val snow files:", SPLIT["val"]["snow"][:3])
"""


MODEL_CELL_PREFIX = r"""
# Exact PromptIR model definitions are embedded in this cell.
# This keeps the notebook self-contained while preserving checkpoint key compatibility.
"""


LOSS_CELL = r"""
try:
    from pytorch_msssim import ms_ssim as _pt_ms_ssim
    from pytorch_msssim import ssim as _pt_ssim
    HAS_PYTORCH_MSSSIM = True
except Exception:
    _pt_ms_ssim = None
    _pt_ssim = None
    HAS_PYTORCH_MSSSIM = False


def gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = torch.outer(g, g)
    return window_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def ssim_components(x, y, window_size=7, sigma=1.5, data_range=1.0):
    x = x.float().clamp(0.0, data_range)
    y = y.float().clamp(0.0, data_range)
    channels = x.shape[1]
    window = gaussian_window(window_size, sigma, channels, x.device, x.dtype)
    padding = window_size // 2
    mu_x = F.conv2d(x, window, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=channels)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    cs = ((2.0 * sigma_xy + c2) / (sigma_x2 + sigma_y2 + c2)).clamp(0.0, 1.0)
    ssim = (((2.0 * mu_xy + c1) / (mu_x2 + mu_y2 + c1)) * cs).clamp(0.0, 1.0)
    return ssim.flatten(1).mean(dim=1), cs.flatten(1).mean(dim=1)


def local_ssim_value(x, y, window_size=7):
    ssim, _ = ssim_components(x, y, window_size=window_size)
    return ssim.mean()


def ms_ssim_weights(levels, device=None, dtype=None):
    levels = int(levels)
    base = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=device, dtype=dtype)
    if levels < 1 or levels > len(base):
        raise ValueError(f"MS_SSIM_LEVELS must be in [1, {len(base)}], got {levels}")
    weights = base[:levels]
    return weights / weights.sum()


def local_ms_ssim_value(x, y, levels=4, window_size=7):
    weights = ms_ssim_weights(levels, x.device, x.dtype)
    mcs = []
    for level in range(levels):
        ssim, cs = ssim_components(x, y, window_size=window_size)
        if level < levels - 1:
            mcs.append(cs)
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            y = F.avg_pool2d(y, kernel_size=2, stride=2)
    value = torch.ones_like(ssim)
    for cs_value, weight in zip(mcs, weights[:-1]):
        value = value * cs_value.clamp_min(1e-6).pow(weight)
    value = value * ssim.clamp_min(1e-6).pow(weights[-1])
    return value.mean()


def structural_similarity_loss(pred, target, use_ms_ssim=True, levels=4, window_size=7):
    # SSIM/MS-SSIM measures structural similarity. Clamp only this branch to the valid image range.
    pred_for_ssim = pred.float().clamp(0.0, 1.0)
    target_for_ssim = target.float().clamp(0.0, 1.0)
    if HAS_PYTORCH_MSSSIM:
        if use_ms_ssim:
            weights = ms_ssim_weights(levels).tolist()
            value = _pt_ms_ssim(
                pred_for_ssim,
                target_for_ssim,
                data_range=1.0,
                size_average=True,
                win_size=window_size,
                weights=weights,
            )
        else:
            value = _pt_ssim(
                pred_for_ssim,
                target_for_ssim,
                data_range=1.0,
                size_average=True,
                win_size=window_size,
            )
    elif use_ms_ssim:
        value = local_ms_ssim_value(pred_for_ssim, target_for_ssim, levels=levels, window_size=window_size)
    else:
        value = local_ssim_value(pred_for_ssim, target_for_ssim, window_size=window_size)
    return 1.0 - value.clamp(0.0, 1.0)


def charbonnier_loss(pred, target, eps=1e-3):
    # Charbonnier is a smooth L1-like penalty, robust to outlier pixels.
    return torch.sqrt((pred - target).pow(2) + eps * eps).mean()


def sobel_gradient_loss(pred, target):
    # Gradient loss compares RGB edge maps so rain streak/snow particle boundaries are penalized.
    pred = pred.float()
    target = target.float()
    channels = pred.shape[1]
    kx = pred.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
    ky = pred.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3) / 8.0
    kx = kx.repeat(channels, 1, 1, 1)
    ky = ky.repeat(channels, 1, 1, 1)
    pred_dx = F.conv2d(pred, kx, padding=1, groups=channels)
    pred_dy = F.conv2d(pred, ky, padding=1, groups=channels)
    target_dx = F.conv2d(target, kx, padding=1, groups=channels)
    target_dy = F.conv2d(target, ky, padding=1, groups=channels)
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def detached_loss_dict(total, l1, ssim_loss, charb, grad):
    return {
        "loss_l1": l1.detach(),
        "loss_ssim": ssim_loss.detach(),
        "loss_charbonnier": charb.detach(),
        "loss_gradient": grad.detach(),
        "loss_total": total.detach(),
    }


class RestorationMultiLoss(nn.Module):
    def __init__(
        self,
        use_ms_ssim=True,
        l1_weight=1.0,
        ssim_weight=0.2,
        charbonnier_weight=0.05,
        gradient_weight=0.03,
        charbonnier_eps=1e-3,
        ms_ssim_levels=4,
        ssim_window_size=7,
    ):
        super().__init__()
        self.use_ms_ssim = use_ms_ssim
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.charbonnier_weight = charbonnier_weight
        self.gradient_weight = gradient_weight
        self.charbonnier_eps = charbonnier_eps
        self.ms_ssim_levels = ms_ssim_levels
        self.ssim_window_size = ssim_window_size

    def forward(self, pred, target):
        pred_f = pred.float()
        target_f = target.float()
        # L1 keeps the optimization directly aligned with pixel reconstruction.
        l1 = F.l1_loss(pred_f, target_f)
        ssim_loss = structural_similarity_loss(
            pred_f,
            target_f,
            use_ms_ssim=self.use_ms_ssim,
            levels=self.ms_ssim_levels,
            window_size=self.ssim_window_size,
        )
        charb = charbonnier_loss(pred_f, target_f, eps=self.charbonnier_eps)
        grad = sobel_gradient_loss(pred_f, target_f)
        total = (
            self.l1_weight * l1
            + self.ssim_weight * ssim_loss
            + self.charbonnier_weight * charb
            + self.gradient_weight * grad
        )
        return total, detached_loss_dict(total, l1, ssim_loss, charb, grad)


class RestorationL1Loss(nn.Module):
    def forward(self, pred, target):
        l1 = F.l1_loss(pred.float(), target.float())
        zero = l1.detach().new_zeros(())
        return l1, detached_loss_dict(l1, l1, zero, zero, zero)


def make_soft_pseudo_degradation_mask(degraded, clean, smooth_kernel_size=3, eps=1e-6):
    with torch.no_grad():
        diff = torch.abs(degraded - clean).mean(dim=1, keepdim=True)
        diff_min = diff.amin(dim=(2, 3), keepdim=True)
        diff_max = diff.amax(dim=(2, 3), keepdim=True)
        pseudo_mask = (diff - diff_min) / (diff_max - diff_min + eps)
        smooth_kernel_size = int(smooth_kernel_size)
        if smooth_kernel_size > 1:
            if smooth_kernel_size % 2 == 0:
                smooth_kernel_size += 1
            pseudo_mask = F.avg_pool2d(
                pseudo_mask,
                kernel_size=smooth_kernel_size,
                stride=1,
                padding=smooth_kernel_size // 2,
            )
    return pseudo_mask.clamp(0.0, 1.0).to(dtype=degraded.dtype)


def soft_mask_supervision_loss(pred_mask, pseudo_mask, loss_type="l1"):
    if loss_type == "l1":
        return F.l1_loss(pred_mask, pseudo_mask)
    if loss_type == "mse":
        return F.mse_loss(pred_mask, pseudo_mask)
    raise ValueError(f"Unsupported mask loss type: {loss_type}")


def local_weighted_l1_loss(pred, target, weight_mask):
    return (weight_mask.to(dtype=pred.dtype) * torch.abs(pred - target)).mean()


def build_restoration_loss():
    if USE_MULTI_LOSS:
        return RestorationMultiLoss(
            use_ms_ssim=USE_MS_SSIM,
            l1_weight=L1_WEIGHT,
            ssim_weight=SSIM_WEIGHT,
            charbonnier_weight=CHARBONNIER_WEIGHT,
            gradient_weight=GRADIENT_WEIGHT,
            charbonnier_eps=CHARBONNIER_EPS,
            ms_ssim_levels=MS_SSIM_LEVELS,
            ssim_window_size=SSIM_WINDOW_SIZE,
        )
    return RestorationL1Loss()


print("Using pytorch_msssim:", HAS_PYTORCH_MSSSIM)
print("Loss mode:", "multi-loss" if USE_MULTI_LOSS else "plain L1")
print("Loss weights:", {
    "l1": L1_WEIGHT,
    "ssim_or_ms_ssim": SSIM_WEIGHT,
    "charbonnier": CHARBONNIER_WEIGHT,
    "gradient": GRADIENT_WEIGHT,
})
"""


LOSS_SANITY_CHECK = r"""
_loss_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_loss_fn = build_restoration_loss().to(_loss_device)
_pred = torch.rand(1, 3, 128, 128, device=_loss_device, requires_grad=True)
_target = torch.rand(1, 3, 128, 128, device=_loss_device)
_total_loss, _loss_parts = _loss_fn(_pred, _target)
print("Sanity total loss:", float(_total_loss.detach().cpu()))
print("Sanity loss components:", {key: float(value.detach().cpu()) for key, value in _loss_parts.items()})
_total_loss.backward()
assert _pred.grad is not None
assert torch.isfinite(_pred.grad).all()
print("Loss backward sanity check: OK")
del _loss_fn, _pred, _target, _total_loss, _loss_parts
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"""


TRAINING_UTILS = r"""
def build_deep_promptir():
    return PromptIR(
        decoder=True,
        dim=MODEL_DIM,
        num_blocks=NUM_BLOCKS,
        num_refinement_blocks=NUM_REFINEMENT_BLOCKS,
        prompt_len=PROMPT_LEN,
        use_task_prompt_routing=not DISABLE_SOFT_PROMPT_ROUTING,
        num_tasks=2,
        use_local_weather_refine=False,
        use_soft_degradation_mask=USE_SOFT_DEGRADATION_MASK,
        mask_hidden_dim=MASK_HIDDEN_DIM,
        mask_guidance_alpha=MASK_GUIDANCE_ALPHA,
    )


def strip_prefix_if_all_keys(state, prefix):
    if state and all(key.startswith(prefix) for key in state.keys()):
        return {key[len(prefix):]: value for key, value in state.items()}
    return state


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "ema_state_dict" in checkpoint:
        print("Using ema_state_dict from checkpoint")
        state = checkpoint["ema_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        print("Using Lightning state_dict from checkpoint")
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        print("Using model state from checkpoint")
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        print("Using raw state_dict checkpoint")
        state = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")
    state = strip_prefix_if_all_keys(state, "net.")
    state = strip_prefix_if_all_keys(state, "module.")
    return state


def load_promptir_checkpoint(model, ckpt_path, strict=True, map_location="cpu"):
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    state = extract_state_dict(checkpoint)
    result = model.load_state_dict(state, strict=strict)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    print("Missing keys:", missing if missing else "<none>")
    print("Unexpected keys:", unexpected if unexpected else "<none>")
    if strict and (missing or unexpected):
        raise RuntimeError("Strict checkpoint load failed")
    return model


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(path, model, epoch, val_metrics, config=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": unwrap_model(model).state_dict(),
        "epoch": epoch,
        "val_metrics": val_metrics,
        "config": config or {},
    }, path)


def model_summary(model, device):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Total parameters: {total / 1e6:.3f}M")
    print(f"Trainable parameters: {trainable / 1e6:.3f}M")
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, PATCH_SIZE, PATCH_SIZE, device=device)
        y = model(x)
    print("Dummy input/output:", tuple(x.shape), tuple(y.shape))


def gpu_memory_text():
    if not torch.cuda.is_available():
        return "cpu"
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved = torch.cuda.memory_reserved() / 1024 ** 3
    return f"{allocated:.2f}G/{reserved:.2f}G"


def make_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def pad_to_multiple(tensor, multiple=8):
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, h, w
    mode = "reflect" if pad_h < h and pad_w < w else "replicate"
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
    return padded, h, w


def unpad(tensor, h, w):
    return tensor[..., :h, :w]


def tile_positions(length, tile, stride):
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile, stride))
    positions.append(length - tile)
    return sorted(set(positions))


def model_forward(model, x):
    y = model(x)
    if isinstance(y, dict):
        return y["final"]
    if isinstance(y, (tuple, list)):
        y = y[0]
    return y


def tile_forward(model, tensor, tile_size=256, tile_overlap=32):
    _, _, h, w = tensor.shape
    tile = min(tile_size, h, w)
    tile = max(8, tile - tile % 8)
    overlap = min(tile_overlap, tile - 8)
    stride = tile - overlap
    if tile >= h and tile >= w:
        return model_forward(model, tensor)

    output = torch.zeros_like(tensor)
    weight = torch.zeros_like(tensor)
    for top in tile_positions(h, tile, stride):
        for left in tile_positions(w, tile, stride):
            patch = tensor[..., top:top + tile, left:left + tile]
            pred = model_forward(model, patch)
            output[..., top:top + tile, left:left + tile] += pred
            weight[..., top:top + tile, left:left + tile] += 1
    return output / weight.clamp_min(1e-8)


def forward_with_padding(model, tensor, use_tile=False, tile_size=256, tile_overlap=32):
    padded, h, w = pad_to_multiple(tensor, multiple=8)
    if use_tile:
        output = tile_forward(model, padded, tile_size=tile_size, tile_overlap=tile_overlap)
    else:
        output = model_forward(model, padded)
    return unpad(output, h, w).clamp(0.0, 1.0)


def augment_tensor(tensor, mode):
    if mode == 0:
        return tensor
    if mode == 1:
        return tensor.flip(-1)
    if mode == 2:
        return tensor.flip(-2)
    if mode == 3:
        return tensor.flip(-1).flip(-2)
    if mode == 4:
        return tensor.transpose(-1, -2)
    if mode == 5:
        return tensor.transpose(-1, -2).flip(-1)
    if mode == 6:
        return tensor.transpose(-1, -2).flip(-2)
    if mode == 7:
        return tensor.transpose(-1, -2).flip(-1).flip(-2)
    raise ValueError(mode)


def deaugment_tensor(tensor, mode):
    if mode == 0:
        return tensor
    if mode == 1:
        return tensor.flip(-1)
    if mode == 2:
        return tensor.flip(-2)
    if mode == 3:
        return tensor.flip(-1).flip(-2)
    if mode == 4:
        return tensor.transpose(-1, -2)
    if mode == 5:
        return tensor.flip(-1).transpose(-1, -2)
    if mode == 6:
        return tensor.flip(-2).transpose(-1, -2)
    if mode == 7:
        return tensor.flip(-1).flip(-2).transpose(-1, -2)
    raise ValueError(mode)


def predict_with_tta(model, tensor, use_tile=True, tile_size=256, tile_overlap=32):
    preds = []
    for mode in range(8):
        aug = augment_tensor(tensor, mode)
        pred = forward_with_padding(
            model,
            aug,
            use_tile=use_tile,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        preds.append(deaugment_tensor(pred, mode))
    return torch.stack(preds, dim=0).mean(dim=0).clamp(0.0, 1.0)


def psnr_tensor(pred, target):
    pred = pred.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)
    mse = (pred - target).pow(2).flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


@torch.no_grad()
def validate_model(model, dataset, device, use_tile=False, use_tta=False, desc="validation"):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    scores = []
    by_task = {"rain": [], "snow": []}
    iterator = tqdm(loader, desc=desc, dynamic_ncols=True) if tqdm is not None else loader
    for meta, degraded, clean in iterator:
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        amp_enabled = PRECISION == "AMP" and device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            if use_tta:
                pred = predict_with_tta(
                    model,
                    degraded,
                    use_tile=use_tile,
                    tile_size=TILE_SIZE,
                    tile_overlap=TILE_OVERLAP,
                )
            else:
                pred = forward_with_padding(
                    model,
                    degraded,
                    use_tile=use_tile,
                    tile_size=TILE_SIZE,
                    tile_overlap=TILE_OVERLAP,
                )
        score = psnr_tensor(pred, clean).item()
        task = meta["task"][0]
        scores.append(score)
        by_task[task].append(score)
        if tqdm is not None:
            iterator.set_postfix({
                "psnr": f"{np.mean(scores):.3f}",
                "rain": f"{np.mean(by_task['rain']):.3f}" if by_task["rain"] else "n/a",
                "snow": f"{np.mean(by_task['snow']):.3f}" if by_task["snow"] else "n/a",
            })
    return {
        "overall": float(np.mean(scores)),
        "rain": float(np.mean(by_task["rain"])),
        "snow": float(np.mean(by_task["snow"])),
    }
"""


TRAINING_RUN = r"""
def build_dataloaders():
    train_set = HW4RestorationDataset(
        DATA_ROOT,
        SPLIT,
        "train",
        patch_size=PATCH_SIZE,
        augment=True,
        cache_images=CACHE_TRAIN_VAL_IMAGES_IN_RAM,
    )
    val_set = HW4RestorationDataset(
        DATA_ROOT,
        SPLIT,
        "val",
        patch_size=PATCH_SIZE,
        augment=False,
        cache_images=CACHE_TRAIN_VAL_IMAGES_IN_RAM,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        drop_last=True,
    )
    print("Train samples:", len(train_set), "Val samples:", len(val_set))
    print("Train batches per epoch:", len(train_loader))
    return train_loader, val_set


def validate_checkpoint(ckpt_path, use_tile=False, use_tta=False, desc="checkpoint validation"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_deep_promptir().to(device)
    load_promptir_checkpoint(model, ckpt_path, strict=True, map_location=device)
    val_set = HW4RestorationDataset(
        DATA_ROOT,
        SPLIT,
        "val",
        patch_size=PATCH_SIZE,
        augment=False,
        cache_images=CACHE_TRAIN_VAL_IMAGES_IN_RAM,
    )
    result = validate_model(model, val_set, device, use_tile=use_tile, use_tta=use_tta, desc=desc)
    print(desc, result)
    return result


def print_delta(metrics, reference=KNOWN_BASELINE):
    print("Reference:", reference["name"])
    for key in ["overall", "rain", "snow"]:
        delta = metrics[key] - reference[key]
        print(f"{key}: {metrics[key]:.4f}  baseline={reference[key]:.4f}  delta={delta:+.4f}")


def train_multiloss_patch128():
    seed_everything(SEED)
    out_dir = Path(OUTPUT_DIR) / "runs" / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    config_path = out_dir / "config.json"
    config = {
        "run_name": RUN_NAME,
        "initialization": "scratch",
        "reference_ckpt": REFERENCE_CKPT_PATH,
        "patch_size": PATCH_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "cache_train_val_images_in_ram": CACHE_TRAIN_VAL_IMAGES_IN_RAM,
        "cache_test_images_in_ram": CACHE_TEST_IMAGES_IN_RAM,
        "use_multi_loss": USE_MULTI_LOSS,
        "use_ms_ssim": USE_MS_SSIM,
        "charbonnier_eps": CHARBONNIER_EPS,
        "loss_weights": {
            "l1": L1_WEIGHT,
            "ssim_or_ms_ssim": SSIM_WEIGHT,
            "charbonnier": CHARBONNIER_WEIGHT,
            "gradient": GRADIENT_WEIGHT,
        },
        "model_dim": MODEL_DIM,
        "num_blocks": NUM_BLOCKS,
        "num_refinement_blocks": NUM_REFINEMENT_BLOCKS,
        "prompt_len": PROMPT_LEN,
        "use_soft_degradation_mask": USE_SOFT_DEGRADATION_MASK,
        "mask_hidden_dim": MASK_HIDDEN_DIM,
        "mask_guidance_alpha": MASK_GUIDANCE_ALPHA,
        "mask_loss_weight": MASK_LOSS_WEIGHT,
        "local_loss_weight": LOCAL_LOSS_WEIGHT,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, val_set = build_dataloaders()

    model = build_deep_promptir().to(device)
    print("Training from scratch: no checkpoint is loaded into the model.")
    model_summary(model, device)

    if USE_DATA_PARALLEL and torch.cuda.device_count() > 1:
        print("Using DataParallel on", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = max(1, len(train_loader) * EPOCHS)
    warmup_steps = max(0, len(train_loader) * WARMUP_EPOCHS)
    scheduler = make_scheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=PRECISION == "AMP" and device.type == "cuda")
    loss_fn = build_restoration_loss().to(device)

    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_total_loss", "train_l1", "train_ssim",
            "train_charbonnier", "train_gradient", "train_mask",
            "train_local_weighted", "val_psnr",
            "val_rain", "val_snow", "lr", "elapsed_sec",
        ])

    best_psnr = -1.0
    best_ckpt = None
    top_ckpts = []
    history = []

    for epoch in range(EPOCHS):
        model.train()
        start = time.time()
        loss_keys = [
            "loss_total",
            "loss_l1",
            "loss_ssim",
            "loss_charbonnier",
            "loss_gradient",
            "loss_mask",
            "loss_local_weighted",
        ]
        running = {key: 0.0 for key in loss_keys}
        steps = 0
        iterator = tqdm(train_loader, desc=f"epoch {epoch:03d}/{EPOCHS - 1:03d}", dynamic_ncols=True) if tqdm is not None else train_loader
        for meta, degraded, clean in iterator:
            degraded = degraded.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            amp_enabled = PRECISION == "AMP" and device.type == "cuda"
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                if USE_SOFT_DEGRADATION_MASK:
                    model_output = model(degraded, return_aux=True)
                    pred = model_output["final"]
                else:
                    model_output = None
                    pred = model(degraded)
                loss, parts = loss_fn(pred, clean)
                parts = dict(parts)
                mask_loss = pred.new_zeros(())
                local_loss = pred.new_zeros(())
                if USE_SOFT_DEGRADATION_MASK:
                    pseudo_mask = make_soft_pseudo_degradation_mask(
                        degraded,
                        clean,
                        smooth_kernel_size=PSEUDO_MASK_SMOOTH_KERNEL_SIZE,
                    )
                    if USE_MASK_LOSS and MASK_LOSS_WEIGHT > 0:
                        mask_loss = soft_mask_supervision_loss(
                            model_output["pred_mask"],
                            pseudo_mask,
                            loss_type=MASK_LOSS_TYPE,
                        )
                        loss = loss + MASK_LOSS_WEIGHT * mask_loss
                    if USE_LOCAL_WEIGHTED_LOSS and LOCAL_LOSS_WEIGHT > 0:
                        local_loss = local_weighted_l1_loss(pred, clean, pseudo_mask)
                        loss = loss + LOCAL_LOSS_WEIGHT * local_loss
                parts["loss_mask"] = mask_loss.detach()
                parts["loss_local_weighted"] = local_loss.detach()
                parts["loss_total"] = loss.detach()

            if not torch.isfinite(loss):
                print("Skipping non-finite loss at epoch", epoch, "step", steps)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            steps += 1
            for key in running:
                running[key] += float(parts[key].item())

            if tqdm is not None:
                iterator.set_postfix({
                    "loss": f"{running['loss_total'] / steps:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "mem": gpu_memory_text(),
                })

            if TRAIN_LOG_EVERY_N_STEPS and steps % TRAIN_LOG_EVERY_N_STEPS == 0:
                print(
                    f"epoch={epoch} step={steps}/{len(train_loader)} "
                    f"loss={running['loss_total'] / steps:.5f} "
                    f"grad_norm={float(grad_norm):.3f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"mem={gpu_memory_text()}"
                )

        avg = {key: running[key] / max(1, steps) for key in running}
        val = {"overall": float("nan"), "rain": float("nan"), "snow": float("nan")}
        if VALIDATE_EVERY_EPOCH:
            val = validate_model(
                unwrap_model(model),
                val_set,
                device,
                use_tile=False,
                use_tta=False,
                desc=f"simple val epoch {epoch:03d}",
            )

        elapsed = time.time() - start
        lr_now = optimizer.param_groups[0]["lr"]
        ckpt_name = f"epoch{epoch:03d}-psnr{val['overall']:.3f}.ckpt"
        ckpt_path = out_dir / ckpt_name
        save_checkpoint(ckpt_path, model, epoch, val, config=config)
        save_checkpoint(out_dir / "last.ckpt", model, epoch, val, config=config)

        top_ckpts.append((val["overall"], ckpt_path))
        top_ckpts = sorted(top_ckpts, key=lambda item: item[0], reverse=True)
        for _, path in top_ckpts[SAVE_TOP_K:]:
            if path.exists():
                path.unlink()
        top_ckpts = top_ckpts[:SAVE_TOP_K]

        if val["overall"] > best_psnr:
            best_psnr = val["overall"]
            best_ckpt = out_dir / "best.ckpt"
            save_checkpoint(best_ckpt, model, epoch, val, config=config)

        row = {
            "epoch": epoch,
            "train": avg,
            "val": val,
            "lr": lr_now,
            "elapsed_sec": elapsed,
            "checkpoint": str(ckpt_path),
        }
        history.append(row)
        with metrics_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, avg["loss_total"], avg["loss_l1"], avg["loss_ssim"],
                avg["loss_charbonnier"], avg["loss_gradient"], avg["loss_mask"],
                avg["loss_local_weighted"], val["overall"], val["rain"],
                val["snow"], lr_now, elapsed,
            ])
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        print(
            f"Epoch {epoch:03d} | train_total={avg['loss_total']:.5f} | "
            f"l1={avg['loss_l1']:.5f} | ssim={avg['loss_ssim']:.5f} | "
            f"charb={avg['loss_charbonnier']:.5f} | grad={avg['loss_gradient']:.5f} | "
            f"mask={avg['loss_mask']:.5f} | local={avg['loss_local_weighted']:.5f} | "
            f"val_psnr={val['overall']:.3f} | rain={val['rain']:.3f} | snow={val['snow']:.3f} | "
            f"lr={lr_now:.3e} | time={elapsed:.1f}s"
        )
        print_delta(val)

    print("Best simple-validation checkpoint:", best_ckpt)
    print("Best simple-validation PSNR:", best_psnr)
    return best_ckpt, history


if VALIDATE_REFERENCE_CKPT_BEFORE_TRAIN:
    if REFERENCE_CKPT_PATH is None:
        print("VALIDATE_REFERENCE_CKPT_BEFORE_TRAIN=True, but REFERENCE_CKPT_PATH is None.")
    else:
        REFERENCE_SIMPLE = validate_checkpoint(
            REFERENCE_CKPT_PATH,
            use_tile=False,
            use_tta=False,
            desc="reference E015 simple validation",
        )
        print_delta(REFERENCE_SIMPLE)
else:
    print("Skipping reference checkpoint validation. Using logged E015 baseline numbers for comparison:")
    print(KNOWN_BASELINE)

BEST_CKPT, TRAIN_HISTORY = train_multiloss_patch128()
"""


FINAL_VALIDATION = r"""
def load_saved_model_for_eval(ckpt_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_deep_promptir().to(device)
    load_promptir_checkpoint(model, ckpt_path, strict=True, map_location=device)
    model.eval()
    return model, device


VAL_SET = HW4RestorationDataset(
    DATA_ROOT,
    SPLIT,
    "val",
    patch_size=PATCH_SIZE,
    augment=False,
    cache_images=CACHE_TRAIN_VAL_IMAGES_IN_RAM,
)
FINAL_MODEL, FINAL_DEVICE = load_saved_model_for_eval(BEST_CKPT)

FINAL_SIMPLE = validate_model(
    FINAL_MODEL,
    VAL_SET,
    FINAL_DEVICE,
    use_tile=False,
    use_tta=False,
    desc="final simple validation",
)
print("Final simple validation:", FINAL_SIMPLE)
print_delta(FINAL_SIMPLE)

if USE_TILED_VALIDATION or USE_X8_TTA:
    FINAL_TILE_TTA = validate_model(
        FINAL_MODEL,
        VAL_SET,
        FINAL_DEVICE,
        use_tile=USE_TILED_VALIDATION,
        use_tta=USE_X8_TTA,
        desc="final tile+x8 validation",
    )
    print("Final tile+x8 validation:", FINAL_TILE_TTA)
    print_delta(FINAL_TILE_TTA)
else:
    FINAL_TILE_TTA = None
"""


INFERENCE = r"""
def tensor_to_uint8_chw(tensor):
    tensor = tensor.detach().float().cpu().squeeze(0).clamp(0.0, 1.0)
    array = np.rint(tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return array


@torch.no_grad()
def run_test_inference(ckpt_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_deep_promptir().to(device)
    load_promptir_checkpoint(model, ckpt_path, strict=True, map_location=device)
    model.eval()

    test_dir = Path(TEST_DIR) if TEST_DIR is not None else Path(DATA_ROOT) / "test" / "degraded"
    dataset = HW4TestDataset(test_dir, cache_images=CACHE_TEST_IMAGES_IN_RAM)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    predictions = {}
    iterator = tqdm(loader, desc="test inference", dynamic_ncols=True) if tqdm is not None else loader

    for meta, degraded in iterator:
        degraded = degraded.to(device)
        filename = meta["filename"][0]
        h = int(meta["height"].item())
        w = int(meta["width"].item())
        amp_enabled = PRECISION == "AMP" and device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            if USE_X8_TTA:
                pred = predict_with_tta(
                    model,
                    degraded,
                    use_tile=USE_TILED_VALIDATION,
                    tile_size=TILE_SIZE,
                    tile_overlap=TILE_OVERLAP,
                )
            else:
                pred = forward_with_padding(
                    model,
                    degraded,
                    use_tile=USE_TILED_VALIDATION,
                    tile_size=TILE_SIZE,
                    tile_overlap=TILE_OVERLAP,
                )
        array = tensor_to_uint8_chw(pred)
        if array.shape != (3, h, w):
            raise RuntimeError(f"{filename}: expected {(3, h, w)}, got {array.shape}")
        predictions[filename] = array

    out_path = Path(SUBMISSION_NPZ_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **predictions)
    print(f"Saved {len(predictions)} predictions to {out_path}")
    return out_path


def validate_pred_npz(path):
    path = Path(path)
    data = np.load(path)
    keys = sorted(data.files, key=lambda name: int(Path(name).stem) if Path(name).stem.isdigit() else name)
    print("num_keys:", len(keys))
    print("first_keys:", keys[:5])
    for key in keys:
        array = data[key]
        if array.ndim != 3 or array.shape[0] != 3:
            raise ValueError(f"{key}: expected CHW with 3 channels, got {array.shape}")
        if array.dtype != np.uint8:
            raise ValueError(f"{key}: expected uint8, got {array.dtype}")
    print("pred.npz format: OK")
    return keys


def make_submission_zip(npz_path):
    zip_path = Path(SUBMISSION_ZIP_PATH)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(npz_path, arcname="pred.npz")
    print("Created:", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        print("Zip contents:", zf.namelist())
    return zip_path


if MAKE_SUBMISSION:
    PRED_NPZ = run_test_inference(BEST_CKPT)
    SUBMISSION_KEYS = validate_pred_npz(PRED_NPZ)
    ZIP_PATH = make_submission_zip(PRED_NPZ)
else:
    print("MAKE_SUBMISSION=False, skipping test inference and zip creation.")
"""


DONE = r"""
print("Done.")
print("Notebook:", RUN_NAME)
print("Best checkpoint:", BEST_CKPT)
print("Metrics CSV:", Path(OUTPUT_DIR) / "runs" / RUN_NAME / "metrics.csv")
print("History JSON:", Path(OUTPUT_DIR) / "runs" / RUN_NAME / "history.json")
print("Baseline reference:", KNOWN_BASELINE)
"""


def build_notebook() -> dict:
    cells = [
        md(TITLE),
        code(IMPORTS),
        code(CONFIG),
        code(DATA),
        code(MODEL_CELL_PREFIX + "\n" + notebook_model_source()),
        code(LOSS_CELL),
        code(LOSS_SANITY_CHECK),
        code(TRAINING_UTILS),
        code(TRAINING_RUN),
        code(FINAL_VALIDATION),
        code(INFERENCE),
        code(DONE),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "isGpuEnabled": True,
                "isInternetEnabled": False,
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    outputs = [
        ROOT / "promptir-hw4-multiloss.ipynb",
        ROOT / "notebooks" / "hw4_deep_promptir_multiloss_scratch128.ipynb",
    ]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        print("Wrote", path)


if __name__ == "__main__":
    main()
