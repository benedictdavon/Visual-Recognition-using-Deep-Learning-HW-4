"""Create the self-contained Kaggle HW4 PromptIR notebook."""

from __future__ import annotations

import json
from pathlib import Path


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


TITLE = r"""
# HW4 Rain/Snow Image Restoration - PromptIR / Deep PromptIR

This Kaggle notebook trains a single PromptIR-based model from scratch on the HW4 rain/snow restoration dataset, optionally runs progressive patch fine-tuning, validates with tiled x8 TTA, and writes a valid `pred.npz` plus `submission.zip`.

Run the cells from top to bottom. All user-facing settings are in the CAPSLOCKED CONFIG cell. The notebook writes internal scripts under `/kaggle/working` for DDP training.
"""


IMPORTS = r"""
import os
import sys
import json
import math
import time
import random
import zipfile
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import torch

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

OUTPUT_DIR = "/kaggle/working/output"
RUN_NAME = "deep_promptir_kaggle"
SEED = 42

VAL_PER_TASK = 160
USE_FIXED_SPLIT = True
SPLIT_JSON_PATH = "/kaggle/working/hw4_split_seed42.json"

MODEL_DIM = 48
NUM_BLOCKS = [4, 6, 8, 10]
NUM_REFINEMENT_BLOCKS = 6
PROMPT_LEN = 8
NUM_HEADS = [1, 2, 4, 8]
FFN_EXPANSION_FACTOR = 2.66
BIAS = False
LAYER_NORM_TYPE = "WithBias"
DUAL_PIXEL_TASK = False
USE_SOFT_PROMPT_ROUTING = False
AUX_CLS_WEIGHT = 0.0

USE_DDP = True
NUM_GPUS = "AUTO"
PRECISION = "AMP"
MATMUL_PRECISION = "medium"

TRAIN_MODE = "MULTI_STAGE"
RESUME_CHECKPOINT = None

STAGES = [
    {
        "NAME": "scratch128",
        "INIT_CKPT": None,
        "PATCH_SIZE": 128,
        "EPOCHS": 60,
        "BATCH_SIZE_PER_GPU": 4,
        "LR": 2e-4,
        "WEIGHT_DECAY": 1e-4,
        "WARMUP_EPOCHS": 8,
        "GRAD_CLIP": 1.0,
    },
    {
        "NAME": "ft192",
        "INIT_CKPT": "AUTO_BEST_PREVIOUS",
        "PATCH_SIZE": 192,
        "EPOCHS": 30,
        "BATCH_SIZE_PER_GPU": 2,
        "LR": 2e-5,
        "WEIGHT_DECAY": 1e-4,
        "WARMUP_EPOCHS": 0,
        "GRAD_CLIP": 0.75,
    },
    {
        "NAME": "ft256",
        "INIT_CKPT": "AUTO_BEST_PREVIOUS",
        "PATCH_SIZE": 256,
        "EPOCHS": 20,
        "BATCH_SIZE_PER_GPU": 1,
        "LR": 8e-6,
        "WEIGHT_DECAY": 5e-5,
        "WARMUP_EPOCHS": 0,
        "GRAD_CLIP": 0.5,
    },
]

RUN_ONLY_STAGE = None
# If RUN_ONLY_STAGE is None, run all stages sequentially.
# If set to "scratch128", "ft192", or "ft256", run only that stage.

NUM_WORKERS = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True
TRAIN_LOG_EVERY_N_STEPS = 50

VALIDATE_EVERY_EPOCH = True
SAVE_TOP_K = 3
SAVE_LAST = True

USE_TILED_VALIDATION = True
TILE_SIZE = 256
TILE_OVERLAP = 32
USE_X8_TTA = True

INFERENCE_CKPT = "AUTO_BEST"
MAKE_SUBMISSION = True
SUBMISSION_NPZ_PATH = "/kaggle/working/pred.npz"
SUBMISSION_ZIP_PATH = "/kaggle/working/submission.zip"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
torch.set_float32_matmul_precision(MATMUL_PRECISION)
print("Configured RUN_NAME:", RUN_NAME)
print("Output dir:", OUTPUT_DIR)
"""


DATA_DISCOVERY = r"""
def _has_hw4_layout(path):
    path = Path(path)
    return (
        (path / "train" / "degraded").is_dir()
        and (path / "train" / "clean").is_dir()
        and (path / "test" / "degraded").is_dir()
    )


def find_hw4_root(preferred, auto=True):
    preferred = Path(preferred)
    if preferred.exists() and _has_hw4_layout(preferred):
        return preferred
    if not auto:
        raise FileNotFoundError(f"DATA_ROOT does not match expected HW4 layout: {preferred}")
    candidates = []
    for root, dirs, files in os.walk("/kaggle/input"):
        root_path = Path(root)
        if _has_hw4_layout(root_path):
            candidates.append(root_path)
    if not candidates:
        raise FileNotFoundError("Could not auto-find a folder with train/degraded, train/clean, test/degraded under /kaggle/input")
    candidates = sorted(candidates, key=lambda p: (len(str(p)), str(p)))
    return candidates[0]


def _parse_task_index(path):
    stem = Path(path).stem
    task, idx = stem.split("-", 1)
    return task, int(idx)


DATA_ROOT = str(find_hw4_root(DATA_ROOT, AUTO_FIND_DATA_ROOT))
print("Selected DATA_ROOT:", DATA_ROOT)

TRAIN_DEGRADED_DIR = Path(DATA_ROOT) / "train" / "degraded"
TRAIN_CLEAN_DIR = Path(DATA_ROOT) / "train" / "clean"
TEST_DEGRADED_DIR = Path(DATA_ROOT) / "test" / "degraded"

for required in [TRAIN_DEGRADED_DIR, TRAIN_CLEAN_DIR, TEST_DEGRADED_DIR]:
    assert required.is_dir(), f"Missing required folder: {required}"

rain_files = sorted(TRAIN_DEGRADED_DIR.glob("rain-*.png"), key=lambda p: _parse_task_index(p)[1])
snow_files = sorted(TRAIN_DEGRADED_DIR.glob("snow-*.png"), key=lambda p: _parse_task_index(p)[1])
test_files = sorted(TEST_DEGRADED_DIR.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)

print("Rain degraded:", len(rain_files))
print("Snow degraded:", len(snow_files))
print("Test degraded:", len(test_files))

assert rain_files, "No rain training images found"
assert snow_files, "No snow training images found"
assert test_files, "No test images found"

for p in rain_files + snow_files:
    task, idx = _parse_task_index(p)
    clean = TRAIN_CLEAN_DIR / f"{task}_clean-{idx}.png"
    assert clean.exists(), f"Missing clean pair for {p.name}: expected {clean.name}"

print("Pairing check passed.")
print("First test keys:", [p.name for p in test_files[:5]])
"""


UTILITY_CELL = r"""
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_config_py(path="/kaggle/working/hw4_config.py"):
    config_names = [
        "DATA_ROOT", "AUTO_FIND_DATA_ROOT", "OUTPUT_DIR", "RUN_NAME", "SEED",
        "VAL_PER_TASK", "USE_FIXED_SPLIT", "SPLIT_JSON_PATH",
        "MODEL_DIM", "NUM_BLOCKS", "NUM_REFINEMENT_BLOCKS", "PROMPT_LEN",
        "NUM_HEADS", "FFN_EXPANSION_FACTOR", "BIAS", "LAYER_NORM_TYPE",
        "DUAL_PIXEL_TASK", "USE_SOFT_PROMPT_ROUTING", "AUX_CLS_WEIGHT",
        "USE_DDP", "NUM_GPUS", "PRECISION", "MATMUL_PRECISION",
        "TRAIN_MODE", "RESUME_CHECKPOINT", "STAGES", "RUN_ONLY_STAGE",
        "NUM_WORKERS", "PIN_MEMORY", "PERSISTENT_WORKERS",
        "TRAIN_LOG_EVERY_N_STEPS",
        "VALIDATE_EVERY_EPOCH", "SAVE_TOP_K", "SAVE_LAST",
        "USE_TILED_VALIDATION", "TILE_SIZE", "TILE_OVERLAP", "USE_X8_TTA",
        "INFERENCE_CKPT", "MAKE_SUBMISSION", "SUBMISSION_NPZ_PATH",
        "SUBMISSION_ZIP_PATH",
    ]
    lines = ["# Auto-generated by the Kaggle notebook. Do not edit by hand.\n"]
    for name in config_names:
        lines.append(f"{name} = {repr(globals()[name])}\n")
    Path(path).write_text("".join(lines), encoding="utf-8")
    print("Wrote", path)


seed_everything(SEED)
write_config_py()
"""


DATASET_DATALOADER = r"""
# The actual Dataset/DataLoader classes are written into /kaggle/working/hw4_lib.py in the DDP script generation cell.
# This lightweight split preview mirrors the training split logic.
def make_split_preview():
    rng = random.Random(SEED)
    split_path = Path(SPLIT_JSON_PATH)
    if USE_FIXED_SPLIT and split_path.exists():
        split = json.loads(split_path.read_text())
        print("Loaded existing split:", split_path)
        return split
    rain_names = [p.name for p in rain_files]
    snow_names = [p.name for p in snow_files]
    rng.shuffle(rain_names)
    rng.shuffle(snow_names)
    split = {
        "seed": SEED,
        "val_per_task": VAL_PER_TASK,
        "val": {"rain": sorted(rain_names[:VAL_PER_TASK]), "snow": sorted(snow_names[:VAL_PER_TASK])},
        "train": {"rain": sorted(rain_names[VAL_PER_TASK:]), "snow": sorted(snow_names[VAL_PER_TASK:])},
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print("Saved split:", split_path)
    return split


SPLIT = make_split_preview()
print("Train rain/snow:", len(SPLIT["train"]["rain"]), len(SPLIT["train"]["snow"]))
print("Val rain/snow:", len(SPLIT["val"]["rain"]), len(SPLIT["val"]["snow"]))
"""


MODEL_CELL = r"""
# Model implementation summary:
# - PromptIR/Restormer-style encoder-decoder
# - Dynamic channels derived from MODEL_DIM
# - Prompt blocks with PROMPT_LEN learnable components
# - Residual output: restored = network_output + input
#
# The complete implementation is written into /kaggle/working/hw4_lib.py in the next cell.
print("Model config:")
print({
    "MODEL_DIM": MODEL_DIM,
    "NUM_BLOCKS": NUM_BLOCKS,
    "NUM_REFINEMENT_BLOCKS": NUM_REFINEMENT_BLOCKS,
    "PROMPT_LEN": PROMPT_LEN,
    "NUM_HEADS": NUM_HEADS,
    "USE_SOFT_PROMPT_ROUTING": USE_SOFT_PROMPT_ROUTING,
})
"""


SCRIPT_GENERATION = r'''
LIB_CODE = r"""
import os
import csv
import json
import math
import time
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_rank0():
    return int(os.environ.get("RANK", "0")) == 0


def log_rank0(*args, **kwargs):
    if is_rank0():
        print(*args, **kwargs, flush=True)


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
    candidates = []
    for root, dirs, files in os.walk("/kaggle/input"):
        root_path = Path(root)
        if has_hw4_layout(root_path):
            candidates.append(root_path)
    if not candidates:
        raise FileNotFoundError("Could not auto-find HW4 dataset under /kaggle/input")
    return sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0]


def parse_task_index(filename):
    stem = Path(filename).stem
    task, idx = stem.split("-", 1)
    return task, int(idx)


def sorted_image_files(path):
    path = Path(path)
    files = list(path.glob("*.png"))
    def key_fn(p):
        if p.stem.isdigit():
            return (0, int(p.stem))
        task, idx = parse_task_index(p.name)
        return (1 if task == "rain" else 2, idx)
    return sorted(files, key=key_fn)


def build_or_load_split(cfg):
    root = find_hw4_root(cfg.DATA_ROOT, cfg.AUTO_FIND_DATA_ROOT)
    split_path = Path(cfg.SPLIT_JSON_PATH)
    if cfg.USE_FIXED_SPLIT and split_path.exists():
        split = json.loads(split_path.read_text())
        return root, split

    degraded_dir = root / "train" / "degraded"
    clean_dir = root / "train" / "clean"
    rain = sorted(degraded_dir.glob("rain-*.png"), key=lambda p: parse_task_index(p.name)[1])
    snow = sorted(degraded_dir.glob("snow-*.png"), key=lambda p: parse_task_index(p.name)[1])
    for p in rain + snow:
        task, idx = parse_task_index(p.name)
        clean = clean_dir / f"{task}_clean-{idx}.png"
        if not clean.exists():
            raise FileNotFoundError(f"Missing clean pair for {p.name}: {clean}")
    rng = random.Random(cfg.SEED)
    rain_names = [p.name for p in rain]
    snow_names = [p.name for p in snow]
    rng.shuffle(rain_names)
    rng.shuffle(snow_names)
    split = {
        "seed": cfg.SEED,
        "val_per_task": cfg.VAL_PER_TASK,
        "val": {
            "rain": sorted(rain_names[: cfg.VAL_PER_TASK]),
            "snow": sorted(snow_names[: cfg.VAL_PER_TASK]),
        },
        "train": {
            "rain": sorted(rain_names[cfg.VAL_PER_TASK :]),
            "snow": sorted(snow_names[cfg.VAL_PER_TASK :]),
        },
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return root, split


def pil_to_tensor(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class HW4RestorationDataset(Dataset):
    def __init__(self, root, split, split_name="train", patch_size=128, augment=True):
        self.root = Path(root)
        self.split_name = split_name
        self.patch_size = patch_size
        self.augment = augment and split_name == "train"
        self.degraded_dir = self.root / "train" / "degraded"
        self.clean_dir = self.root / "train" / "clean"
        self.items = []
        for task in ["rain", "snow"]:
            for name in split[split_name][task]:
                _, idx = parse_task_index(name)
                clean_name = f"{task}_clean-{idx}.png"
                self.items.append({
                    "task": task,
                    "task_id": 0 if task == "rain" else 1,
                    "degraded": name,
                    "clean": clean_name,
                })
        if split_name == "train":
            rng = random.Random(1234)
            rng.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        degraded = pil_to_tensor(self.degraded_dir / item["degraded"])
        clean = pil_to_tensor(self.clean_dir / item["clean"])
        if self.split_name == "train":
            degraded, clean = self.random_crop_pair(degraded, clean, self.patch_size)
            if self.augment:
                if random.random() < 0.5:
                    degraded = torch.flip(degraded, dims=[2])
                    clean = torch.flip(clean, dims=[2])
                if random.random() < 0.5:
                    degraded = torch.flip(degraded, dims=[1])
                    clean = torch.flip(clean, dims=[1])
        meta = {"filename": item["degraded"], "task": item["task"], "task_id": item["task_id"]}
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
        return (
            degraded[:, top : top + patch_size, left : left + patch_size],
            clean[:, top : top + patch_size, left : left + patch_size],
        )


class HW4TestDataset(Dataset):
    def __init__(self, test_dir):
        self.test_dir = Path(test_dir)
        self.files = sorted_image_files(self.test_dir)
        if not self.files:
            raise FileNotFoundError(f"No PNG files found in {self.test_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        tensor = pil_to_tensor(path)
        _, h, w = tensor.shape
        meta = {"filename": path.name, "height": h, "width": w}
        return meta, tensor


def to_3d(x):
    b, c, h, w = x.shape
    return x.reshape(b, c, h * w).transpose(1, 2).contiguous()


def to_4d(x, h, w):
    b, hw, c = x.shape
    if hw != h * w:
        raise RuntimeError(f"LayerNorm reshape mismatch: tokens={hw}, h*w={h * w}")
    return x.transpose(1, 2).reshape(b, c, h, w).contiguous()


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm2d(nn.Module):
    def __init__(self, dim, layer_norm_type="WithBias"):
        super().__init__()
        self.body = BiasFreeLayerNorm(dim) if layer_norm_type == "BiasFree" else WithBiasLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor, bias):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.reshape(b, c, h, w)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, layer_norm_type):
        super().__init__()
        self.norm1 = LayerNorm2d(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c, embed_dim, bias):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat, bias):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, padding=1, bias=bias),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat, bias):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, padding=1, bias=bias),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class PromptGenBlock(nn.Module):
    def __init__(self, in_dim, prompt_dim, prompt_len, prompt_size, bias):
        super().__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(in_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, 3, padding=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        emb = x.mean(dim=(-2, -1))
        weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = weights[:, :, None, None, None] * self.prompt_param.expand(b, -1, -1, -1, -1)
        prompt = prompt.sum(dim=1)
        prompt = F.interpolate(prompt, size=(h, w), mode="bilinear", align_corners=False)
        return self.conv3x3(prompt)


class PromptInteraction(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, layer_norm_type):
        super().__init__()
        self.reduce = nn.Conv2d(dim * 2, dim, 1, bias=bias)
        self.block = TransformerBlock(dim, num_heads, ffn_expansion_factor, bias, layer_norm_type)

    def forward(self, x, prompt):
        return self.block(self.reduce(torch.cat([x, prompt], dim=1)))


class PromptIR(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 6, 8, 10),
        num_refinement_blocks=6,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66,
        bias=False,
        layer_norm_type="WithBias",
        prompt_len=8,
    ):
        super().__init__()
        c1, c2, c3, c4 = dim, dim * 2, dim * 4, dim * 8
        self.patch_embed = OverlapPatchEmbed(inp_channels, c1, bias)
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(c1, heads[0], ffn_expansion_factor, bias, layer_norm_type)
            for _ in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(c1, bias)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(c2, heads[1], ffn_expansion_factor, bias, layer_norm_type)
            for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(c2, bias)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(c3, heads[2], ffn_expansion_factor, bias, layer_norm_type)
            for _ in range(num_blocks[2])
        ])
        self.down3_4 = Downsample(c3, bias)
        self.latent = nn.Sequential(*[
            TransformerBlock(c4, heads[3], ffn_expansion_factor, bias, layer_norm_type)
            for _ in range(num_blocks[3])
        ])

        self.up4_3 = Upsample(c4, bias)
        self.reduce3 = nn.Conv2d(c3 * 2, c3, 1, bias=bias)
        self.prompt3 = PromptGenBlock(c3, c3, prompt_len, 16, bias)
        self.interact3 = PromptInteraction(c3, heads[2], ffn_expansion_factor, bias, layer_norm_type)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(c3, heads[2], ffn_expansion_factor, bias, layer_norm_type)
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(c3, bias)
        self.reduce2 = nn.Conv2d(c2 * 2, c2, 1, bias=bias)
        self.prompt2 = PromptGenBlock(c2, c2, prompt_len, 32, bias)
        self.interact2 = PromptInteraction(c2, heads[1], ffn_expansion_factor, bias, layer_norm_type)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(c2, heads[1], ffn_expansion_factor, bias, layer_norm_type)
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(c2, bias)
        self.reduce1 = nn.Conv2d(c1 * 2, c1, 1, bias=bias)
        self.prompt1 = PromptGenBlock(c1, c1, prompt_len, 64, bias)
        self.interact1 = PromptInteraction(c1, heads[0], ffn_expansion_factor, bias, layer_norm_type)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(c1, heads[0], ffn_expansion_factor, bias, layer_norm_type)
            for _ in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock(c1, heads[0], ffn_expansion_factor, bias, layer_norm_type)
            for _ in range(num_refinement_blocks)
        ])
        self.output = nn.Conv2d(c1, out_channels, 3, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc1 = self.patch_embed(inp_img)
        out_enc1 = self.encoder_level1(inp_enc1)
        inp_enc2 = self.down1_2(out_enc1)
        out_enc2 = self.encoder_level2(inp_enc2)
        inp_enc3 = self.down2_3(out_enc2)
        out_enc3 = self.encoder_level3(inp_enc3)
        inp_enc4 = self.down3_4(out_enc3)
        latent = self.latent(inp_enc4)

        x = self.up4_3(latent)
        x = self.reduce3(torch.cat([x, out_enc3], dim=1))
        x = self.interact3(x, self.prompt3(x))
        x = self.decoder_level3(x)

        x = self.up3_2(x)
        x = self.reduce2(torch.cat([x, out_enc2], dim=1))
        x = self.interact2(x, self.prompt2(x))
        x = self.decoder_level2(x)

        x = self.up2_1(x)
        x = self.reduce1(torch.cat([x, out_enc1], dim=1))
        x = self.interact1(x, self.prompt1(x))
        x = self.decoder_level1(x)
        x = self.refinement(x)
        return self.output(x) + inp_img


def build_model(cfg):
    if cfg.USE_SOFT_PROMPT_ROUTING:
        raise NotImplementedError("Soft prompt routing is intentionally disabled for this Kaggle notebook default.")
    return PromptIR(
        dim=cfg.MODEL_DIM,
        num_blocks=tuple(cfg.NUM_BLOCKS),
        num_refinement_blocks=cfg.NUM_REFINEMENT_BLOCKS,
        heads=tuple(cfg.NUM_HEADS),
        ffn_expansion_factor=cfg.FFN_EXPANSION_FACTOR,
        bias=cfg.BIAS,
        layer_norm_type=cfg.LAYER_NORM_TYPE,
        prompt_len=cfg.PROMPT_LEN,
    )


def model_summary(model, cfg, device="cuda"):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model config:", {
        "dim": cfg.MODEL_DIM,
        "num_blocks": cfg.NUM_BLOCKS,
        "num_refinement_blocks": cfg.NUM_REFINEMENT_BLOCKS,
        "prompt_len": cfg.PROMPT_LEN,
    })
    print(f"Total params: {total / 1e6:.3f}M")
    print(f"Trainable params: {trainable / 1e6:.3f}M")
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 128, 128, device=device)
        y = model.to(device)(x)
    print("Dummy input/output:", tuple(x.shape), tuple(y.shape))


def strip_prefix_if_present(state, prefix):
    if all(k.startswith(prefix) for k in state.keys()):
        return {k[len(prefix):]: v for k, v in state.items()}
    return state


def load_checkpoint(model, ckpt_path, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")
    state = strip_prefix_if_present(state, "module.")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. Missing={missing[:5]}, unexpected={unexpected[:5]}")
    return model


def save_checkpoint(path, model, stage, epoch, val_psnr, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({
        "model": state,
        "stage": stage,
        "epoch": epoch,
        "val_psnr": float(val_psnr),
        "config": {
            "MODEL_DIM": cfg.MODEL_DIM,
            "NUM_BLOCKS": cfg.NUM_BLOCKS,
            "NUM_REFINEMENT_BLOCKS": cfg.NUM_REFINEMENT_BLOCKS,
            "PROMPT_LEN": cfg.PROMPT_LEN,
        },
    }, path)


def pad_to_multiple(x, multiple=8):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    mode = "reflect" if h > pad_h and w > pad_w else "replicate"
    x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, (h, w)


def unpad(x, size):
    h, w = size
    return x[..., :h, :w]


def tile_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def model_forward(model, x):
    y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    return y


def tiled_forward(model, x, tile_size=256, tile_overlap=32):
    b, c, h, w = x.shape
    stride = max(1, tile_size - tile_overlap)
    ys = tile_starts(h, tile_size, stride)
    xs = tile_starts(w, tile_size, stride)
    output = torch.zeros_like(x)
    weight = torch.zeros_like(x)
    for y0 in ys:
        for x0 in xs:
            tile = x[..., y0 : y0 + tile_size, x0 : x0 + tile_size]
            pred = model_forward(model, tile)
            output[..., y0 : y0 + tile.shape[-2], x0 : x0 + tile.shape[-1]] += pred
            weight[..., y0 : y0 + tile.shape[-2], x0 : x0 + tile.shape[-1]] += 1
    return output / weight.clamp_min(1)


def forward_with_padding(model, x, multiple=8, use_tile=False, tile_size=256, tile_overlap=32):
    x_pad, original_size = pad_to_multiple(x, multiple)
    if use_tile:
        y = tiled_forward(model, x_pad, tile_size, tile_overlap)
    else:
        y = model_forward(model, x_pad)
    return unpad(y, original_size)


def tta_transforms(x):
    return [
        (x, lambda y: y),
        (torch.flip(x, [-1]), lambda y: torch.flip(y, [-1])),
        (torch.flip(x, [-2]), lambda y: torch.flip(y, [-2])),
        (torch.flip(x, [-2, -1]), lambda y: torch.flip(y, [-2, -1])),
        (x.transpose(-2, -1), lambda y: y.transpose(-2, -1)),
        (torch.flip(x.transpose(-2, -1), [-1]), lambda y: torch.flip(y, [-1]).transpose(-2, -1)),
        (torch.flip(x.transpose(-2, -1), [-2]), lambda y: torch.flip(y, [-2]).transpose(-2, -1)),
        (torch.flip(x.transpose(-2, -1), [-2, -1]), lambda y: torch.flip(y, [-2, -1]).transpose(-2, -1)),
    ]


def predict_with_tta(model, x, use_tile=True, tile_size=256, tile_overlap=32):
    preds = []
    for aug, inv in tta_transforms(x):
        pred = forward_with_padding(model, aug, use_tile=use_tile, tile_size=tile_size, tile_overlap=tile_overlap)
        preds.append(inv(pred))
    return torch.stack(preds, dim=0).mean(dim=0)


def psnr_tensor(pred, target):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = (pred - target).pow(2).flatten(1).mean(dim=1)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    return psnr


@torch.no_grad()
def validate_model(model, dataset, device, use_tile=False, tile_size=256, tile_overlap=32, use_tta=False, amp=True):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    all_scores = []
    by_task = {"rain": [], "snow": []}
    pbar = tqdm(
        loader,
        total=len(loader),
        disable=not is_rank0(),
        dynamic_ncols=True,
        leave=False,
        desc="validation",
    ) if tqdm is not None else loader
    for meta, degraded, clean in pbar:
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            if use_tta:
                pred = predict_with_tta(model, degraded, use_tile=use_tile, tile_size=tile_size, tile_overlap=tile_overlap)
            else:
                pred = forward_with_padding(model, degraded, use_tile=use_tile, tile_size=tile_size, tile_overlap=tile_overlap)
        score = psnr_tensor(pred.float(), clean.float()).item()
        task = meta["task"][0]
        all_scores.append(score)
        by_task[task].append(score)
    return {
        "overall": float(np.mean(all_scores)),
        "rain": float(np.mean(by_task["rain"])),
        "snow": float(np.mean(by_task["snow"])),
    }


def make_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def stage_dir(cfg, stage_name):
    return Path(cfg.OUTPUT_DIR) / "runs" / cfg.RUN_NAME / stage_name


def find_best_checkpoint(path):
    path = Path(path)
    candidates = sorted(path.glob("epoch*-psnr*.ckpt"))
    if not candidates:
        candidates = sorted(path.glob("best*.ckpt"))
    if not candidates:
        return None
    def score(p):
        name = p.stem
        if "psnr" in name:
            try:
                return float(name.split("psnr")[-1])
            except ValueError:
                return -1.0
        return -1.0
    return max(candidates, key=score)


def resolve_init_checkpoint(cfg, stage, previous_best):
    if cfg.RESUME_CHECKPOINT:
        return cfg.RESUME_CHECKPOINT
    init = stage.get("INIT_CKPT")
    if init is None:
        return None
    if init == "AUTO_BEST_PREVIOUS":
        if previous_best:
            return str(previous_best)
        stage_names = [s["NAME"] for s in cfg.STAGES]
        idx = stage_names.index(stage["NAME"])
        if idx <= 0:
            return None
        return str(find_best_checkpoint(stage_dir(cfg, stage_names[idx - 1])))
    return init


def run_training(cfg):
    torch.set_float32_matmul_precision(cfg.MATMUL_PRECISION)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if distributed:
        dist.init_process_group(backend="nccl")
    seed_everything(cfg.SEED + local_rank)

    root, split = build_or_load_split(cfg)
    if is_rank0():
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg.SPLIT_JSON_PATH, Path(cfg.OUTPUT_DIR) / "split.json")
        Path(cfg.OUTPUT_DIR, "config.json").write_text(json.dumps({
            k: getattr(cfg, k) for k in dir(cfg) if k.isupper()
        }, indent=2, default=str), encoding="utf-8")
        log_rank0("DATA_ROOT:", root)
        log_rank0("WORLD_SIZE:", world_size)

    stages = cfg.STAGES
    if cfg.RUN_ONLY_STAGE is not None:
        stages = [s for s in stages if s["NAME"] == cfg.RUN_ONLY_STAGE]
        if not stages:
            raise ValueError(f"RUN_ONLY_STAGE not found: {cfg.RUN_ONLY_STAGE}")

    previous_best = None
    for stage in stages:
        stage_name = stage["NAME"]
        out_dir = stage_dir(cfg, stage_name)
        if is_rank0():
            out_dir.mkdir(parents=True, exist_ok=True)
            log_rank0(f"\n=== Stage {stage_name} ===")

        train_set = HW4RestorationDataset(root, split, "train", patch_size=stage["PATCH_SIZE"], augment=True)
        val_set = HW4RestorationDataset(root, split, "val", patch_size=stage["PATCH_SIZE"], augment=False)
        sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
        loader = DataLoader(
            train_set,
            batch_size=stage["BATCH_SIZE_PER_GPU"],
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            persistent_workers=cfg.PERSISTENT_WORKERS and cfg.NUM_WORKERS > 0,
            drop_last=True,
        )

        model = build_model(cfg).to(device)
        init_ckpt = resolve_init_checkpoint(cfg, stage, previous_best)
        if init_ckpt:
            log_rank0("Loading init checkpoint:", init_ckpt)
            load_checkpoint(model, init_ckpt, map_location=device)
        if is_rank0():
            model_summary(model, cfg, device=device)
            log_rank0("Batch per GPU:", stage["BATCH_SIZE_PER_GPU"])
            log_rank0("Effective batch:", stage["BATCH_SIZE_PER_GPU"] * world_size)

        if distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        optimizer = torch.optim.AdamW(model.parameters(), lr=stage["LR"], weight_decay=stage["WEIGHT_DECAY"])
        total_steps = max(1, len(loader) * stage["EPOCHS"])
        warmup_steps = max(0, len(loader) * stage["WARMUP_EPOCHS"])
        scheduler = make_scheduler(optimizer, warmup_steps, total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.PRECISION == "AMP" and device.type == "cuda")
        loss_fn = nn.L1Loss()
        metrics_path = out_dir / "metrics.csv"
        if is_rank0():
            with metrics_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["stage", "epoch", "train_l1", "val_psnr", "val_rain", "val_snow", "lr", "elapsed_sec"])

        best = -1.0
        top = []
        for epoch in range(stage["EPOCHS"]):
            start = time.time()
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            running = 0.0
            steps = 0
            pbar = tqdm(
                loader,
                total=len(loader),
                disable=not is_rank0(),
                dynamic_ncols=True,
                leave=False,
                desc=f"{stage_name} epoch {epoch}",
            ) if tqdm is not None else loader
            for meta, degraded, clean in pbar:
                degraded = degraded.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=cfg.PRECISION == "AMP" and device.type == "cuda"):
                    pred = model(degraded)
                    loss = loss_fn(pred, clean)
                if not torch.isfinite(loss):
                    log_rank0("Skipping non-finite loss")
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), stage["GRAD_CLIP"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running += loss.item()
                steps += 1
                if is_rank0() and tqdm is not None:
                    pbar.set_postfix({
                        "loss": f"{running / max(1, steps):.5f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    })
                if (
                    is_rank0()
                    and cfg.TRAIN_LOG_EVERY_N_STEPS
                    and (steps % cfg.TRAIN_LOG_EVERY_N_STEPS == 0)
                ):
                    log_rank0(
                        f"stage={stage_name} epoch={epoch} "
                        f"step={steps}/{len(loader)} "
                        f"train_l1={running / max(1, steps):.5f} "
                        f"lr={optimizer.param_groups[0]['lr']:.3e}"
                    )

            if distributed:
                dist.barrier()
            val = {"overall": float("nan"), "rain": float("nan"), "snow": float("nan")}
            unwrapped = model.module if hasattr(model, "module") else model
            if is_rank0() and cfg.VALIDATE_EVERY_EPOCH:
                val = validate_model(unwrapped, val_set, device, use_tile=False, amp=cfg.PRECISION == "AMP")
                avg_loss = running / max(1, steps)
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                ckpt_name = f"epoch{epoch:03d}-psnr{val['overall']:.3f}.ckpt"
                ckpt_path = out_dir / ckpt_name
                save_checkpoint(ckpt_path, unwrapped, stage_name, epoch, val["overall"], cfg)
                if cfg.SAVE_LAST:
                    save_checkpoint(out_dir / "last.ckpt", unwrapped, stage_name, epoch, val["overall"], cfg)
                top.append((val["overall"], ckpt_path))
                top = sorted(top, key=lambda x: x[0], reverse=True)
                for _, p in top[cfg.SAVE_TOP_K:]:
                    if p.exists():
                        p.unlink()
                top = top[: cfg.SAVE_TOP_K]
                if val["overall"] > best:
                    best = val["overall"]
                    save_checkpoint(out_dir / "best.ckpt", unwrapped, stage_name, epoch, val["overall"], cfg)
                    previous_best = out_dir / "best.ckpt"
                with metrics_path.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([stage_name, epoch, avg_loss, val["overall"], val["rain"], val["snow"], lr, elapsed])
                log_rank0(
                    f"Validation epoch {epoch}: val_psnr={val['overall']:.3f}, "
                    f"rain={val['rain']:.3f}, snow={val['snow']:.3f}, "
                    f"train_l1={avg_loss:.5f}, lr={lr:.3e}, time={elapsed:.1f}s"
                )
            if distributed:
                dist.barrier()
        if is_rank0() and previous_best is None:
            previous_best = find_best_checkpoint(out_dir)

    if distributed:
        dist.destroy_process_group()


def resolve_inference_checkpoint(cfg):
    if cfg.INFERENCE_CKPT and cfg.INFERENCE_CKPT != "AUTO_BEST":
        return Path(cfg.INFERENCE_CKPT)
    if cfg.RUN_ONLY_STAGE is not None:
        candidates = [cfg.RUN_ONLY_STAGE]
    else:
        candidates = [s["NAME"] for s in cfg.STAGES][::-1]
    for name in candidates:
        ckpt = find_best_checkpoint(stage_dir(cfg, name))
        if ckpt is not None:
            return ckpt
        best = stage_dir(cfg, name) / "best.ckpt"
        if best.exists():
            return best
    raise FileNotFoundError("Could not resolve INFERENCE_CKPT=AUTO_BEST")


def tensor_to_uint8_chw(x):
    x = x.detach().float().cpu().squeeze(0).clamp(0, 1)
    arr = (x.numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    return arr


@torch.no_grad()
def run_final_validation(cfg, ckpt_path=None):
    root, split = build_or_load_split(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    ckpt = Path(ckpt_path) if ckpt_path else resolve_inference_checkpoint(cfg)
    load_checkpoint(model, ckpt, map_location=device)
    print("Validating checkpoint:", ckpt)
    val_set = HW4RestorationDataset(root, split, "val", patch_size=128, augment=False)
    model_summary(model, cfg, device=device)
    result = validate_model(
        model,
        val_set,
        device,
        use_tile=cfg.USE_TILED_VALIDATION,
        tile_size=cfg.TILE_SIZE,
        tile_overlap=cfg.TILE_OVERLAP,
        use_tta=cfg.USE_X8_TTA,
        amp=cfg.PRECISION == "AMP",
    )
    print("Final validation:", result)
    return result, ckpt


@torch.no_grad()
def run_test_inference(cfg, ckpt_path=None):
    root = find_hw4_root(cfg.DATA_ROOT, cfg.AUTO_FIND_DATA_ROOT)
    test_dir = root / "test" / "degraded"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    ckpt = Path(ckpt_path) if ckpt_path else resolve_inference_checkpoint(cfg)
    load_checkpoint(model, ckpt, map_location=device)
    model.eval()
    dataset = HW4TestDataset(test_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    predictions = {}
    print("Inference checkpoint:", ckpt)
    for meta, degraded in loader:
        degraded = degraded.to(device)
        filename = meta["filename"][0]
        h = int(meta["height"].item())
        w = int(meta["width"].item())
        with torch.amp.autocast("cuda", enabled=cfg.PRECISION == "AMP" and device.type == "cuda"):
            pred = predict_with_tta(
                model,
                degraded,
                use_tile=cfg.USE_TILED_VALIDATION,
                tile_size=cfg.TILE_SIZE,
                tile_overlap=cfg.TILE_OVERLAP,
            ) if cfg.USE_X8_TTA else forward_with_padding(
                model,
                degraded,
                use_tile=cfg.USE_TILED_VALIDATION,
                tile_size=cfg.TILE_SIZE,
                tile_overlap=cfg.TILE_OVERLAP,
            )
        arr = tensor_to_uint8_chw(pred)
        if arr.shape != (3, h, w):
            raise RuntimeError(f"{filename}: expected {(3, h, w)}, got {arr.shape}")
        predictions[filename] = arr
    out_path = Path(cfg.SUBMISSION_NPZ_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **predictions)
    print(f"Saved {len(predictions)} predictions to {out_path}")
    return out_path


def validate_pred_npz(cfg):
    root = find_hw4_root(cfg.DATA_ROOT, cfg.AUTO_FIND_DATA_ROOT)
    test_dir = root / "test" / "degraded"
    expected = {p.name: Image.open(p).convert("RGB").size[::-1] for p in sorted_image_files(test_dir)}
    data = np.load(cfg.SUBMISSION_NPZ_PATH)
    keys = sorted(data.files, key=lambda x: int(Path(x).stem) if Path(x).stem.isdigit() else x)
    print("num_keys:", len(keys))
    print("first_keys:", keys[:5])
    assert set(keys) == set(expected.keys()), "pred.npz keys do not match test filenames"
    for key in keys:
        arr = data[key]
        h, w = expected[key]
        assert arr.shape == (3, h, w), f"{key}: expected {(3, h, w)}, got {arr.shape}"
        assert arr.dtype == np.uint8, f"{key}: expected uint8, got {arr.dtype}"
        assert arr.min() >= 0 and arr.max() <= 255, f"{key}: values out of uint8 range"
    print("pred.npz format: OK")
    return keys


def create_submission_zip(cfg):
    zip_path = Path(cfg.SUBMISSION_ZIP_PATH)
    npz_path = Path(cfg.SUBMISSION_NPZ_PATH)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if zip_path.exists():
        zip_path.unlink()
    import zipfile
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(npz_path, arcname="pred.npz")
    print("Created:", zip_path)
    return zip_path
"""

TRAIN_DDP_CODE = r"""
import sys
sys.path.insert(0, "/kaggle/working")
import hw4_config as C
from hw4_lib import run_training

if __name__ == "__main__":
    run_training(C)
"""

INFER_CODE = r"""
import sys
sys.path.insert(0, "/kaggle/working")
import hw4_config as C
from hw4_lib import run_final_validation, run_test_inference, validate_pred_npz, create_submission_zip

if __name__ == "__main__":
    result, ckpt = run_final_validation(C)
    if C.MAKE_SUBMISSION:
        run_test_inference(C, ckpt)
        validate_pred_npz(C)
        create_submission_zip(C)
"""

Path("/kaggle/working/hw4_lib.py").write_text(LIB_CODE, encoding="utf-8")
Path("/kaggle/working/train_ddp.py").write_text(TRAIN_DDP_CODE, encoding="utf-8")
Path("/kaggle/working/infer_hw4.py").write_text(INFER_CODE, encoding="utf-8")
print("Wrote /kaggle/working/hw4_lib.py")
print("Wrote /kaggle/working/train_ddp.py")
print("Wrote /kaggle/working/infer_hw4.py")

sys.path.insert(0, "/kaggle/working")
import importlib
import hw4_config as C
import hw4_lib
importlib.reload(hw4_lib)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = hw4_lib.build_model(C)
hw4_lib.model_summary(model, C, device=device)
'''


TRAIN_LAUNCH = r"""
write_config_py()

gpu_count = torch.cuda.device_count()
if NUM_GPUS == "AUTO":
    nproc = 2 if (USE_DDP and gpu_count >= 2) else min(1, gpu_count)
else:
    nproc = int(NUM_GPUS)
    nproc = min(nproc, gpu_count)

if nproc >= 2:
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        "/kaggle/working/train_ddp.py",
    ]
else:
    cmd = [sys.executable, "/kaggle/working/train_ddp.py"]

print("Launching training:")
print(" ".join(cmd))
result = subprocess.run(cmd, check=False)
if result.returncode != 0:
    raise RuntimeError(f"Training failed with return code {result.returncode}")
"""


VALIDATION_CELL = r"""
write_config_py()
sys.path.insert(0, "/kaggle/working")
import importlib
import hw4_config as C
import hw4_lib
importlib.reload(hw4_lib)

FINAL_VAL_RESULT, FINAL_CKPT = hw4_lib.run_final_validation(C)
print("Compare against current best local tile+x8 validation: 29.0522")
print("Selected checkpoint:", FINAL_CKPT)
"""


TEST_INFERENCE_CELL = r"""
write_config_py()
if MAKE_SUBMISSION:
    sys.path.insert(0, "/kaggle/working")
    import importlib
    import hw4_config as C
    import hw4_lib
    importlib.reload(hw4_lib)
    PRED_NPZ_PATH = hw4_lib.run_test_inference(C, FINAL_CKPT if "FINAL_CKPT" in globals() else None)
else:
    print("MAKE_SUBMISSION=False, skipping test inference.")
"""


VALIDATOR_CELL = r"""
if MAKE_SUBMISSION:
    sys.path.insert(0, "/kaggle/working")
    import hw4_config as C
    import hw4_lib
    SUBMISSION_KEYS = hw4_lib.validate_pred_npz(C)
else:
    print("MAKE_SUBMISSION=False, skipping pred.npz validation.")
"""


ZIP_CELL = r"""
if MAKE_SUBMISSION:
    sys.path.insert(0, "/kaggle/working")
    import hw4_config as C
    import hw4_lib
    ZIP_PATH = hw4_lib.create_submission_zip(C)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        print("Zip contents:", zf.namelist())
else:
    print("MAKE_SUBMISSION=False, skipping zip creation.")
"""


SUMMARY_CELL = r"""
print("Done.")
print("If auto-detection fails, set DATA_ROOT to the folder containing train/degraded, train/clean, and test/degraded.")
print("Run only ft192: set RUN_ONLY_STAGE = 'ft192' and ensure ft192 INIT_CKPT resolves or set RESUME_CHECKPOINT.")
print("Run only ft256: set RUN_ONLY_STAGE = 'ft256' and ensure ft256 INIT_CKPT resolves or set RESUME_CHECKPOINT.")
print("Resume from a checkpoint: set RESUME_CHECKPOINT = '/kaggle/working/output/.../best.ckpt'.")
print("pred.npz:", SUBMISSION_NPZ_PATH)
print("submission.zip:", SUBMISSION_ZIP_PATH)
print("Compare final validation against local tile+x8 best: 29.0522")
print("Compare public LB against current best public LB: 29.8")
"""


def main() -> None:
    out = Path("notebooks/hw4_promptir_kaggle.ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": [
            md(TITLE),
            code(IMPORTS),
            code(CONFIG),
            code(DATA_DISCOVERY),
            code(UTILITY_CELL),
            code(DATASET_DATALOADER),
            code(MODEL_CELL),
            code(SCRIPT_GENERATION),
            code(TRAIN_LAUNCH),
            code(VALIDATION_CELL),
            code(TEST_INFERENCE_CELL),
            code(VALIDATOR_CELL),
            code(ZIP_CELL),
            code(SUMMARY_CELL),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
