import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
import torch

try:
    from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
except Exception:  # pragma: no cover - fallback for minimal/CPU environments
    class ToTensor:
        def __call__(self, image):
            if isinstance(image, Image.Image):
                image = np.array(image)
            if torch.is_tensor(image):
                return image.float()
            if image.ndim == 2:
                image = image[:, :, None]
            array = np.ascontiguousarray(image.transpose(2, 0, 1))
            return torch.from_numpy(array).float().div(255.0)

    class ToPILImage:
        def __call__(self, image):
            if isinstance(image, Image.Image):
                return image
            if torch.is_tensor(image):
                image = image.detach().cpu().numpy()
                if image.ndim == 3 and image.shape[0] in {1, 3}:
                    image = image.transpose(1, 2, 0)
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(image)

    class RandomCrop:
        def __init__(self, size):
            self.size = (size, size) if isinstance(size, int) else tuple(size)

        def __call__(self, image):
            if isinstance(image, Image.Image):
                width, height = image.size
                crop_h, crop_w = self.size
                top = random.randint(0, height - crop_h)
                left = random.randint(0, width - crop_w)
                return image.crop((left, top, left + crop_w, top + crop_h))
            height, width = image.shape[:2]
            crop_h, crop_w = self.size
            top = random.randint(0, height - crop_h)
            left = random.randint(0, width - crop_w)
            return image[top:top + crop_h, left:left + crop_w]

    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, value):
            for transform in self.transforms:
                value = transform(value)
            return value

from utils.image_utils import random_augmentation, crop_img
try:
    from utils.degradation_utils import Degradation
except Exception:
    Degradation = None

    
class PromptTrainDataset(Dataset):
    def __init__(self, args):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        if Degradation is None:
            raise ImportError("Original PromptTrainDataset requires torchvision-compatible degradation_utils. Use HW4RestorationDataset for HW4 training.")
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur' : 5}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()

        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        ref_file = self.args.data_file_dir + "noisy/denoise_airnet.txt"
        temp_ids = []
        temp_ids+= [id_.strip() for id_ in open(ref_file)]
        clean_ids = []
        name_list = os.listdir(self.args.denoise_dir)
        clean_ids += [self.args.denoise_dir + id_ for id_ in name_list if id_.strip() in temp_ids]

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x,"de_type":0} for x in clean_ids]
            self.s15_ids = self.s15_ids * 3
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x,"de_type":1} for x in clean_ids]
            self.s25_ids = self.s25_ids * 3
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x,"de_type":2} for x in clean_ids]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = self.args.data_file_dir + "hazy/hazy_outside.txt"
        temp_ids+= [self.args.dehaze_dir + id_.strip() for id_ in open(hazy)]
        self.hazy_ids = [{"clean_id" : x,"de_type":4} for x in temp_ids]

        self.hazy_counter = 0
        
        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_rs_ids(self):
        temp_ids = []
        rs = self.args.data_file_dir + "rainy/rainTrain.txt"
        temp_ids+= [self.args.derain_dir + id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id":x,"de_type":3} for x in temp_ids]
        self.rs_ids = self.rs_ids * 120

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))
    

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_name = rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]
        return gt_name

    def _get_nonhazy_name(self, hazy_name):
        dir_name = hazy_name.split("synthetic")[0] + 'original/'
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = '.' + hazy_name.split('.')[-1]
        nonhazy_name = dir_name + name + suffix
        return nonhazy_name

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
            self.sample_ids += self.s25_ids
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids+= self.rs_ids
        
        if "dehaze" in self.de_type:
            self.sample_ids+= self.hazy_ids
        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        if de_id < 3:
            if de_id == 0:
                clean_id = sample["clean_id"]
            elif de_id == 1:
                clean_id = sample["clean_id"]
            elif de_id == 2:
                clean_id = sample["clean_id"]

            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch= np.array(clean_patch)

            clean_name = clean_id.split("/")[-1].split('.')[0]

            clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.D.single_degrade(clean_patch, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_gt_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)


        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15

        self._init_clean_ids()

        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_path)
        self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]

        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]

        noisy_img, _ = self._add_gaussian_noise(clean_img)
        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)

        return [clean_name], noisy_img, clean_img
    def tile_degrad(input_,tile=128,tile_overlap =0):
        sigma_dict = {0:0,1:15,2:25,3:50}
        b, c, h, w = input_.shape
        tile = min(tile, h, w)
        assert tile % 8 == 0, "tile size should be multiple of 8"

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h, w).type_as(input_)
        W = torch.zeros_like(E)
        s = 0
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = in_patch
                # out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(in_patch)

                E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
        # restored = E.div_(W)

        restored = torch.clamp(restored, 0, 1)
        return restored
    def __len__(self):
        return self.num_clean


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain",addnoise = False,sigma = None):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma

        self.set_dataset(task)
    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.derain_path + 'input/')
            # print(name_list)
            print(self.args.derain_path)
            self.ids += [self.args.derain_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(self.args.dehaze_path + 'input/')
            self.ids += [self.args.dehaze_path + 'input/' + id_ for id_ in name_list]

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 1:
            dir_name = degraded_name.split("input")[0] + 'target/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img,_ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        if os.path.isdir(root):
            name_list = []
            for image_file in os.listdir(root):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            if len(name_list) == 0:
                raise Exception('The input directory does not contain any image files')
            self.degraded_ids += [root + id_ for id_ in name_list]
        else:
            if any([root.endswith(ext) for ext in extensions]):
                name_list = [root]
            else:
                raise Exception('Please pass an Image file')
            self.degraded_ids = name_list
        print("Total Images : {}".format(name_list))

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = np.array(Image.open(self.degraded_ids[idx]).convert('RGB'))
        name = self.degraded_ids[idx].split('/')[-1][:-4]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img
    




# -------------------------------------------------------------------------
# HW4 rain/snow paired dataset utilities
#
# Expected layout:
#   <hw4_data_root>/train/degraded/*.png
#   <hw4_data_root>/train/clean/*.png
#   <test_dir>/*.png for submission inference
#
# Restoration samples return metadata plus BCHW-ready tensors in [0, 1].
# Submission export keeps the original test filename and image size.
# -------------------------------------------------------------------------

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


HW4_TASK_TO_ID = {"rain": 0, "snow": 1}
HW4_ID_TO_TASK = {0: "rain", 1: "snow"}


def _natural_key(path_like):
    """Sort filenames such as rain-2.png before rain-10.png."""
    text = str(path_like)
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", text)]


def _read_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def _hw4_clean_name(degraded_name):
    stem = Path(degraded_name).stem
    suffix = Path(degraded_name).suffix
    if stem.startswith("rain-"):
        return f"rain_clean-{stem.split('-')[-1]}{suffix}", "rain"
    if stem.startswith("snow-"):
        return f"snow_clean-{stem.split('-')[-1]}{suffix}", "snow"
    raise ValueError(
        f"Unsupported HW4 degraded filename: {degraded_name}. "
        "Expected rain-<id>.png or snow-<id>.png."
    )


def _collect_hw4_pairs(data_root):
    data_root = Path(data_root)
    degraded_dir = data_root / "train" / "degraded"
    clean_dir = data_root / "train" / "clean"
    if not degraded_dir.is_dir():
        raise FileNotFoundError(f"Missing HW4 degraded directory: {degraded_dir}")
    if not clean_dir.is_dir():
        raise FileNotFoundError(f"Missing HW4 clean directory: {clean_dir}")

    pairs = []
    for degraded_path in sorted(degraded_dir.glob("*.png"), key=_natural_key):
        clean_name, task = _hw4_clean_name(degraded_path.name)
        clean_path = clean_dir / clean_name
        if not clean_path.is_file():
            raise FileNotFoundError(
                f"Missing clean target for {degraded_path.name}: {clean_path}"
            )
        pairs.append({
            "filename": degraded_path.name,
            "clean_filename": clean_name,
            "degraded": str(Path("train") / "degraded" / degraded_path.name),
            "clean": str(Path("train") / "clean" / clean_name),
            "task": task,
            "task_id": HW4_TASK_TO_ID[task],
        })

    if not pairs:
        raise RuntimeError(f"No HW4 training images found in {degraded_dir}")
    return pairs


def make_hw4_split(data_root, split_file, val_per_task=160, seed=42):
    """Create or load a fixed stratified HW4 split file."""
    split_path = Path(split_file)
    if split_path.is_file():
        with split_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    pairs = _collect_hw4_pairs(data_root)
    rng = random.Random(seed)
    by_task = {task: [] for task in HW4_TASK_TO_ID}
    for sample in pairs:
        by_task[sample["task"]].append(sample)

    train_samples = []
    val_samples = []
    for task, samples in by_task.items():
        samples = sorted(samples, key=lambda item: _natural_key(item["filename"]))
        rng.shuffle(samples)
        if len(samples) <= val_per_task:
            raise ValueError(
                f"Task {task} has only {len(samples)} samples, but "
                f"val_per_task={val_per_task}."
            )
        val_samples.extend(samples[:val_per_task])
        train_samples.extend(samples[val_per_task:])

    train_samples = sorted(train_samples, key=lambda item: _natural_key(item["filename"]))
    val_samples = sorted(val_samples, key=lambda item: _natural_key(item["filename"]))
    split = {
        "seed": seed,
        "val_per_task": val_per_task,
        "task_to_id": HW4_TASK_TO_ID,
        "train": train_samples,
        "val": val_samples,
    }

    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    return split


def _pad_if_smaller(image, min_size):
    h, w = image.shape[:2]
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)
    if pad_h == 0 and pad_w == 0:
        return image

    # Reflection padding is good for restoration boundaries, but NumPy reflect
    # can fail when the requested padding is too large for a very small image.
    # Official HW4 images should be larger than the patch size, but this fallback
    # keeps smoke tests and unusual inputs safe.
    mode = "reflect"
    if h <= 1 or w <= 1 or pad_h >= h or pad_w >= w:
        mode = "edge"
    return np.pad(
        image,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode=mode,
    )


def _crop_pair(degraded, clean, patch_size):
    degraded = _pad_if_smaller(degraded, patch_size)
    clean = _pad_if_smaller(clean, patch_size)
    h, w = degraded.shape[:2]
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    degraded_patch = degraded[top:top + patch_size, left:left + patch_size]
    clean_patch = clean[top:top + patch_size, left:left + patch_size]
    return degraded_patch, clean_patch


class HW4RestorationDataset(Dataset):
    """Native paired rain/snow dataset for NYCU VRDL HW4.

    Folder layout expected under ``data_root``::

        train/degraded/rain-1.png ... rain-1600.png
        train/degraded/snow-1.png ... snow-1600.png
        train/clean/rain_clean-1.png ... rain_clean-1600.png
        train/clean/snow_clean-1.png ... snow_clean-1600.png

    Returns:
        metadata: dict with filename, clean_filename, task, task_id, height, width
        degraded_tensor: float tensor in [0, 1], shape (3, H, W)
        clean_tensor: float tensor in [0, 1], shape (3, H, W)
    """

    def __init__(self, args, split="train"):
        super(HW4RestorationDataset, self).__init__()
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be one of: train, val, all")
        self.args = args
        self.split = split
        self.data_root = Path(args.hw4_data_root)
        self.patch_size = int(args.patch_size)
        self.to_tensor = ToTensor()

        split_data = make_hw4_split(
            data_root=self.data_root,
            split_file=args.hw4_split_file,
            val_per_task=args.hw4_val_per_task,
            seed=args.hw4_seed,
        )
        if split == "all":
            self.samples = split_data["train"] + split_data["val"]
        else:
            self.samples = split_data[split]

        if not self.samples:
            raise RuntimeError(f"HW4 {split} split is empty")

        counts = {"rain": 0, "snow": 0}
        for sample in self.samples:
            counts[sample["task"]] += 1
        print(f"HW4 {split} samples: {counts}")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        degraded_path = self.data_root / sample["degraded"]
        clean_path = self.data_root / sample["clean"]
        degraded = _read_rgb(degraded_path)
        clean = _read_rgb(clean_path)
        height, width = degraded.shape[:2]

        if degraded.shape != clean.shape:
            raise ValueError(
                f"Shape mismatch: {degraded_path} has {degraded.shape}, "
                f"but {clean_path} has {clean.shape}."
            )

        if self.split == "train":
            degraded, clean = _crop_pair(degraded, clean, self.patch_size)
            degraded, clean = random_augmentation(degraded, clean)

        metadata = {
            "filename": sample["filename"],
            "clean_filename": sample["clean_filename"],
            "task": sample["task"],
            "task_id": int(sample["task_id"]),
            "height": int(height),
            "width": int(width),
        }
        return metadata, self.to_tensor(degraded), self.to_tensor(clean)

    def __len__(self):
        return len(self.samples)


class HW4TestDataset(Dataset):
    """HW4 test set loader that preserves exact image dimensions and filenames."""

    def __init__(self, test_dir):
        super(HW4TestDataset, self).__init__()
        self.test_dir = Path(test_dir)
        if not self.test_dir.is_dir():
            raise FileNotFoundError(f"Missing HW4 test directory: {self.test_dir}")
        self.image_paths = sorted(self.test_dir.glob("*.png"), key=_natural_key)
        if not self.image_paths:
            raise RuntimeError(f"No PNG test images found in {self.test_dir}")
        self.to_tensor = ToTensor()
        print(f"HW4 test images: {len(self.image_paths)}")

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = _read_rgb(path)
        height, width = image.shape[:2]
        metadata = {
            "filename": path.name,
            "height": int(height),
            "width": int(width),
        }
        return metadata, self.to_tensor(image)

    def __len__(self):
        return len(self.image_paths)
