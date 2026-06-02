"""HW4 validation, inference, and submission helpers.

Tensor conventions used here:

- model inputs and outputs are BCHW float tensors in ``[0, 1]``;
- padding is applied only for model compatibility and removed before export;
- submission arrays are CHW ``uint8`` values saved under original PNG filenames.
"""

from pathlib import Path
import re
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def natural_key(path_like):
    """Natural sort key for numeric filenames such as 2.png and 10.png."""
    text = str(path_like)
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", text)]


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int = 8,
) -> Tuple[torch.Tensor, int, int]:
    """Pad BCHW tensor to a spatial multiple and return original H/W."""
    if tensor.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(tensor.shape)}")
    height, width = tensor.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, height, width

    mode = "reflect"
    if pad_h >= height or pad_w >= width:
        mode = "replicate"
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
    return padded, height, width


def unpad(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Remove spatial padding from a BCHW tensor."""
    return tensor[..., :height, :width]


def calculate_psnr_torch(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Compute mean PSNR over a BCHW batch."""
    prediction = prediction.clamp(0.0, data_range)
    target = target.clamp(0.0, data_range)
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    psnr = torch.where(
        mse <= 1e-12,
        torch.full_like(mse, 100.0),
        10.0 * torch.log10((data_range ** 2) / mse),
    )
    return psnr.mean()


def extract_restored_output(model_output) -> torch.Tensor:
    """Return the restored tensor from tensor/dict/tuple model outputs."""
    if torch.is_tensor(model_output):
        return model_output
    if isinstance(model_output, dict):
        if "final" in model_output:
            return model_output["final"]
        if "restored" in model_output:
            return model_output["restored"]
        raise KeyError(
            "Model output dict must contain a 'final' or 'restored' tensor."
        )
    if isinstance(model_output, (tuple, list)):
        if not model_output:
            raise ValueError("Model output tuple/list is empty.")
        first = model_output[0]
        if not torch.is_tensor(first):
            raise TypeError(
                "Expected first tuple/list model output to be a tensor, got "
                f"{type(first)}."
            )
        return first
    raise TypeError(f"Unsupported model output type: {type(model_output)}")


def _tile_positions(length: int, tile: int, stride: int):
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile, stride))
    positions.append(length - tile)
    return sorted(set(positions))


def _patch_grid_positions(length: int, tile: int, grid: int):
    if length <= tile:
        return [0]
    grid = max(2, int(grid))
    min_grid_for_coverage = int(np.ceil((length - tile) / max(1, tile - 1))) + 1
    grid = max(grid, min_grid_for_coverage)
    positions = np.rint(np.linspace(0, length - tile, num=grid)).astype(np.int64)
    return sorted(set(int(position) for position in positions))


def _gaussian_patch_mask(
    tile: int,
    device: torch.device,
    dtype: torch.dtype,
    sigma_scale: float = 0.25,
) -> torch.Tensor:
    sigma = max(float(tile) * float(sigma_scale), 1e-6)
    coords = torch.arange(tile, device=device, dtype=torch.float32)
    coords = coords - (tile - 1) / 2.0
    gaussian_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    mask = gaussian_1d[:, None] * gaussian_1d[None, :]
    mask = mask / mask.max().clamp_min(1e-12)
    return mask.to(dtype=dtype).view(1, 1, tile, tile)


def tile_forward(
    model,
    tensor: torch.Tensor,
    tile_size: int = 256,
    tile_overlap: int = 32,
) -> torch.Tensor:
    """Run overlap-tiled inference on a padded BCHW tensor."""
    _, _, height, width = tensor.shape
    tile = min(tile_size, height, width)
    tile = max(8, tile - tile % 8)
    tile_overlap = min(tile_overlap, tile - 8)
    stride = tile - tile_overlap

    if tile >= height and tile >= width:
        return extract_restored_output(model(tensor))

    output = torch.zeros_like(tensor)
    weight = torch.zeros_like(tensor)
    for top in _tile_positions(height, tile, stride):
        for left in _tile_positions(width, tile, stride):
            patch = tensor[..., top:top + tile, left:left + tile]
            pred = extract_restored_output(model(patch)).clamp(0.0, 1.0)
            mask = torch.ones_like(pred)
            output[..., top:top + tile, left:left + tile] += pred
            weight[..., top:top + tile, left:left + tile] += mask
    return output / weight.clamp_min(1e-8)


def gaussian_patch_forward(
    model,
    tensor: torch.Tensor,
    tile_size: int = 128,
    patch_grid: int = 6,
    gaussian_sigma_scale: float = 0.25,
    patch_batch_size: int = 1,
) -> torch.Tensor:
    """Run Gaussian-weighted patch testing on a padded BCHW tensor.

    For HW4 256x256 test images, ``tile_size=128`` and ``patch_grid=6`` creates
    the 36-patch inference path used in previous experiments.
    """
    batch, _, height, width = tensor.shape
    tile = min(tile_size, height, width)
    tile = max(8, tile - tile % 8)

    if tile >= height and tile >= width:
        return extract_restored_output(model(tensor))

    mask = _gaussian_patch_mask(
        tile,
        device=tensor.device,
        dtype=tensor.dtype,
        sigma_scale=gaussian_sigma_scale,
    )
    output = torch.zeros_like(tensor)
    weight = torch.zeros_like(tensor)
    coords = [
        (top, left)
        for top in _patch_grid_positions(height, tile, patch_grid)
        for left in _patch_grid_positions(width, tile, patch_grid)
    ]
    patch_batch_size = max(1, int(patch_batch_size))
    for start in range(0, len(coords), patch_batch_size):
        coord_batch = coords[start:start + patch_batch_size]
        patches = torch.cat(
            [
                tensor[..., top:top + tile, left:left + tile]
                for top, left in coord_batch
            ],
            dim=0,
        )
        preds = extract_restored_output(model(patches)).clamp(0.0, 1.0)
        patch_mask = mask.to(dtype=preds.dtype)
        for idx, (top, left) in enumerate(coord_batch):
            pred = preds[idx * batch:(idx + 1) * batch]
            output[..., top:top + tile, left:left + tile] += pred * patch_mask
            weight[..., top:top + tile, left:left + tile] += patch_mask
    return output / weight.clamp_min(1e-8)


def forward_with_padding(
    model,
    tensor: torch.Tensor,
    multiple: int = 8,
    use_tile: bool = False,
    tile_size: int = 256,
    tile_overlap: int = 32,
    use_gaussian_patch: bool = False,
    patch_grid: int = 6,
    gaussian_sigma_scale: float = 0.25,
    patch_batch_size: int = 1,
) -> torch.Tensor:
    """Pad input, run the selected inference path, and unpad original size.

    The caller receives a BCHW float tensor in ``[0, 1]`` with the same H/W as
    the input tensor.
    """
    padded, height, width = pad_to_multiple(tensor, multiple=multiple)
    if use_gaussian_patch:
        output = gaussian_patch_forward(
            model,
            padded,
            tile_size=tile_size,
            patch_grid=patch_grid,
            gaussian_sigma_scale=gaussian_sigma_scale,
            patch_batch_size=patch_batch_size,
        )
    elif use_tile:
        output = tile_forward(
            model,
            padded,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
    else:
        output = extract_restored_output(model(padded))
    output = output.clamp(0.0, 1.0)
    return unpad(output, height, width)


def _augment_tensor(tensor: torch.Tensor, mode: int) -> torch.Tensor:
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
    raise ValueError(f"Invalid TTA mode: {mode}")


def _deaugment_tensor(tensor: torch.Tensor, mode: int) -> torch.Tensor:
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
    raise ValueError(f"Invalid TTA mode: {mode}")


def predict_with_tta(
    model,
    tensor: torch.Tensor,
    multiple: int = 8,
    use_tile: bool = False,
    tile_size: int = 256,
    tile_overlap: int = 32,
    use_gaussian_patch: bool = False,
    patch_grid: int = 6,
    gaussian_sigma_scale: float = 0.25,
    patch_batch_size: int = 1,
) -> torch.Tensor:
    """Run x8 self-ensemble inference and average de-augmented predictions."""
    predictions = []
    for mode in range(8):
        aug = _augment_tensor(tensor, mode)
        pred = forward_with_padding(
            model,
            aug,
            multiple=multiple,
            use_tile=use_tile,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            use_gaussian_patch=use_gaussian_patch,
            patch_grid=patch_grid,
            gaussian_sigma_scale=gaussian_sigma_scale,
            patch_batch_size=patch_batch_size,
        )
        predictions.append(_deaugment_tensor(pred, mode))
    return torch.stack(predictions, dim=0).mean(dim=0).clamp(0.0, 1.0)


def tensor_to_uint8_chw(tensor: torch.Tensor) -> np.ndarray:
    """Convert one restored image tensor to the official CHW uint8 array."""
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("Only batch size 1 can be converted to one image")
        tensor = tensor[0]
    array = tensor.detach().cpu().clamp(0.0, 1.0).numpy()
    array = np.rint(array * 255.0).clip(0, 255).astype(np.uint8)
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"Expected CHW image with 3 channels, got {array.shape}")
    return array


def save_npz_submission(
    predictions: Dict[str, np.ndarray],
    output_path: str,
    expected_count: Optional[int] = 100,
) -> None:
    """Save and validate the official HW4 ``pred.npz`` format."""
    if expected_count is not None and len(predictions) != expected_count:
        raise ValueError(
            f"Expected {expected_count} predictions, got {len(predictions)}."
        )
    for key, value in predictions.items():
        if not key.endswith(".png"):
            raise ValueError(f"Submission key must keep .png extension: {key}")
        if value.ndim != 3 or value.shape[0] != 3:
            raise ValueError(
                f"Prediction {key} must have shape (3,H,W), got {value.shape}"
            )
        if value.dtype != np.uint8:
            raise ValueError(f"Prediction {key} must be uint8, got {value.dtype}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **predictions)


def _strip_prefix_if_present(state_dict, prefix: str):
    if state_dict and all(key.startswith(prefix) for key in state_dict.keys()):
        return {
            key[len(prefix):]: value
            for key, value in state_dict.items()
        }
    return state_dict


def strip_lightning_prefix(state_dict):
    """Convert wrapped state dict keys to raw PromptIR keys when needed."""
    state_dict = _strip_prefix_if_present(state_dict, "net.")
    state_dict = _strip_prefix_if_present(state_dict, "module.")
    return state_dict


def _print_checkpoint_key_report(missing_keys, unexpected_keys):
    print("Checkpoint load missing keys:")
    if missing_keys:
        for key in missing_keys:
            print(f"  {key}")
    else:
        print("  <none>")

    print("Checkpoint load unexpected keys:")
    if unexpected_keys:
        for key in unexpected_keys:
            print(f"  {key}")
    else:
        print("  <none>")


def _select_promptir_state_dict(checkpoint, use_ema: bool = True):
    """Select the actual PromptIR weights from common checkpoint wrappers."""
    if use_ema and isinstance(checkpoint, dict) and "ema_state_dict" in checkpoint:
        return checkpoint["ema_state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")


def detect_lwr_variant_from_checkpoint(
    checkpoint_path: str,
    map_location="cpu",
    use_ema: bool = True,
):
    """Return the LWR head variant implied by checkpoint key names, if present."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = strip_lightning_prefix(
        _select_promptir_state_dict(checkpoint, use_ema=use_ema)
    )
    keys = set(state_dict.keys())
    if any(key.startswith("local_weather_refine.pre.") for key in keys):
        return "notebook"
    if any(key.startswith("local_weather_refine.project_in.") for key in keys):
        return "local"
    return None


def load_promptir_checkpoint(
    model,
    checkpoint_path: str,
    map_location="cpu",
    use_ema: bool = True,
    allow_partial_load_for_lwr: bool = False,
    allow_partial_load_for_hf_refine: bool = False,
    allow_partial_load_for_mscf: bool = False,
    allow_partial_load_for_intermediate_supervision: bool = False,
    allow_partial_load_for_soft_degradation_mask: bool = False,
):
    """Load PromptIR weights without changing checkpoint semantics.

    EMA weights are preferred when present and ``use_ema`` is true. Partial
    loading is only allowed for explicitly requested optional modules.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = strip_lightning_prefix(
        _select_promptir_state_dict(checkpoint, use_ema=use_ema)
    )
    if allow_partial_load_for_lwr and not hasattr(model, "local_weather_refine"):
        raise ValueError(
            "allow_partial_load_for_lwr requires a model with local_weather_refine."
        )
    if allow_partial_load_for_hf_refine and not hasattr(model, "hf_refine"):
        raise ValueError(
            "allow_partial_load_for_hf_refine requires a model with hf_refine."
        )
    if (
        allow_partial_load_for_mscf
        and not hasattr(model, "multiscale_context_fusion")
    ):
        raise ValueError(
            "allow_partial_load_for_mscf requires a model with "
            "multiscale_context_fusion."
        )
    if (
        allow_partial_load_for_intermediate_supervision
        and not (
            hasattr(model, "aux_head_l1")
            and hasattr(model, "aux_head_l2")
        )
    ):
        raise ValueError(
            "allow_partial_load_for_intermediate_supervision requires a model "
            "with aux_head_l1 and aux_head_l2."
        )
    if (
        allow_partial_load_for_soft_degradation_mask
        and getattr(model, "degradation_mask_head", None) is None
    ):
        raise ValueError(
            "allow_partial_load_for_soft_degradation_mask requires a model "
            "with degradation_mask_head."
        )

    allow_partial_load = (
        allow_partial_load_for_lwr
        or allow_partial_load_for_hf_refine
        or allow_partial_load_for_mscf
        or allow_partial_load_for_intermediate_supervision
        or allow_partial_load_for_soft_degradation_mask
    )

    load_result = model.load_state_dict(
        state_dict,
        strict=not allow_partial_load,
    )
    if allow_partial_load:
        missing_keys = list(load_result.missing_keys)
        unexpected_keys = list(load_result.unexpected_keys)
        _print_checkpoint_key_report(missing_keys, unexpected_keys)
        allowed_missing_prefixes = []
        if allow_partial_load_for_lwr:
            allowed_missing_prefixes.append("local_weather_refine.")
        if allow_partial_load_for_hf_refine:
            allowed_missing_prefixes.append("hf_refine.")
        if allow_partial_load_for_mscf:
            allowed_missing_prefixes.append("multiscale_context_fusion.")
        if allow_partial_load_for_intermediate_supervision:
            allowed_missing_prefixes.extend([
                "aux_head_l1.",
                "aux_head_l2.",
            ])
        if allow_partial_load_for_soft_degradation_mask:
            allowed_missing_prefixes.append("degradation_mask_head.")
        bad_missing = [
            key for key in missing_keys
            if not any(
                key.startswith(prefix)
                for prefix in allowed_missing_prefixes
            )
        ]
        if bad_missing or unexpected_keys:
            raise RuntimeError(
                "Partial checkpoint load only allows missing keys for "
                f"{allowed_missing_prefixes} and no unexpected keys."
            )
    return model
