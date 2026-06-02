"""Create a DDP Kaggle notebook for Deep PromptIR 128 multi-loss training."""

from __future__ import annotations

import json
from pathlib import Path

import create_kaggle_hw4_notebook as base


TITLE = r"""
# HW4 Deep PromptIR 128 Multi-Loss DDP

This notebook keeps the same fast `torchrun` / DistributedDataParallel training path as `notebooks/hw4_promptir_kaggle.ipynb`, but changes the training loss to a configurable restoration multi-loss.

Default experiment:

- Deep PromptIR 128 from scratch
- patch size 128 only
- 2x T4 DDP on Kaggle when available
- no pretrained perceptual/VGG loss
- loss = L1 + SSIM/MS-SSIM + Charbonnier + Sobel gradient

Use this notebook for the fair multi-loss vs Deep PromptIR 128 L1 baseline comparison.
"""


LOSS_CONFIG = r"""
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
"""


LOSS_LIB_CODE = r'''
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
    return torch.sqrt((pred - target).pow(2) + eps * eps).mean()


def sobel_gradient_loss(pred, target):
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


def build_restoration_loss(cfg):
    if cfg.USE_MULTI_LOSS:
        return RestorationMultiLoss(
            use_ms_ssim=cfg.USE_MS_SSIM,
            l1_weight=cfg.L1_WEIGHT,
            ssim_weight=cfg.SSIM_WEIGHT,
            charbonnier_weight=cfg.CHARBONNIER_WEIGHT,
            gradient_weight=cfg.GRADIENT_WEIGHT,
            charbonnier_eps=cfg.CHARBONNIER_EPS,
            ms_ssim_levels=cfg.MS_SSIM_LEVELS,
            ssim_window_size=cfg.SSIM_WINDOW_SIZE,
        )
    return RestorationL1Loss()


def restoration_loss_description(cfg):
    if not cfg.USE_MULTI_LOSS:
        return "plain L1"
    flavor = "MS-SSIM" if cfg.USE_MS_SSIM else "SSIM"
    return (
        f"L1*{cfg.L1_WEIGHT:g} + {flavor}*{cfg.SSIM_WEIGHT:g} + "
        f"Charbonnier*{cfg.CHARBONNIER_WEIGHT:g} + Gradient*{cfg.GRADIENT_WEIGHT:g}"
    )
'''


LOSS_SANITY_CELL = r"""
write_config_py()
sys.path.insert(0, "/kaggle/working")
import importlib
import hw4_config as C
import hw4_lib
importlib.reload(hw4_lib)

loss_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_fn = hw4_lib.build_restoration_loss(C).to(loss_device)
pred = torch.rand(1, 3, 128, 128, device=loss_device, requires_grad=True)
target = torch.rand(1, 3, 128, 128, device=loss_device)
total_loss, parts = loss_fn(pred, target)
print("Loss description:", hw4_lib.restoration_loss_description(C))
print("Using pytorch_msssim:", hw4_lib.HAS_PYTORCH_MSSSIM)
print("Sanity total loss:", float(total_loss.detach().cpu()))
print("Sanity components:", {key: float(value.detach().cpu()) for key, value in parts.items()})
total_loss.backward()
assert pred.grad is not None
assert torch.isfinite(pred.grad).all()
print("Loss backward sanity check: OK")
del loss_fn, pred, target, total_loss, parts
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"""


def make_config() -> str:
    config = base.CONFIG
    config = config.replace('RUN_NAME = "deep_promptir_kaggle"', 'RUN_NAME = "exp18_deep128_multiloss_ddp_scratch128"')
    config = config.replace("AUX_CLS_WEIGHT = 0.0\n\nUSE_DDP = True", "AUX_CLS_WEIGHT = 0.0\n\n" + LOSS_CONFIG.strip() + "\n\nUSE_DDP = True")
    config = config.replace("RUN_ONLY_STAGE = None", 'RUN_ONLY_STAGE = "scratch128"')
    config = config.replace("TRAIN_LOG_EVERY_N_STEPS = 50", "TRAIN_LOG_EVERY_N_STEPS = 0\nSHOW_PROGRESS_BARS = False")
    config = config.replace("MAKE_SUBMISSION = True", "MAKE_SUBMISSION = False")
    config = config.replace(
        'print("Output dir:", OUTPUT_DIR)',
        'print("Output dir:", OUTPUT_DIR)\n'
        'print("RUN_ONLY_STAGE:", RUN_ONLY_STAGE)\n'
        'print("SHOW_PROGRESS_BARS:", SHOW_PROGRESS_BARS)\n'
        'print("Loss config:", {\n'
        '    "use_multi_loss": USE_MULTI_LOSS,\n'
        '    "use_ms_ssim": USE_MS_SSIM,\n'
        '    "l1": L1_WEIGHT,\n'
        '    "ssim": SSIM_WEIGHT,\n'
        '    "charbonnier": CHARBONNIER_WEIGHT,\n'
        '    "gradient": GRADIENT_WEIGHT,\n'
        '})',
    )
    return config


def make_utility_cell() -> str:
    utility = base.UTILITY_CELL
    utility = utility.replace(
        '"DUAL_PIXEL_TASK", "USE_SOFT_PROMPT_ROUTING", "AUX_CLS_WEIGHT",',
        '"DUAL_PIXEL_TASK", "USE_SOFT_PROMPT_ROUTING", "AUX_CLS_WEIGHT",\n'
        '        "USE_MULTI_LOSS", "USE_MS_SSIM", "L1_WEIGHT", "SSIM_WEIGHT",\n'
        '        "CHARBONNIER_WEIGHT", "GRADIENT_WEIGHT", "CHARBONNIER_EPS",\n'
        '        "MS_SSIM_LEVELS", "SSIM_WINDOW_SIZE",',
    )
    utility = utility.replace(
        '"TRAIN_LOG_EVERY_N_STEPS",',
        '"TRAIN_LOG_EVERY_N_STEPS", "SHOW_PROGRESS_BARS",',
    )
    return utility


def make_script_generation() -> str:
    script = base.SCRIPT_GENERATION
    tqdm_anchor = """try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None
"""
    script = script.replace(tqdm_anchor, tqdm_anchor + "\n\n" + LOSS_LIB_CODE.strip() + "\n", 1)
    script = script.replace(
        "def validate_model(model, dataset, device, use_tile=False, tile_size=256, tile_overlap=32, use_tta=False, amp=True):",
        "def validate_model(model, dataset, device, use_tile=False, tile_size=256, tile_overlap=32, use_tta=False, amp=True, show_progress=True):",
    )
    script = script.replace(
        '        disable=not is_rank0(),\n'
        '        dynamic_ncols=True,\n'
        '        leave=False,\n'
        '        desc="validation",',
        '        disable=(not is_rank0()) or (not show_progress),\n'
        '        dynamic_ncols=True,\n'
        '        leave=False,\n'
        '        desc="validation",',
    )
    script = script.replace(
        "                disable=not is_rank0(),",
        "                disable=(not is_rank0()) or (not cfg.SHOW_PROGRESS_BARS),",
    )
    script = script.replace(
        "        scaler = torch.amp.GradScaler(\"cuda\", enabled=cfg.PRECISION == \"AMP\" and device.type == \"cuda\")\n        loss_fn = nn.L1Loss()",
        "        scaler = torch.amp.GradScaler(\"cuda\", enabled=cfg.PRECISION == \"AMP\" and device.type == \"cuda\")\n"
        "        loss_fn = build_restoration_loss(cfg).to(device)\n"
        "        if is_rank0():\n"
        "            log_rank0(\"Using restoration loss:\", restoration_loss_description(cfg))\n"
        "            log_rank0(\"Using pytorch_msssim:\", HAS_PYTORCH_MSSSIM)",
    )
    script = script.replace(
        '                writer.writerow(["stage", "epoch", "train_l1", "val_psnr", "val_rain", "val_snow", "lr", "elapsed_sec"])',
        '                writer.writerow(["stage", "epoch", "train_total_loss", "train_l1", "train_ssim", "train_charbonnier", "train_gradient", "val_psnr", "val_rain", "val_snow", "lr", "elapsed_sec"])',
    )
    script = script.replace(
        "            running = 0.0\n            steps = 0",
        '            loss_keys = ["loss_total", "loss_l1", "loss_ssim", "loss_charbonnier", "loss_gradient"]\n'
        "            running = {key: 0.0 for key in loss_keys}\n"
        "            steps = 0",
    )
    script = script.replace("                    loss = loss_fn(pred, clean)", "                    loss, parts = loss_fn(pred, clean)")
    script = script.replace(
        "                running += loss.item()\n                steps += 1",
        "                steps += 1\n"
        "                for key in running:\n"
        "                    running[key] += float(parts[key].item())",
    )
    script = script.replace(
        '                        "loss": f"{running / max(1, steps):.5f}",\n                        "lr": f"{optimizer.param_groups[0][\'lr\']:.2e}",',
        '                        "loss": f"{running[\'loss_total\'] / max(1, steps):.5f}",\n'
        '                        "l1": f"{running[\'loss_l1\'] / max(1, steps):.5f}",\n'
        '                        "ssim": f"{running[\'loss_ssim\'] / max(1, steps):.5f}",\n'
        '                        "lr": f"{optimizer.param_groups[0][\'lr\']:.2e}",',
    )
    script = script.replace(
        '                        f"train_l1={running / max(1, steps):.5f} "\n                        f"lr={optimizer.param_groups[0][\'lr\']:.3e}"',
        '                        f"train_total={running[\'loss_total\'] / max(1, steps):.5f} "\n'
        '                        f"l1={running[\'loss_l1\'] / max(1, steps):.5f} "\n'
        '                        f"ssim={running[\'loss_ssim\'] / max(1, steps):.5f} "\n'
        '                        f"charb={running[\'loss_charbonnier\'] / max(1, steps):.5f} "\n'
        '                        f"grad={running[\'loss_gradient\'] / max(1, steps):.5f} "\n'
        '                        f"lr={optimizer.param_groups[0][\'lr\']:.3e}"',
    )
    script = script.replace(
        "            if distributed:\n                dist.barrier()\n            val = {\"overall\": float(\"nan\"), \"rain\": float(\"nan\"), \"snow\": float(\"nan\")}",
        "            stats = torch.tensor(\n"
        "                [running[key] for key in loss_keys] + [float(steps)],\n"
        "                device=device,\n"
        "                dtype=torch.float64,\n"
        "            )\n"
        "            if distributed:\n"
        "                dist.all_reduce(stats, op=dist.ReduceOp.SUM)\n"
        "                dist.barrier()\n"
        "            total_train_steps = max(1.0, float(stats[-1].item()))\n"
        "            avg = {key: float(stats[idx].item() / total_train_steps) for idx, key in enumerate(loss_keys)}\n"
        "            val = {\"overall\": float(\"nan\"), \"rain\": float(\"nan\"), \"snow\": float(\"nan\")}",
    )
    script = script.replace("                avg_loss = running / max(1, steps)", '                avg_loss = avg["loss_total"]')
    script = script.replace(
        '                val = validate_model(unwrapped, val_set, device, use_tile=False, amp=cfg.PRECISION == "AMP")',
        '                val = validate_model(unwrapped, val_set, device, use_tile=False, amp=cfg.PRECISION == "AMP", show_progress=cfg.SHOW_PROGRESS_BARS)',
    )
    script = script.replace(
        '        use_tta=cfg.USE_X8_TTA,\n'
        '        amp=cfg.PRECISION == "AMP",',
        '        use_tta=cfg.USE_X8_TTA,\n'
        '        amp=cfg.PRECISION == "AMP",\n'
        '        show_progress=cfg.SHOW_PROGRESS_BARS,',
    )
    script = script.replace(
        '                    writer.writerow([stage_name, epoch, avg_loss, val["overall"], val["rain"], val["snow"], lr, elapsed])',
        '                    writer.writerow([stage_name, epoch, avg["loss_total"], avg["loss_l1"], avg["loss_ssim"], avg["loss_charbonnier"], avg["loss_gradient"], val["overall"], val["rain"], val["snow"], lr, elapsed])',
    )
    script = script.replace(
        '                    f"Validation epoch {epoch}: val_psnr={val[\'overall\']:.3f}, "\n'
        '                    f"rain={val[\'rain\']:.3f}, snow={val[\'snow\']:.3f}, "\n'
        '                    f"train_l1={avg_loss:.5f}, lr={lr:.3e}, time={elapsed:.1f}s"\n',
        '                    f"Epoch {epoch} | train_total={avg[\'loss_total\']:.5f} | "\n'
        '                    f"l1={avg[\'loss_l1\']:.5f} | ssim={avg[\'loss_ssim\']:.5f} | "\n'
        '                    f"charb={avg[\'loss_charbonnier\']:.5f} | grad={avg[\'loss_gradient\']:.5f} | "\n'
        '                    f"val_psnr={val[\'overall\']:.3f} | rain={val[\'rain\']:.3f} | "\n'
        '                    f"snow={val[\'snow\']:.3f} | lr={lr:.3e} | time={elapsed:.1f}s"\n',
    )
    return script


def make_summary_cell() -> str:
    return base.SUMMARY_CELL.replace(
        'print("Compare public LB against current best public LB: 29.8")',
        'print("This is the DDP multi-loss scratch128 notebook. Compare simple val against Deep PromptIR 128 L1 baseline: 28.855.")',
    )


def build_notebook() -> dict:
    return {
        "cells": [
            base.md(TITLE),
            base.code(base.IMPORTS),
            base.code(make_config()),
            base.code(base.DATA_DISCOVERY),
            base.code(make_utility_cell()),
            base.code(base.DATASET_DATALOADER),
            base.code(base.MODEL_CELL),
            base.code(make_script_generation()),
            base.code(LOSS_SANITY_CELL),
            base.code(base.TRAIN_LAUNCH),
            base.code(base.VALIDATION_CELL),
            base.code(base.TEST_INFERENCE_CELL),
            base.code(base.VALIDATOR_CELL),
            base.code(base.ZIP_CELL),
            base.code(make_summary_cell()),
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
    out = Path("promptir-hw4-multiloss-ddp.ipynb")
    out.write_text(json.dumps(build_notebook(), indent=1), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
