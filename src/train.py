"""HW4 training entry point for PromptIR.

This file keeps the original PromptIR backbone, but replaces the upstream
Denoise/Derain/Dehaze data assumptions with the NYCU HW4 paired rain/snow
pipeline. It trains one PromptIR-based model from scratch, validates with PSNR,
and saves the best checkpoint by validation PSNR.
"""

from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from net.model import PromptIR
from options import options as opt
from utils.dataset_utils import HW4RestorationDataset
from utils.loss_utils import (
    local_weighted_l1_loss,
    make_soft_pseudo_degradation_mask,
    soft_mask_supervision_loss,
)
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.submission_utils import (
    calculate_psnr_torch,
    extract_restored_output,
    load_promptir_checkpoint,
    pad_to_multiple,
    unpad,
)


def get_aux_weight(base_weight, epoch, total_epochs, decay_start_ratio=0.7):
    decay_start = int(total_epochs * decay_start_ratio)
    if epoch < decay_start:
        return base_weight
    progress = (epoch - decay_start) / max(1, total_epochs - decay_start)
    return base_weight * max(0.0, 1.0 - progress)


class EMACallback(Callback):
    """Maintain an exponential moving average of PromptIR weights.

    Validation runs with EMA weights so the monitored val_psnr matches the
    checkpointed EMA state used by infer_hw4.py.
    """

    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = decay
        self.shadow = None
        self.backup = None

    def on_fit_start(self, trainer, pl_module):
        del trainer
        self.shadow = {
            name: param.detach().clone()
            for name, param in pl_module.net.state_dict().items()
        }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        del trainer, outputs, batch, batch_idx
        if self.shadow is None:
            return
        with torch.no_grad():
            current = pl_module.net.state_dict()
            for name, value in current.items():
                if torch.is_floating_point(value):
                    self.shadow[name].mul_(self.decay).add_(
                        value.detach(), alpha=1.0 - self.decay
                    )
                else:
                    self.shadow[name] = value.detach().clone()

    def on_validation_epoch_start(self, trainer, pl_module):
        del trainer
        if self.shadow is None:
            return
        self.backup = deepcopy(pl_module.net.state_dict())
        pl_module.net.load_state_dict(self.shadow, strict=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        del trainer
        if self.backup is not None:
            pl_module.net.load_state_dict(self.backup, strict=True)
            self.backup = None

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        del trainer, pl_module
        if self.shadow is not None:
            checkpoint["ema_state_dict"] = {
                name: value.detach().cpu()
                for name, value in self.shadow.items()
            }


class PromptIRModel(pl.LightningModule):
    """Lightning wrapper for HW4 PromptIR training."""

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.args = args
        self.net = PromptIR(
            decoder=True,
            dim=args.model_dim,
            num_blocks=args.num_blocks,
            num_refinement_blocks=args.num_refinement_blocks,
            prompt_len=args.prompt_len,
            use_task_prompt_routing=not args.disable_soft_prompt_routing,
            num_tasks=2,
            use_local_weather_refine=args.use_local_weather_refine,
            lwr_expansion=args.lwr_expansion,
            lwr_use_directional=args.lwr_use_directional,
            lwr_use_dilated=args.lwr_use_dilated,
            lwr_residual_scale_init=args.lwr_residual_scale_init,
            lwr_variant=args.lwr_variant,
            lwr_zero_init_final=args.lwr_zero_init_final,
            use_hf_refine=args.use_hf_refine,
            hf_refine_blocks=args.hf_refine_blocks,
            hf_residual_scale_init=args.hf_residual_scale_init,
            use_multiscale_context_fusion=args.use_multiscale_context_fusion,
            mscf_hidden_ratio=args.mscf_hidden_ratio,
            mscf_residual_scale_init=args.mscf_residual_scale_init,
            use_intermediate_supervision=args.use_intermediate_supervision,
            use_soft_degradation_mask=args.use_soft_degradation_mask,
            mask_hidden_dim=args.mask_hidden_dim,
            mask_guidance_alpha=args.mask_guidance_alpha,
        )
        total_params = sum(param.numel() for param in self.net.parameters())
        print(
            "Model initialization: "
            f"use_hf_refine={args.use_hf_refine}, "
            f"hf_refine_blocks={args.hf_refine_blocks}, "
            f"hf_residual_scale_init={args.hf_residual_scale_init}, "
            f"use_local_weather_refine={args.use_local_weather_refine}, "
            f"lwr_variant={args.lwr_variant}, "
            f"use_multiscale_context_fusion={args.use_multiscale_context_fusion}, "
            f"mscf_hidden_ratio={args.mscf_hidden_ratio}, "
            f"mscf_residual_scale_init={args.mscf_residual_scale_init}, "
            f"use_intermediate_supervision={args.use_intermediate_supervision}, "
            f"use_soft_degradation_mask={args.use_soft_degradation_mask}, "
            f"mask_guidance_alpha={args.mask_guidance_alpha}, "
            f"total_params={total_params:,}"
        )
        self.loss_fn = nn.L1Loss()
        self.aux_cls_weight = args.aux_cls_weight
        self.validation_outputs = []

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        del batch_idx
        metadata, degraded_patch, clean_patch = batch
        task_id = metadata["task_id"].long()
        needs_aux_outputs = (
            self.args.use_intermediate_supervision
            or self.args.use_soft_degradation_mask
        )
        if needs_aux_outputs:
            outputs = self.net(
                degraded_patch,
                return_gate=True,
                return_aux=True,
            )
            restored = outputs["final"]
            gate_logits = outputs.get("gate_logits")
        else:
            outputs = {}
            restored, gate_logits = self.net(degraded_patch, return_gate=True)

        restoration_loss = self.loss_fn(restored, clean_patch)
        loss = restoration_loss
        self.log(
            "train_restore_loss",
            restoration_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/loss_final",
            restoration_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train_l1",
            restoration_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        mask_loss = restored.new_zeros(())
        local_loss = restored.new_zeros(())
        if self.args.use_soft_degradation_mask:
            pred_mask = outputs.get("pred_mask")
            if pred_mask is None:
                raise RuntimeError(
                    "Soft degradation mask is enabled, but the model did not "
                    "return pred_mask. Call PromptIR with return_aux=True."
                )
            needs_pseudo_mask = (
                self.args.use_mask_loss
                and self.args.mask_loss_weight > 0
            ) or (
                self.args.use_local_weighted_loss
                and self.args.local_loss_weight > 0
            )
            pseudo_mask = None
            if needs_pseudo_mask:
                pseudo_mask = make_soft_pseudo_degradation_mask(
                    degraded_patch,
                    clean_patch,
                    smooth_kernel_size=self.args.pseudo_mask_smooth_kernel_size,
                )
            if (
                self.args.use_mask_loss
                and self.args.mask_loss_weight > 0
                and pseudo_mask is not None
            ):
                mask_loss = soft_mask_supervision_loss(
                    pred_mask,
                    pseudo_mask,
                    loss_type=self.args.mask_loss_type,
                )
                loss = loss + self.args.mask_loss_weight * mask_loss
            if (
                self.args.use_local_weighted_loss
                and self.args.local_loss_weight > 0
                and pseudo_mask is not None
            ):
                local_loss = local_weighted_l1_loss(
                    restored,
                    clean_patch,
                    pseudo_mask,
                )
                loss = loss + self.args.local_loss_weight * local_loss

            self.log(
                "train_mask_loss",
                mask_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "train/mask_loss",
                mask_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "train_local_weighted_loss",
                local_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "train/local_weighted_loss",
                local_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        aux_l1_loss = restored.new_zeros(())
        aux_l2_loss = restored.new_zeros(())
        aux_l1_weight = 0.0
        aux_l2_weight = 0.0
        if self.args.use_intermediate_supervision:
            if self.args.use_aux_weight_decay:
                aux_l1_weight = get_aux_weight(
                    self.args.aux_l1_weight,
                    self.current_epoch,
                    self.args.epochs,
                    self.args.aux_decay_start_ratio,
                )
                aux_l2_weight = get_aux_weight(
                    self.args.aux_l2_weight,
                    self.current_epoch,
                    self.args.epochs,
                    self.args.aux_decay_start_ratio,
                )
            else:
                aux_l1_weight = self.args.aux_l1_weight
                aux_l2_weight = self.args.aux_l2_weight

            if "aux_l1" in outputs and aux_l1_weight > 0:
                aux_l1_loss = self.loss_fn(outputs["aux_l1"], clean_patch)
                loss = loss + aux_l1_weight * aux_l1_loss
            if "aux_l2" in outputs and aux_l2_weight > 0:
                aux_l2_loss = self.loss_fn(outputs["aux_l2"], clean_patch)
                loss = loss + aux_l2_weight * aux_l2_loss

            self.log(
                "train/loss_aux_l1",
                aux_l1_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "train/loss_aux_l2",
                aux_l2_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "train/aux_l1_weight",
                float(aux_l1_weight),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "train/aux_l2_weight",
                float(aux_l2_weight),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        if gate_logits is not None and self.aux_cls_weight > 0:
            cls_loss = F.cross_entropy(gate_logits, task_id)
            cls_acc = (gate_logits.argmax(dim=1) == task_id).float().mean()
            loss = loss + self.aux_cls_weight * cls_loss
            self.log("train_cls_loss", cls_loss, on_epoch=True, sync_dist=True)
            self.log("train_cls_acc", cls_acc, on_epoch=True, sync_dist=True)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_total",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/loss_total",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return loss

    def on_validation_epoch_start(self):
        self.validation_outputs = []

    def validation_step(self, batch, batch_idx):
        del batch_idx
        metadata, degraded, clean = batch
        degraded_pad, height, width = pad_to_multiple(degraded, multiple=8)
        restored_pad = extract_restored_output(self.net(degraded_pad)).clamp(0.0, 1.0)
        restored = unpad(restored_pad, height, width)
        psnr = calculate_psnr_torch(restored, clean)
        task_id = metadata["task_id"].long()
        self.validation_outputs.append({
            "psnr": psnr.detach(),
            "task_id": task_id.detach(),
        })
        return psnr

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return
        psnr_values = torch.stack([item["psnr"] for item in self.validation_outputs])
        task_values = torch.cat([
            item["task_id"].reshape(-1)
            for item in self.validation_outputs
        ])
        val_psnr = psnr_values.mean()
        self.log("val_psnr", val_psnr, prog_bar=True, sync_dist=True)
        self.log("val/psnr_final", val_psnr, prog_bar=False, sync_dist=True)

        rain_mask = task_values == 0
        snow_mask = task_values == 1
        rain_psnr = None
        snow_psnr = None
        if rain_mask.any():
            rain_psnr = psnr_values[rain_mask].mean()
            self.log(
                "val_psnr_rain",
                rain_psnr,
                prog_bar=False,
                sync_dist=True,
            )
        if snow_mask.any():
            snow_psnr = psnr_values[snow_mask].mean()
            self.log(
                "val_psnr_snow",
                snow_psnr,
                prog_bar=False,
                sync_dist=True,
            )
        rain_text = f"{rain_psnr.item():.3f}" if rain_psnr is not None else "n/a"
        snow_text = f"{snow_psnr.item():.3f}" if snow_psnr is not None else "n/a"
        self.print(
            f"Validation epoch {self.current_epoch}: "
            f"val_psnr={val_psnr.item():.3f}, "
            f"rain={rain_text}, snow={snow_text}"
        )

    def lr_scheduler_step(self, scheduler, metric):
        del metric
        scheduler.step()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=self.args.warmup_epochs,
            max_epochs=self.args.epochs,
        )
        return [optimizer], [scheduler]


def build_logger(args):
    if args.wblogger:
        return WandbLogger(project=args.wblogger, name=args.run_name)
    return TensorBoardLogger(save_dir=args.log_dir, name=args.run_name)


def main():
    print("Options")
    print(opt)
    torch.set_float32_matmul_precision(opt.matmul_precision)
    torch.backends.cudnn.benchmark = not opt.disable_cudnn_benchmark
    pl.seed_everything(opt.hw4_seed, workers=True)

    train_set = HW4RestorationDataset(opt, split="train")
    val_set = HW4RestorationDataset(opt, split="val")
    train_loader = DataLoader(
        train_set,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers,
        persistent_workers=opt.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        num_workers=max(0, min(opt.num_workers, 4)),
        persistent_workers=opt.num_workers > 0,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=opt.ckpt_dir,
            filename="epoch{epoch:03d}-psnr{val_psnr:.3f}",
            monitor="val_psnr",
            mode="max",
            save_top_k=opt.save_top_k,
            save_last=True,
            auto_insert_metric_name=False,
        )
    ]
    if not opt.disable_ema:
        callbacks.append(EMACallback(decay=opt.ema_decay))

    model = PromptIRModel(opt)
    init_ckpt_path = opt.init_ckpt_path or opt.resume_from_checkpoint
    if init_ckpt_path:
        print(f"Initializing model weights from: {init_ckpt_path}")
        load_promptir_checkpoint(
            model.net,
            init_ckpt_path,
            map_location="cpu",
            use_ema=not opt.disable_ema,
            allow_partial_load_for_lwr=opt.allow_partial_load_for_lwr,
            allow_partial_load_for_hf_refine=opt.allow_partial_load_for_hf_refine,
            allow_partial_load_for_mscf=opt.allow_partial_load_for_mscf,
            allow_partial_load_for_intermediate_supervision=(
                opt.allow_partial_load_for_intermediate_supervision
            ),
            allow_partial_load_for_soft_degradation_mask=(
                opt.allow_partial_load_for_soft_degradation_mask
            ),
        )
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator=opt.accelerator,
        devices=opt.num_gpus,
        precision=opt.precision,
        gradient_clip_val=opt.gradient_clip_val,
        logger=build_logger(opt),
        callbacks=callbacks,
        num_sanity_val_steps=0,
        log_every_n_steps=opt.log_every_n_steps,
        check_val_every_n_epoch=opt.check_val_every_n_epoch,
        limit_train_batches=opt.limit_train_batches,
        limit_val_batches=opt.limit_val_batches,
        accumulate_grad_batches=opt.accumulate_grad_batches,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    main()
