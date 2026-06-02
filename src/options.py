"""Command-line options for HW4 PromptIR training."""

import argparse


USE_SOFT_DEGRADATION_MASK = False
USE_MASK_LOSS = True
USE_LOCAL_WEIGHTED_LOSS = True
MASK_LOSS_WEIGHT = 0.05
LOCAL_LOSS_WEIGHT = 0.15
MASK_HIDDEN_DIM = 32
MASK_GUIDANCE_ALPHA = 0.5
MASK_LOSS_TYPE = "l1"
PSEUDO_MASK_SMOOTH_KERNEL_SIZE = 3


def limit_batches(value):
    if "." in value:
        return float(value)
    return int(value)


parser = argparse.ArgumentParser()

# General runtime
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--accelerator", type=str, default="auto")
parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs/devices to use")
parser.add_argument("--precision", type=str, default="16-mixed")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--log_every_n_steps", type=int, default=20)
parser.add_argument(
    "--matmul_precision",
    type=str,
    default="high",
    choices=["highest", "high", "medium"],
    help="Float32 matmul precision hint for NVIDIA Tensor Cores.",
)
parser.add_argument(
    "--disable_cudnn_benchmark",
    action="store_true",
    help="Disable cuDNN autotuning. Leave enabled for fixed-size training patches.",
)

# Training hyperparameters
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--warmup_epochs", type=int, default=15)
parser.add_argument("--patch_size", type=int, default=128)
parser.add_argument("--gradient_clip_val", type=float, default=1.0)
parser.add_argument(
    "--check_val_every_n_epoch",
    type=int,
    default=1,
    help="Run validation every N epochs.",
)
parser.add_argument(
    "--limit_train_batches",
    type=limit_batches,
    default=1.0,
    help="Lightning limit_train_batches value. Use <1.0 for quick experiments.",
)
parser.add_argument(
    "--limit_val_batches",
    type=limit_batches,
    default=1.0,
    help="Lightning limit_val_batches value. Use <1.0 to shorten validation.",
)
parser.add_argument(
    "--accumulate_grad_batches",
    type=int,
    default=1,
    help="Accumulate gradients across N batches before each optimizer step.",
)

# HW4 dataset and split
parser.add_argument("--hw4_data_root", type=str, default="data")
parser.add_argument("--hw4_split_file", type=str, default="splits/hw4_split_seed42.json")
parser.add_argument("--hw4_val_per_task", type=int, default=160)
parser.add_argument("--hw4_seed", type=int, default=42)

# HW4 model modifications
parser.add_argument("--model_dim", type=int, default=48)
parser.add_argument("--num_blocks", type=int, nargs=4, default=[4, 6, 6, 8])
parser.add_argument("--num_refinement_blocks", type=int, default=4)
parser.add_argument("--prompt_len", type=int, default=5)
parser.add_argument(
    "--disable_soft_prompt_routing",
    action="store_true",
    help="Disable soft rain/snow prompt routing and use original PromptIR prompts.",
)
parser.add_argument("--aux_cls_weight", type=float, default=0.02)
parser.add_argument(
    "--use_local_weather_refine",
    action="store_true",
    help="Enable the feature-level Local Weather Refinement Head.",
)
parser.add_argument("--lwr_expansion", type=float, default=1.0)
parser.add_argument(
    "--lwr_variant",
    choices=["local", "notebook", "kaggle"],
    default="local",
    help="Local Weather Refinement implementation variant.",
)
parser.add_argument(
    "--lwr_use_directional",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use 1x7 and 7x1 depthwise branches in the Local Weather Refinement Head.",
)
parser.add_argument(
    "--lwr_use_dilated",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use the dilated 3x3 depthwise branch in the Local Weather Refinement Head.",
)
parser.add_argument("--lwr_residual_scale_init", type=float, default=1.0)
parser.add_argument(
    "--lwr_zero_init_final",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Zero-initialize the final projection in notebook/kaggle LWR.",
)
parser.add_argument(
    "--allow_partial_load_for_lwr",
    action="store_true",
    help="Allow loading a non-LWR checkpoint into an LWR model when only LWR keys are missing.",
)
parser.add_argument(
    "--use_hf_refine",
    action="store_true",
    help="Enable the feature-level High-Frequency Refinement Head.",
)
parser.add_argument("--hf_refine_blocks", type=int, default=2)
parser.add_argument("--hf_residual_scale_init", type=float, default=0.1)
parser.add_argument(
    "--allow_partial_load_for_hf_refine",
    action="store_true",
    help="Allow loading a non-HF-refine checkpoint into an HF-refine model when only HF keys are missing.",
)
parser.add_argument(
    "--use_multiscale_context_fusion",
    action="store_true",
    help="Enable zero-init multi-scale decoder context fusion before the output head.",
)
parser.add_argument(
    "--mscf_hidden_ratio",
    type=float,
    default=0.5,
    help="Hidden-channel ratio for multi-scale context fusion.",
)
parser.add_argument(
    "--mscf_residual_scale_init",
    type=float,
    default=0.05,
    help="Initial residual scale for multi-scale context fusion.",
)
parser.add_argument(
    "--allow_partial_load_for_mscf",
    action="store_true",
    help="Allow loading a non-MSCF checkpoint into an MSCF model when only MSCF keys are missing.",
)
parser.add_argument(
    "--use_intermediate_supervision",
    action="store_true",
    help="Enable auxiliary decoder RGB heads for deep supervision during training.",
)
parser.add_argument("--aux_l1_weight", type=float, default=0.10)
parser.add_argument("--aux_l2_weight", type=float, default=0.05)
parser.add_argument(
    "--use_aux_weight_decay",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Linearly decay intermediate-supervision weights near the end of training.",
)
parser.add_argument("--aux_decay_start_ratio", type=float, default=0.70)
parser.add_argument(
    "--allow_partial_load_for_intermediate_supervision",
    action="store_true",
    help=(
        "Allow loading a checkpoint without auxiliary decoder heads into an "
        "intermediate-supervision model."
    ),
)
parser.add_argument(
    "--use_soft_degradation_mask",
    action=argparse.BooleanOptionalAction,
    default=USE_SOFT_DEGRADATION_MASK,
    help=(
        "Enable soft degradation-map prediction and mask-guided residual "
        "refinement. The default is False to keep the baseline path unchanged."
    ),
)
parser.add_argument(
    "--mask_hidden_dim",
    type=int,
    default=MASK_HIDDEN_DIM,
    help="Hidden channels in the lightweight soft degradation mask head.",
)
parser.add_argument(
    "--mask_guidance_alpha",
    type=float,
    default=MASK_GUIDANCE_ALPHA,
    help="Residual gating strength: output = x + residual * (1 + alpha * mask).",
)
parser.add_argument(
    "--use_mask_loss",
    action=argparse.BooleanOptionalAction,
    default=USE_MASK_LOSS,
    help="Apply auxiliary L1/MSE supervision to the predicted soft mask.",
)
parser.add_argument(
    "--mask_loss_weight",
    type=float,
    default=MASK_LOSS_WEIGHT,
    help="Weight for soft-mask pseudo supervision.",
)
parser.add_argument(
    "--mask_loss_type",
    type=str,
    choices=["l1", "mse"],
    default=MASK_LOSS_TYPE,
    help="Loss type for soft pseudo-mask supervision.",
)
parser.add_argument(
    "--use_local_weighted_loss",
    action=argparse.BooleanOptionalAction,
    default=USE_LOCAL_WEIGHTED_LOSS,
    help="Apply pseudo-mask weighted reconstruction loss.",
)
parser.add_argument(
    "--local_loss_weight",
    type=float,
    default=LOCAL_LOSS_WEIGHT,
    help="Weight for pseudo-mask weighted reconstruction loss.",
)
parser.add_argument(
    "--pseudo_mask_smooth_kernel_size",
    type=int,
    default=PSEUDO_MASK_SMOOTH_KERNEL_SIZE,
    help="Odd average-pooling kernel used to smooth soft pseudo masks.",
)
parser.add_argument(
    "--allow_partial_load_for_soft_degradation_mask",
    action="store_true",
    help=(
        "Allow loading a checkpoint without soft-mask-head keys into a "
        "soft-degradation-mask model."
    ),
)
parser.add_argument(
    "--disable_ema",
    action="store_true",
    help="Disable EMA validation/checkpoint state.",
)
parser.add_argument("--ema_decay", type=float, default=0.999)

# Checkpointing and logging
parser.add_argument("--ckpt_dir", type=str, default="train_ckpt_hw4")
parser.add_argument("--save_top_k", type=int, default=3)
parser.add_argument("--wblogger", type=str, default=None)
parser.add_argument("--log_dir", type=str, default="logs")
parser.add_argument("--run_name", type=str, default="promptir_hw4")

# Legacy upstream PromptIR arguments retained for compatibility with test.py/demo.py.
parser.add_argument(
    "--de_type",
    nargs="+",
    default=["derain", "desnow"],
    help="Legacy argument; HW4 training uses rain/snow pairs from hw4_data_root.",
)
parser.add_argument("--data_file_dir", type=str, default="data_dir/")
parser.add_argument("--denoise_dir", type=str, default="data/Train/Denoise/")
parser.add_argument("--derain_dir", type=str, default="data/Train/Derain/")
parser.add_argument("--dehaze_dir", type=str, default="data/Train/Dehaze/")
parser.add_argument("--output_path", type=str, default="output/")
parser.add_argument("--ckpt_path", type=str, default="ckpt/Denoise/")
parser.add_argument(
    "--init_ckpt_path",
    type=str,
    default=None,
    help="Initialize model weights from a checkpoint, but start a fresh training run.",
)
parser.add_argument(
    "--resume_from_checkpoint",
    type=str,
    default=None,
    help="Alias for --init_ckpt_path in this HW4 trainer; loads weights only.",
)

options = parser.parse_args()
