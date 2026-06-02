"""Small tensor-only color postprocessing helpers.

All functions expect BCHW float tensors in ``[0, 1]`` and return tensors in the
same convention.
"""

import torch


def rgb_to_ycbcr_centered(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor in [0, 1] to YCbCr with zero-centered Cb/Cr."""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"Expected BCHW RGB tensor, got {tuple(rgb.shape)}")
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299000 * r + 0.587000 * g + 0.114000 * b
    cb = -0.168736 * r - 0.331264 * g + 0.500000 * b
    cr = 0.500000 * r - 0.418688 * g - 0.081312 * b
    return torch.cat([y, cb, cr], dim=1)


def ycbcr_centered_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """Convert YCbCr with zero-centered Cb/Cr back to RGB."""
    if ycbcr.ndim != 4 or ycbcr.shape[1] != 3:
        raise ValueError(f"Expected BCHW YCbCr tensor, got {tuple(ycbcr.shape)}")
    y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
    r = y + 1.402000 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772000 * cb
    return torch.cat([r, g, b], dim=1)


def shrink_chroma_residual(
    prediction: torch.Tensor,
    degraded: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Keep predicted luminance but shrink Cb/Cr changes from degraded input."""
    pred_ycbcr = rgb_to_ycbcr_centered(prediction)
    degraded_ycbcr = rgb_to_ycbcr_centered(degraded)
    output_ycbcr = pred_ycbcr.clone()
    gamma = float(gamma)
    output_ycbcr[:, 1:3] = degraded_ycbcr[:, 1:3] + gamma * (
        pred_ycbcr[:, 1:3] - degraded_ycbcr[:, 1:3]
    )
    return ycbcr_centered_to_rgb(output_ycbcr).clamp(0.0, 1.0)
