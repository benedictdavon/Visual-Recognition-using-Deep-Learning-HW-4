import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss


def make_soft_pseudo_degradation_mask(
    degraded: torch.Tensor,
    clean: torch.Tensor,
    smooth_kernel_size: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build a soft artifact target from paired degraded/clean tensors.

    Inputs are expected to be BCHW tensors in [0, 1]. The output remains soft;
    it is never thresholded into a binary segmentation target.
    """
    if degraded.shape != clean.shape:
        raise ValueError(
            "degraded and clean tensors must have the same shape, got "
            f"{tuple(degraded.shape)} and {tuple(clean.shape)}"
        )
    if degraded.ndim != 4:
        raise ValueError(f"Expected BCHW tensors, got shape {tuple(degraded.shape)}")

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


def soft_mask_supervision_loss(
    pred_mask: torch.Tensor,
    pseudo_mask: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    """Compare predicted and pseudo degradation maps."""
    if pred_mask.shape != pseudo_mask.shape:
        raise ValueError(
            "pred_mask and pseudo_mask must have the same shape, got "
            f"{tuple(pred_mask.shape)} and {tuple(pseudo_mask.shape)}"
        )
    if loss_type == "l1":
        return F.l1_loss(pred_mask, pseudo_mask)
    if loss_type == "mse":
        return F.mse_loss(pred_mask, pseudo_mask)
    raise ValueError(f"Unsupported mask loss type: {loss_type}")


def local_weighted_l1_loss(
    restored: torch.Tensor,
    clean: torch.Tensor,
    weight_mask: torch.Tensor,
) -> torch.Tensor:
    """L1 reconstruction loss emphasized by a soft BCHW/B1HW mask."""
    if restored.shape != clean.shape:
        raise ValueError(
            "restored and clean tensors must have the same shape, got "
            f"{tuple(restored.shape)} and {tuple(clean.shape)}"
        )
    if weight_mask.ndim != 4 or weight_mask.shape[0] != restored.shape[0]:
        raise ValueError(
            "weight_mask must be BCHW-compatible with restored, got "
            f"{tuple(weight_mask.shape)} for restored {tuple(restored.shape)}"
        )
    return (weight_mask.to(dtype=restored.dtype) * torch.abs(restored - clean)).mean()


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                     tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or(self.real_label_var.numel() != input.numel()))
            # pdb.set_trace()
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                # self.real_label_var = Variable(real_tensor, requires_grad=False)
                # self.real_label_var = torch.Tensor(real_tensor)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            # pdb.set_trace()
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                # self.fake_label_var = Variable(fake_tensor, requires_grad=False)
                # self.fake_label_var = torch.Tensor(fake_tensor)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        # pdb.set_trace()
        return self.loss(input, target_tensor)
