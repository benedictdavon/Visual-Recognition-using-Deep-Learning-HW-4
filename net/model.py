## PromptIR: Prompting for All-in-One Blind Image Restoration
## Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2306.13090


import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from einops.layers.torch import Rearrange
import time


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    




class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



class resblock(nn.Module):
    def __init__(self, dim):

        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x




##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    """Original PromptIR prompt generation block.

    This block keeps the upstream PromptIR behavior for compatibility with the
    original denoise/derain/dehaze scripts.
    """

    def __init__(
        self,
        prompt_dim=128,
        prompt_len=5,
        prompt_size=96,
        lin_dim=192,
    ):
        super(PromptGenBlock, self).__init__()
        self.prompt_param = nn.Parameter(
            torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size)
        )
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(
            prompt_dim,
            prompt_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x, task_probs=None):
        del task_probs
        batch, channels, height, width = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = (
            prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            * self.prompt_param.unsqueeze(0).repeat(batch, 1, 1, 1, 1, 1).squeeze(1)
        )
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (height, width), mode="bilinear")
        prompt = self.conv3x3(prompt)
        return prompt


class SoftTaskPromptGenBlock(nn.Module):
    """Prompt block with soft rain/snow prompt routing.

    The model remains a single PromptIR network. Instead of hard-selecting a
    rain or snow branch, a degradation gate predicts a soft probability over
    task-specific prompt banks. The selected bank is still mixed with the
    original content-adaptive prompt weights from PromptIR.
    """

    def __init__(
        self,
        prompt_dim=128,
        prompt_len=5,
        prompt_size=96,
        lin_dim=192,
        num_tasks=2,
    ):
        super(SoftTaskPromptGenBlock, self).__init__()
        self.num_tasks = num_tasks
        self.prompt_param = nn.Parameter(
            torch.rand(num_tasks, prompt_len, prompt_dim, prompt_size, prompt_size)
        )
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(
            prompt_dim,
            prompt_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x, task_probs):
        if task_probs is None:
            task_probs = x.new_full((x.shape[0], self.num_tasks), 1.0 / self.num_tasks)

        batch, channels, height, width = x.shape
        del channels
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)

        task_mixed_prompts = torch.einsum(
            "bt,tnchw->bnchw", task_probs.to(x.dtype), self.prompt_param
        )
        prompt = task_mixed_prompts * prompt_weights[:, :, None, None, None]
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(
            prompt,
            (height, width),
            mode="bilinear",
            align_corners=False,
        )
        prompt = self.conv3x3(prompt)
        return prompt


class DegradationGate(nn.Module):
    """Predicts soft rain/snow probabilities from latent PromptIR features."""

    def __init__(self, in_dim, num_tasks=2, hidden_dim=128):
        super(DegradationGate, self).__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, x):
        return self.net(x)


class LocalWeatherRefinementHead(nn.Module):
    """Zero-initialized local feature refinement for rain/snow artifacts."""

    def __init__(
        self,
        channels: int,
        expansion: float = 1.0,
        use_directional: bool = True,
        use_dilated: bool = True,
        residual_scale_init: float = 1.0,
    ):
        super(LocalWeatherRefinementHead, self).__init__()
        hidden_channels = max(1, int(round(channels * expansion)))
        self.project_in = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
        )

        branches = [
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=True,
            )
        ]
        if use_directional:
            branches.extend([
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=(1, 7),
                    padding=(0, 3),
                    groups=hidden_channels,
                    bias=True,
                ),
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=(7, 1),
                    padding=(3, 0),
                    groups=hidden_channels,
                    bias=True,
                ),
            ])
        if use_dilated:
            branches.append(
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=2,
                    dilation=2,
                    groups=hidden_channels,
                    bias=True,
                )
            )
        self.branches = nn.ModuleList(branches)
        self.fuse = nn.Sequential(
            nn.Conv2d(
                hidden_channels * len(branches),
                hidden_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),
        )
        self.project_out = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            bias=True,
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )
        nn.init.zeros_(self.project_out.weight)
        if self.project_out.bias is not None:
            nn.init.zeros_(self.project_out.bias)

    def forward(self, features):
        local_features = self.project_in(features)
        branch_features = [branch(local_features) for branch in self.branches]
        fused = self.fuse(torch.cat(branch_features, dim=1))
        residual = self.project_out(fused)
        return features + self.residual_scale.to(residual.dtype) * residual


class NotebookLocalWeatherRefinementHead(nn.Module):
    """Checkpoint-compatible LWR head used by the Kaggle notebook experiments."""

    def __init__(
        self,
        channels: int,
        expansion: float = 1.0,
        residual_scale_init: float = 0.05,
        zero_init_final: bool = True,
    ):
        super(NotebookLocalWeatherRefinementHead, self).__init__()
        hidden_channels = max(1, int(round(channels * expansion)))

        self.pre = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
        )
        self.dw_3x3 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            bias=True,
        )
        self.dw_1x7 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=(1, 7),
            padding=(0, 3),
            groups=hidden_channels,
            bias=True,
        )
        self.dw_7x1 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=(7, 1),
            padding=(3, 0),
            groups=hidden_channels,
            bias=True,
        )
        self.dw_dilated = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=hidden_channels,
            bias=True,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
        )
        self.out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        self.residual_scale = nn.Parameter(
            torch.ones(1, dtype=torch.float32) * float(residual_scale_init)
        )

        if zero_init_final:
            nn.init.zeros_(self.out.weight)
            if self.out.bias is not None:
                nn.init.zeros_(self.out.bias)

    def forward(self, features):
        local_features = self.pre(features)
        b1 = self.dw_3x3(local_features)
        b2 = self.dw_1x7(local_features)
        b3 = self.dw_7x1(local_features)
        b4 = self.dw_dilated(local_features)
        fused = self.fuse(torch.cat([b1, b2, b3, b4], dim=1))
        residual = self.out(fused)
        return features + self.residual_scale.to(residual.dtype) * residual


class ResidualDWConvBlock(nn.Module):
    """Lightweight local high-frequency residual feature refinement."""

    def __init__(self, dim, residual_scale_init=0.1):
        super(ResidualDWConvBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )

    def forward(self, x):
        residual = self.body(x)
        return x + self.residual_scale.to(residual.dtype) * residual


class MultiScaleContextFusion(nn.Module):
    """Zero-init residual fusion of decoder features from multiple scales."""

    def __init__(
        self,
        level1_channels,
        level2_channels,
        level3_channels,
        latent_channels,
        hidden_ratio=0.5,
        residual_scale_init=0.05,
    ):
        super(MultiScaleContextFusion, self).__init__()
        hidden_channels = max(16, int(round(level1_channels * float(hidden_ratio))))
        self.project_l2 = nn.Conv2d(level2_channels, level1_channels, kernel_size=1)
        self.project_l3 = nn.Conv2d(level3_channels, level1_channels, kernel_size=1)
        self.project_latent = nn.Conv2d(latent_channels, level1_channels, kernel_size=1)
        self.context_logits = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.fuse = nn.Sequential(
            nn.Conv2d(level1_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_channels, level1_channels, kernel_size=1),
        )
        self.residual_scale = nn.Parameter(
            torch.ones(1, dtype=torch.float32) * float(residual_scale_init)
        )

        nn.init.zeros_(self.fuse[-1].weight)
        if self.fuse[-1].bias is not None:
            nn.init.zeros_(self.fuse[-1].bias)

    def _project_and_resize(self, projector, features, target_size):
        features = projector(features)
        if features.shape[-2:] != target_size:
            features = F.interpolate(
                features,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return features

    def forward(self, level1_features, level2_features, level3_features, latent_features):
        target_size = level1_features.shape[-2:]
        l2 = self._project_and_resize(self.project_l2, level2_features, target_size)
        l3 = self._project_and_resize(self.project_l3, level3_features, target_size)
        latent = self._project_and_resize(
            self.project_latent,
            latent_features,
            target_size,
        )
        weights = F.softmax(self.context_logits, dim=0).to(level1_features.dtype)
        fused = (
            weights[0] * level1_features
            + weights[1] * l2
            + weights[2] * l3
            + weights[3] * latent
        )
        residual = self.fuse(fused)
        return level1_features + self.residual_scale.to(residual.dtype) * residual


class AuxRestorationHead(nn.Module):
    """Lightweight residual RGB prediction head for intermediate supervision."""

    def __init__(self, in_channels, hidden_channels=None):
        super(AuxRestorationHead, self).__init__()
        if hidden_channels is None:
            hidden_channels = max(16, in_channels // 2)

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, feat, inp_img):
        residual = self.body(feat)
        if residual.shape[-2:] != inp_img.shape[-2:]:
            residual = F.interpolate(
                residual,
                size=inp_img.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return inp_img + residual


class DegradationMaskHead(nn.Module):
    """Predict a soft rain/snow artifact map from PromptIR image features."""

    def __init__(self, in_channels, hidden_dim=32):
        super(DegradationMaskHead, self).__init__()
        hidden_dim = max(8, int(hidden_dim))
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        final_conv = self.body[-2]
        nn.init.zeros_(final_conv.weight)
        if final_conv.bias is not None:
            nn.init.constant_(final_conv.bias, -2.0)

    def forward(self, features, out_size=None):
        mask = self.body(features)
        if out_size is not None and mask.shape[-2:] != tuple(out_size):
            mask = F.interpolate(
                mask,
                size=out_size,
                mode="bilinear",
                align_corners=False,
            )
        return mask


##########################################################################
##---------- PromptIR -----------------------


class PromptIR(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        prompt_len=5,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        decoder=False,
        use_task_prompt_routing=False,
        num_tasks=2,
        use_local_weather_refine=False,
        lwr_expansion=1.0,
        lwr_use_directional=True,
        lwr_use_dilated=True,
        lwr_residual_scale_init=1.0,
        lwr_variant="local",
        lwr_zero_init_final=True,
        use_hf_refine=False,
        hf_refine_blocks=2,
        hf_residual_scale_init=0.1,
        use_multiscale_context_fusion=False,
        mscf_hidden_ratio=0.5,
        mscf_residual_scale_init=0.05,
        use_intermediate_supervision=False,
        use_soft_degradation_mask=False,
        mask_hidden_dim=32,
        mask_guidance_alpha=0.5,
    ):
        super(PromptIR, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder = decoder
        self.use_task_prompt_routing = use_task_prompt_routing
        self.num_tasks = num_tasks
        self.use_local_weather_refine = use_local_weather_refine
        self.lwr_variant = str(lwr_variant).strip().lower()
        self.use_hf_refine = use_hf_refine
        self.use_multiscale_context_fusion = use_multiscale_context_fusion
        self.use_intermediate_supervision = use_intermediate_supervision
        self.use_soft_degradation_mask = use_soft_degradation_mask
        self.mask_guidance_alpha = float(mask_guidance_alpha)
        decoder_level2_channels = int(dim * 2 ** 1)
        refinement_channels = int(dim * 2 ** 1)
        level1_channels = dim
        level2_channels = int(dim * 2 ** 1)
        level3_channels = int(dim * 2 ** 2)
        latent_channels = int(dim * 2 ** 3)

        if self.decoder:
            prompt_block = SoftTaskPromptGenBlock if use_task_prompt_routing else PromptGenBlock
            common_kwargs = {"num_tasks": num_tasks} if use_task_prompt_routing else {}
            self.prompt1 = prompt_block(
                prompt_dim=64,
                prompt_len=prompt_len,
                prompt_size=64,
                lin_dim=level2_channels,
                **common_kwargs,
            )
            self.prompt2 = prompt_block(
                prompt_dim=128,
                prompt_len=prompt_len,
                prompt_size=32,
                lin_dim=level3_channels,
                **common_kwargs,
            )
            self.prompt3 = prompt_block(
                prompt_dim=320,
                prompt_len=prompt_len,
                prompt_size=16,
                lin_dim=latent_channels,
                **common_kwargs,
            )
            if use_task_prompt_routing:
                self.degradation_gate = DegradationGate(
                    in_dim=latent_channels,
                    num_tasks=num_tasks,
                )
            else:
                self.degradation_gate = None
        else:
            self.degradation_gate = None

        self.chnl_reduce1 = nn.Conv2d(64, 64, kernel_size=1, bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128, 128, kernel_size=1, bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320, 256, kernel_size=1, bias=bias)

        self.reduce_noise_channel_1 = nn.Conv2d(
            level1_channels + 64,
            level1_channels,
            kernel_size=1,
            bias=bias,
        )
        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=level1_channels,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(level1_channels)

        self.reduce_noise_channel_2 = nn.Conv2d(
            level2_channels + 128,
            level2_channels,
            kernel_size=1,
            bias=bias,
        )
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=level2_channels,
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(level2_channels)

        self.reduce_noise_channel_3 = nn.Conv2d(
            level3_channels + 256,
            level3_channels,
            kernel_size=1,
            bias=bias,
        )
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=level3_channels,
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(level3_channels)
        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=latent_channels,
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(level3_channels)
        self.reduce_chan_level3 = nn.Conv2d(
            level2_channels + level3_channels,
            level3_channels,
            kernel_size=1,
            bias=bias,
        )
        self.noise_level3 = TransformerBlock(
            dim=latent_channels + 320,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )
        self.reduce_noise_level3 = nn.Conv2d(
            latent_channels + 320,
            level3_channels,
            kernel_size=1,
            bias=bias,
        )

        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=level3_channels,
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(level3_channels)
        self.reduce_chan_level2 = nn.Conv2d(
            level3_channels,
            level2_channels,
            kernel_size=1,
            bias=bias,
        )
        self.noise_level2 = TransformerBlock(
            dim=level3_channels + 128,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )
        self.reduce_noise_level2 = nn.Conv2d(
            level3_channels + 128,
            level3_channels,
            kernel_size=1,
            bias=bias,
        )

        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=decoder_level2_channels,
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(level2_channels)

        self.noise_level1 = TransformerBlock(
            dim=level2_channels + 64,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )
        self.reduce_noise_level1 = nn.Conv2d(
            level2_channels + 64,
            level2_channels,
            kernel_size=1,
            bias=bias,
        )

        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=refinement_channels,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=refinement_channels,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_refinement_blocks)
                ]
            )

        if self.use_intermediate_supervision:
            self.aux_head_l2 = AuxRestorationHead(decoder_level2_channels)
            self.aux_head_l1 = AuxRestorationHead(refinement_channels)
        if self.use_soft_degradation_mask:
            self.degradation_mask_head = DegradationMaskHead(
                in_channels=dim,
                hidden_dim=mask_hidden_dim,
            )

        if self.use_local_weather_refine:
            if self.lwr_variant in {"notebook", "kaggle"}:
                self.local_weather_refine = NotebookLocalWeatherRefinementHead(
                    channels=refinement_channels,
                    expansion=lwr_expansion,
                    residual_scale_init=lwr_residual_scale_init,
                    zero_init_final=lwr_zero_init_final,
                )
            elif self.lwr_variant == "local":
                self.local_weather_refine = LocalWeatherRefinementHead(
                    channels=refinement_channels,
                    expansion=lwr_expansion,
                    use_directional=lwr_use_directional,
                    use_dilated=lwr_use_dilated,
                    residual_scale_init=lwr_residual_scale_init,
                )
            else:
                raise ValueError(
                    "lwr_variant must be one of {'local', 'notebook', 'kaggle'}, "
                    f"got {lwr_variant!r}."
                )
        if self.use_hf_refine:
            self.hf_refine = nn.Sequential(
                *[
                    ResidualDWConvBlock(
                        refinement_channels,
                        residual_scale_init=hf_residual_scale_init,
                    )
                    for _ in range(hf_refine_blocks)
                ]
            )
        if self.use_multiscale_context_fusion:
            self.multiscale_context_fusion = MultiScaleContextFusion(
                level1_channels=refinement_channels,
                level2_channels=level2_channels,
                level3_channels=level3_channels,
                latent_channels=level3_channels,
                hidden_ratio=mscf_hidden_ratio,
                residual_scale_init=mscf_residual_scale_init,
            )

        self.output = nn.Conv2d(
            refinement_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    def forward(
        self,
        inp_img,
        noise_emb=None,
        return_gate=False,
        return_aux=False,
    ):
        del noise_emb
        gate_logits = None
        task_probs = None
        aux_outputs = {}

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        if self.decoder:
            if self.use_task_prompt_routing and self.degradation_gate is not None:
                gate_logits = self.degradation_gate(latent)
                task_probs = F.softmax(gate_logits, dim=1)

            dec3_param = self.prompt3(latent, task_probs=task_probs)
            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)
        fusion_latent = latent

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3, task_probs=task_probs)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)
        fusion_level3 = out_dec_level3

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        if self.use_intermediate_supervision and return_aux:
            aux_outputs["aux_l2"] = self.aux_head_l2(out_dec_level2, inp_img)
        if self.decoder:
            dec1_param = self.prompt1(out_dec_level2, task_probs=task_probs)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)
        fusion_level2 = out_dec_level2

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        if self.use_intermediate_supervision and return_aux:
            aux_outputs["aux_l1"] = self.aux_head_l1(out_dec_level1, inp_img)
        out_dec_level1 = self.refinement(out_dec_level1)
        if self.use_multiscale_context_fusion:
            out_dec_level1 = self.multiscale_context_fusion(
                out_dec_level1,
                fusion_level2,
                fusion_level3,
                fusion_latent,
            )
        if self.use_local_weather_refine:
            out_dec_level1 = self.local_weather_refine(out_dec_level1)
        if self.use_hf_refine:
            out_dec_level1 = self.hf_refine(out_dec_level1)
        base_output = self.output(out_dec_level1) + inp_img
        final_output = base_output
        pred_mask = None
        if self.use_soft_degradation_mask:
            pred_mask = self.degradation_mask_head(
                out_enc_level1,
                out_size=inp_img.shape[-2:],
            )
            residual = base_output - inp_img
            final_output = inp_img + residual * (
                1.0 + self.mask_guidance_alpha * pred_mask
            )

        if return_aux:
            outputs = {"final": final_output}
            outputs.update(aux_outputs)
            if self.use_soft_degradation_mask:
                outputs["pred_mask"] = pred_mask
                outputs["base_out"] = base_output
            if return_gate:
                outputs["gate_logits"] = gate_logits
            return outputs

        if return_gate:
            return final_output, gate_logits
        return final_output
