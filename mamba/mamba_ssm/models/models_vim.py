# Copyright (c) 2024.
# [VIM ADDED] Vision Mamba (VIM) 顶层模型
# 论文: Vision Mamba: Efficient Visual Representation Learning with
#       Bidirectional State Space Model (arXiv:2401.09417, ICML 2024)
#
# 本文件是基于 mamba_ssm 标准源码新增的, 不修改任何原始代码。
# 所有 VIM 论文中的核心改动都在此处实现:
#   1. PatchEmbed         —— Section 3.2 "Patch Embedding"
#   2. cls_token          —— Section 3.2 "Class Token"
#   3. pos_embed          —— Section 3.2 "Position Embedding"
#   4. Bidirectional SSM  —— Section 3.2 "Vim Block" (通过 Mamba(bimamba_type='v2') 启用)
#   5. 分类头              —— Section 3.2 "Classification"

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

try:
    from timm.models.layers import trunc_normal_, DropPath, lecun_normal_, to_2tuple
except ImportError:
    from torch.nn.init import trunc_normal_
    DropPath = None
    def to_2tuple(x):
        return (x, x) if not isinstance(x, (list, tuple)) else tuple(x)

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# ============================================================================
# 1. PatchEmbed —— [VIM ADDED] 论文 Section 3.2 "Patch Embedding"
# ----------------------------------------------------------------------------
# 将 (B, C, H, W) 图像通过 Conv2d (kernel=patch_size, stride=patch_size)
# 切成 num_patches 个 patch, 每个 patch 投影到 embed_dim 维度。
# 输出 (B, num_patches, embed_dim) 序列, 之后送入 Mamba block。
# ============================================================================
class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding (与 ViT/DeiT 一致)."""
    def __init__(self, img_size=224, patch_size=16, stride=16,
                 in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            (img_size[0] - patch_size[0]) // stride + 1,
            (img_size[1] - patch_size[1]) // stride + 1,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)                  # (B, embed_dim, H', W')
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        x = self.norm(x)
        return x


# ============================================================================
# 2. create_block —— 构造单个 Vim Block
# ----------------------------------------------------------------------------
# 每个 Block 的 mixer 都是 Mamba(bimamba_type='v2'), 即开启双向 SSM。
# norm_cls 默认用 RMSNorm (与 Mamba 论文一致), fused_add_norm=True 调用 Triton 融合算子。
# ============================================================================
def create_block(
    d_model,
    d_state=16,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    device=None,
    dtype=None,
    bimamba_type="v2",        # [VIM ADDED] 默认启用双向 SSM v2
    if_divide_out=True,       # [VIM ADDED] v2 求和后除 2
    init_layer_scale=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    # mixer = Mamba(..., bimamba_type='v2')  [VIM 关键改动]
    mixer_cls = partial(
        Mamba,
        d_state=d_state,
        layer_idx=layer_idx,
        bimamba_type=bimamba_type,
        if_divide_out=if_divide_out,
        init_layer_scale=init_layer_scale,
        **ssm_cfg, **factory_kwargs,
    )
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm,
        eps=norm_epsilon, **factory_kwargs,
    )

    # 这里直接用一个轻量 Block (不带 mlp), 与 VIM 官方实现一致。
    # 若需要带 MLP 的版本, 也可以使用 mamba_ssm.modules.block.Block (传 mlp_cls)。
    block = _VimBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class _VimBlock(nn.Module):
    """[VIM ADDED] 简化版 Block (无 MLP), 与 VIM 官方 Block 完全一致。
    结构: (residual + drop_path(hidden_states)) → RMSNorm → Mixer(bimamba)
    """
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm,
                 fused_add_norm=False, residual_in_fp32=False, drop_path=0.):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if (DropPath is not None and drop_path > 0.) else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm))

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None,
                inference_params=None):
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


# ============================================================================
# 3. _init_weights —— GPT-2 风格初始化 (Mamba 通用, VIM 也用)
# ============================================================================
def _init_weights(module, n_layer, initializer_range=0.02,
                  rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        if hasattr(nn.init, 'kaiming_normal_'):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


# ============================================================================
# 4. VisionMamba —— [VIM ADDED] 完整 VIM 模型
# ----------------------------------------------------------------------------
# 流水线 (对应论文 Figure 2):
#   Image (B, 3, 224, 224)
#      → PatchEmbed                    (B, N, embed_dim)            [Section 3.2]
#      → prepend cls_token              (B, N+1, embed_dim)         [Section 3.2]
#      → + pos_embed                    (B, N+1, embed_dim)         [Section 3.2]
#      → [Vim Block × depth]            双向 SSM                     [Section 3.2]
#      → final RMSNorm
#      → 取 cls_token (位置 0)
#      → Linear(embed_dim → num_classes)                            [Section 3.2]
# ============================================================================
class VisionMamba(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        depth=24,
        embed_dim=192,
        d_state=16,
        channels=3,
        num_classes=1000,
        ssm_cfg=None,
        drop_rate=0.,
        drop_path_rate=0.1,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        fused_add_norm=True,
        residual_in_fp32=True,
        bimamba_type="v2",     # [VIM ADDED] 默认 v2 双向
        if_divide_out=True,
        if_cls_token=True,
        if_abs_pos_embed=True,
        init_layer_scale=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim
        self.if_cls_token = if_cls_token
        self.if_abs_pos_embed = if_abs_pos_embed
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.num_tokens = 1 if if_cls_token else 0

        # ---- (1) PatchEmbed ----
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride,
            in_chans=channels, embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # ---- (2) cls_token ----
        if if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ---- (3) 位置编码 ----
        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + self.num_tokens, embed_dim)
            )
            self.pos_drop = nn.Dropout(p=drop_rate)

        # ---- (4) Vim Block (双向 SSM) × depth ----
        # 随机深度概率随层线性增加 (DeiT 训练技巧)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList([
            create_block(
                embed_dim,
                d_state=d_state,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                drop_path=dpr[i],
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                bimamba_type=bimamba_type,
                if_divide_out=if_divide_out,
                init_layer_scale=init_layer_scale,
                **factory_kwargs,
            )
            for i in range(depth)
        ])

        # ---- (5) 末层 RMSNorm + 分类头 ----
        self.norm_f = (RMSNorm if rms_norm else nn.LayerNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs,
        )
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # ---- 初始化 ----
        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if if_cls_token:
            trunc_normal_(self.cls_token, std=.02)
        # GPT-2 风格 mamba 初始化
        self.apply(partial(_init_weights, n_layer=depth))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    # ------------------------------------------------------------------
    def forward_features(self, x: Tensor, inference_params=None) -> Tensor:
        # (B, 3, H, W) -> (B, N, D)
        x = self.patch_embed(x)
        B = x.size(0)

        if self.if_cls_token:
            cls = self.cls_token.expand(B, -1, -1)            # (B, 1, D)
            x = torch.cat((cls, x), dim=1)                    # (B, N+1, D)

        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        # 双流残差循环 (对应 Mamba 推荐的 Add->LN->Mixer 结构)
        residual = None
        hidden_states = x
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params,
            )

        # 末层 fused norm
        if not self.fused_add_norm:
            residual = (residual + hidden_states) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # 取 cls_token (位置 0) 作为图像表征 [Section 3.2 Classification]
        if self.if_cls_token:
            return hidden_states[:, 0]
        else:
            # 假设: 论文未明确时, fallback 到 mean pooling
            return hidden_states.mean(dim=1)

    def forward(self, x: Tensor, inference_params=None) -> Tensor:
        feat = self.forward_features(x, inference_params=inference_params)
        return self.head(feat)


# ============================================================================
# 5. 模型工厂 (常用规格)
# ----------------------------------------------------------------------------
# 与 VIM 论文表 1 中 Vim-Tiny / Vim-Small / Vim-Base 的配置对齐。
# ============================================================================
def vim_tiny(num_classes=1000, **kwargs):
    return VisionMamba(
        img_size=224, patch_size=16, embed_dim=192, depth=24,
        rms_norm=True, fused_add_norm=True, residual_in_fp32=True,
        bimamba_type="v2", if_divide_out=True,
        if_cls_token=True, if_abs_pos_embed=True,
        num_classes=num_classes, **kwargs,
    )


def vim_small(num_classes=1000, **kwargs):
    return VisionMamba(
        img_size=224, patch_size=16, embed_dim=384, depth=24,
        rms_norm=True, fused_add_norm=True, residual_in_fp32=True,
        bimamba_type="v2", if_divide_out=True,
        if_cls_token=True, if_abs_pos_embed=True,
        num_classes=num_classes, **kwargs,
    )


def vim_base(num_classes=1000, **kwargs):
    return VisionMamba(
        img_size=224, patch_size=16, embed_dim=768, depth=24,
        rms_norm=True, fused_add_norm=True, residual_in_fp32=True,
        bimamba_type="v2", if_divide_out=True,
        if_cls_token=True, if_abs_pos_embed=True,
        num_classes=num_classes, **kwargs,
    )