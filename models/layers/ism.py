"""
ISM — Intra-modal Sequence Modeling Block (对齐 MSAmba/Vision Mamba)
====================================================================
严格对齐 MSAmba/models/mamba_block.py::Block_GLCE 的语义:

[VIM ALIGNED] 本版本直接使用已植入 bimamba_type 支持的内置 mamba_ssm
(见 H-DCD/coupled_BI_Mamba3/mamba/mamba_ssm), 通过
    Mamba(d_model, bimamba_type="v2")
启用 Vision Mamba (Vim) 的双向 SSM, 与 MSAmba/Vim 官方语义完全一致.

保留:
    1. RMSNorm + fused_add_norm (Triton 融合残差+归一化)
    2. GLCE 在 fused_add_norm 分支内执行
    3. 双流残差: (hidden_states, residual) 跨层传递
    4. GPT-2 风格权重初始化 (_init_weights)
    5. CLS token + 可学习位置编码

参考论文:
    - Vision Mamba (Vim): Efficient Visual Representation Learning with
      Bidirectional State Space Model (ICML 2024)
"""

from __future__ import annotations

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

# ---- Mamba 导入 (标准 mamba_ssm, 含 bimamba_type="v2" 支持) ----
from mamba_ssm.modules.mamba_simple import Mamba

# ---- Mamba-3 BiMamba3 导入 (可选, 不可用时优雅降级) ----
try:
    from mamba_ssm.modules.bimamba3 import BiMamba3
    BIMAMBA3_AVAILABLE = True
except ImportError:
    BiMamba3 = None
    BIMAMBA3_AVAILABLE = False

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    try:
        from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
    except ImportError:
        RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

try:
    from timm.models.layers import trunc_normal_
except ImportError:
    from torch.nn.init import trunc_normal_

try:
    from timm.models.layers import DropPath
except ImportError:
    DropPath = None


# ---------------------------------------------------------------------------
# BiMamba: 薄包装, 直接调用原生 Mamba(bimamba_type="v2")
# ---------------------------------------------------------------------------
# [VIM ALIGNED] mamba_ssm 已植入 Vim 双向 SSM 支持 (BiMambaInnerFn + v1/v2 分支).
# 这里保留 BiMamba 类名以避免破坏旧的导入, 但内部仅实例化一个
# Mamba(bimamba_type="v2"), 其 forward 内部:
#     out_f = mamba_inner_fn_no_out_proj(xz, 正向参数, A, ...)
#     out_b = mamba_inner_fn_no_out_proj(xz.flip([-1]), 反向参数, A_b, ...)
#     out = out_f + out_b.flip([-1]); out = out_proj(out)
# 性能优于双 Mamba 实例 (共享 in_proj/out_proj, 一次 CUDA kernel 完成双扫描).
class BiMamba(nn.Module):
    """Bidirectional Mamba via native `bimamba_type="v2"` (Vim 论文一致语义)."""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, layer_idx: int = 0,
                 if_divide_out: bool = True, init_layer_scale=None, **kwargs):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=layer_idx,
            bimamba_type="v2",                # [VIM 关键: 启用双向]
            if_divide_out=if_divide_out,
            init_layer_scale=init_layer_scale,
        )

    def forward(self, x: Tensor, inference_params=None, **kwargs) -> Tensor:
        """x: (B, L, D) -> (B, L, D)"""
        return self.mamba(x, inference_params=inference_params)


# ---------------------------------------------------------------------------
# BiMamba3Wrapper: 屏蔽 Mamba-3 与 Mamba-2 不一致的构造参数 (d_conv/expand)
# ---------------------------------------------------------------------------
# Mamba-3 没有 conv1d (d_conv 不适用), 且 d_state 默认为 128 (远大于 Mamba-2 的 16).
# 这里把 ISMEncoder 旧接口 (d_state, d_conv, expand, layer_idx) 适配到 BiMamba3.
class BiMamba3Wrapper(nn.Module):
    """Bidirectional Mamba-3 适配 ISMEncoder 旧的 mixer_cls(dim) 接口."""
    def __init__(
        self, d_model: int,
        # ↓ 兼容 ISMEncoder 旧接口 (会被 partial 注入), 但仅 d_state 会用到
        d_state: int = 128, d_conv: int = 4, expand: int = 2, layer_idx: int = 0,
        # ↓ Mamba-3 专属
        headdim: int = 64, ngroups: int = 1, rope_fraction: float = 0.5,
        chunk_size: int = 64, is_mimo: bool = False, mimo_rank: int = 4,
        is_outproj_norm: bool = False,
        # ↓ BiMamba3 双向相关
        bimamba_type: str = "v2", fusion: str = "add_divide2",
        share_mimo: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert BIMAMBA3_AVAILABLE, "BiMamba3 不可用, 请检查 mamba_ssm.modules.bimamba3 是否存在"
        # d_conv / expand 在 Mamba-3 中无意义, 这里直接忽略
        self.bimamba3 = BiMamba3(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            ngroups=ngroups,
            rope_fraction=rope_fraction,
            chunk_size=chunk_size,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            is_outproj_norm=is_outproj_norm,
            bimamba_type=bimamba_type,
            fusion=fusion,
            share_mimo=share_mimo,
            layer_idx=layer_idx,
        )

    def forward(self, x: Tensor, inference_params=None, **kwargs) -> Tensor:
        """x: (B, L, D) -> (B, L, D)"""
        return self.bimamba3(x, inference_params=inference_params)


# ---------------------------------------------------------------------------
# Block_GLCE: 对齐 MSAmba 原版 (mixer 使用原生 Mamba(bimamba_type="v2"))
# ---------------------------------------------------------------------------
class Block_GLCE(nn.Module):
    """
    对齐 MSAmba/models/mamba_block.py::Block_GLCE.

    结构: Add → RMSNorm(fused) → GLCE → LN2(fused) → BiMamba
    接口: forward(hidden_states, residual) -> (hidden_states, residual)
    """
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm,
        fused_add_norm=True, residual_in_fp32=True,
        drop_path=0., use_mlp=False, seq_len=51,
        # ★ Phase 19: BSSM 风格门控 (CAGMamba 对齐)
        use_bssm_gate: bool = False,
        bssm_gate_expand: int = 2,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if (DropPath is not None and drop_path > 0.) else nn.Identity()

        if self.fused_add_norm:
            assert RMSNorm is not None, "需要安装 mamba_ssm 以使用 fused RMSNorm"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm))

        self.use_mlp = use_mlp
        if self.use_mlp:
            self.mlp = nn.Linear(dim, dim)

        # ★ BSSM 门控 (CAGMamba 公式 15-16): MLP_gate → SiLU → element-wise ⊙
        self.use_bssm_gate = use_bssm_gate
        if use_bssm_gate:
            gate_hidden = dim * bssm_gate_expand
            self.bssm_gate = nn.Sequential(
                nn.Linear(dim, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, dim),
                nn.SiLU(),  # 输出 ∈ [0, ∞), 与 BSSM 原版 SiLU 一致
            )

        # GLCE: 全局-局部上下文提取 (与 MSAmba 完全一致)
        self.seq_len = seq_len
        self.local_extractor = nn.Conv1d(self.seq_len, self.seq_len, kernel_size=3, stride=1, padding=1)
        self.global_extractor = nn.Linear(self.seq_len, self.seq_len)
        self.layer_norm_2 = norm_cls(dim)

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None,
        inference_params=None, use_checkpoint=False,
    ):
        if not self.fused_add_norm:
            # 非融合模式 fallback (一般不走这里)
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
            # GLCE: 全局 + 局部 + shortcut (与 MSAmba 完全一致)
            hidden_states_t = hidden_states.permute(0, 2, 1)
            hidden_states_t = self.global_extractor(hidden_states_t)
            hidden_states = hidden_states_t.permute(0, 2, 1) + hidden_states + self.local_extractor(hidden_states)
            # 第二个 LayerNorm (fused, prenorm=False 即后归一化)
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.layer_norm_2.weight,
                self.layer_norm_2.bias,
                residual=None,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.layer_norm_2.eps,
            )

        # BiMamba SSM (内部为 Mamba(bimamba_type="v2"), 与 MSAmba 的 Mamba(bimamba=True) 等价)
        if use_checkpoint:
            import torch.utils.checkpoint as cp
            hidden_states = cp.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)

        # ★ Phase 19: BSSM 风格门控 (CAGMamba 公式 15-16)
        #   G_ssm = σ_SiLU(MLP_gate(X))  →  Y = G_ssm ⊙ X
        #   逐元素控制 SSM 输出信息流, 抑制噪声通道
        if self.use_bssm_gate:
            gate = self.bssm_gate(hidden_states)
            hidden_states = gate * hidden_states

        if self.use_mlp:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# ---------------------------------------------------------------------------
# 权重初始化 (对齐 MSAmba 的 _init_weights, GPT-2 风格)
# ---------------------------------------------------------------------------
def _init_weights(
    module, n_layer, initializer_range=0.02,
    rescale_prenorm_residual=True, n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


# ---------------------------------------------------------------------------
# ISMEncoder: 对齐 MSAmba 的完整 ISM 流水线
# ---------------------------------------------------------------------------
class ISMEncoder(nn.Module):
    """
    对齐 MSAmba 原版的单模态序列建模模块:
        - CLS token + 可学习位置编码
        - Block_GLCE × depth (fused_add_norm + RMSNorm + BiMamba)
        - GPT-2 风格权重初始化

    Args:
        d_model:    特征维度
        seq_len:    输入序列长度 (不含 CLS, MSAmba 中为 50)
        depth:      堆叠层数 (MSAmba 中 sm_depth=2)
        d_state:    Mamba SSM 状态维度 (MSAmba 默认 16)
        d_conv:     Mamba conv 卷积核大小 (默认 4)
        expand:     Mamba 扩展比 (默认 2)
        dropout:    保留接口兼容 (未使用)
    """
    def __init__(
        self,
        d_model: int = 128,
        seq_len: int = 50,
        depth: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        # --- 新增: mixer 切换开关 ---
        mixer_type: str = "bimamba",          # "bimamba" (Mamba-2 双向) | "bimamba3" (Mamba-3 双向)
        # ↓ 仅 mixer_type == "bimamba3" 时生效
        bimamba3_headdim: int = 64,
        bimamba3_ngroups: int = 1,
        bimamba3_rope_fraction: float = 0.5,
        bimamba3_chunk_size: int = 64,
        bimamba3_is_mimo: bool = False,
        bimamba3_mimo_rank: int = 4,
        bimamba3_is_outproj_norm: bool = False,
        bimamba3_fusion: str = "add_divide2",
        bimamba3_share_mimo: bool = True,
        # ★ Phase 19: BSSM 门控
        use_bssm_gate: bool = False,
        bssm_gate_expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.depth = depth
        self.mixer_type = mixer_type
        self.use_bssm_gate = use_bssm_gate

        # CLS token + 位置编码 (对齐 MSAmba: trunc_normal_ std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

        # 根据 mixer_type 选择 mixer 类与构造参数
        if mixer_type == "bimamba":
            # Mamba-2 双向: 走 BiMamba (内部 Mamba(bimamba_type="v2"))
            mixer_partial = partial(
                BiMamba, d_state=d_state, d_conv=d_conv, expand=expand,
            )
        elif mixer_type == "bimamba3":
            assert BIMAMBA3_AVAILABLE, (
                "mixer_type='bimamba3' 需要 mamba_ssm.modules.bimamba3, "
                "请检查 H-DCD/coupled_BI_Mamba3/mamba/mamba_ssm/modules/bimamba3.py"
            )
            # Mamba-3 双向: 走 BiMamba3Wrapper, Mamba-3 默认 d_state=128
            # 这里如果用户传了 16 (Mamba-2 默认) 而不主动改 ism_d_state, 我们尊重原值,
            # 但 Mamba-3 在小 d_state 下表现可能差, 建议 ism_d_state >= 64.
            mixer_partial = partial(
                BiMamba3Wrapper,
                d_state=d_state,
                headdim=bimamba3_headdim,
                ngroups=bimamba3_ngroups,
                rope_fraction=bimamba3_rope_fraction,
                chunk_size=bimamba3_chunk_size,
                is_mimo=bimamba3_is_mimo,
                mimo_rank=bimamba3_mimo_rank,
                is_outproj_norm=bimamba3_is_outproj_norm,
                bimamba_type="v2",
                fusion=bimamba3_fusion,
                share_mimo=bimamba3_share_mimo,
            )
        else:
            raise ValueError(f"未知的 mixer_type: {mixer_type!r}, 应为 'bimamba' 或 'bimamba3'")

        # Block_GLCE 层
        norm_cls = partial(RMSNorm, eps=1e-5) if RMSNorm is not None else partial(nn.LayerNorm, eps=1e-5)

        self.layers = nn.ModuleList([
            Block_GLCE(
                dim=d_model,
                mixer_cls=partial(mixer_partial, layer_idx=i),
                norm_cls=norm_cls,
                fused_add_norm=True,
                residual_in_fp32=True,
                drop_path=0.,
                use_mlp=False,
                seq_len=seq_len + 1,  # +1 for CLS token
                use_bssm_gate=use_bssm_gate,
                bssm_gate_expand=bssm_gate_expand,
            )
            for i in range(depth)
        ])

        # GPT-2 风格权重初始化 (对齐 MSAmba)
        self.layers.apply(partial(_init_weights, n_layer=depth))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, return_cls: bool = False):
        """
        x: (B, L, D)   输入序列
        mask: (B, L) bool, True=valid, False=pad  (可选, 用于 padding-aware 训练)

        Args:
            mask: 序列有效位掩码。若提供, 在以下位置做 zero-out, 防止 pad 区污染 BiMamba 状态:
                  - 入口 (冗余保险, 因 classifier._encode 已 zero-out 过)
                  - 每个 Block_GLCE 之后 (BiMamba conv1d/SSM 会把 0 输入经过 bias 后变成非 0)
                  - 最终 hidden_states (在切 CLS / seq 之前)
                  CLS token 位置永远保持 valid (mask 拼接时左侧 pad True).
            return_cls: True 时返回 (seq_without_cls, cls_token); False 保持旧接口只返回 seq

        返回:
            - return_cls=False (默认, 向后兼容): (B, L, D)  去掉 CLS token 的序列 (已 zero-out pad)
            - return_cls=True: (seq, cls)
                seq: (B, L, D)   去掉 CLS token 的序列 (已 zero-out pad)
                cls: (B, D)      ISM 聚合后的 CLS token (用于 sub_loss / 跨模态引导)
        """
        B, L, _ = x.shape

        # ---- mask 前处理: 扩展 CLS 位 (CLS 永远 valid) ----
        if mask is not None:
            # mask: (B, L) -> mask_ext: (B, L+1)  CLS 位填 True
            cls_mask = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)
            mask_ext = torch.cat([cls_mask, mask], dim=1)        # (B, L+1)
            mask_ext_f = mask_ext.unsqueeze(-1).to(x.dtype)      # (B, L+1, 1)
            # 入口冗余 zero-out (classifier 层应已做过, 但 pos_embed 之前 x 是干净的)
            x = x * mask.unsqueeze(-1).to(x.dtype)
        else:
            mask_ext_f = None

        # 拼接 CLS token + 位置编码 (对齐 MSAmba)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)       # (B, L+1, D)
        x = x + self.pos_embed                       # (B, L+1, D)

        # 注意: pos_embed 加到 pad 位也会有非 0 值, 这里立刻 zero-out
        # (CLS 位 mask_ext 为 True, 不受影响)
        if mask_ext_f is not None:
            x = x * mask_ext_f

        # 双流残差传递 (对齐 MSAmba 的 forward 循环)
        residual = None
        hidden_states = x
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
            # 每层 BiMamba 之后立刻 zero-out pad 位
            # (conv1d 的 bias / SSM 的状态会让 0 输入变成非 0, 阻止其污染下层与跨模态)
            if mask_ext_f is not None:
                hidden_states = hidden_states * mask_ext_f
                if residual is not None:
                    residual = residual * mask_ext_f

        # 最终: hidden_states + residual (对齐 MSAmba 取 cls_token 前的处理)
        if residual is not None:
            hidden_states = hidden_states + residual

        # 最终 zero-out (双保险, 防 LayerNorm/RMSNorm 后 pad 区漂移)
        if mask_ext_f is not None:
            hidden_states = hidden_states * mask_ext_f

        if return_cls:
            cls = hidden_states[:, 0, :]            # (B, D)
            seq = hidden_states[:, 1:, :]           # (B, L, D)
            return seq, cls
        # 向后兼容: 只返回 seq
        return hidden_states[:, 1:, :]