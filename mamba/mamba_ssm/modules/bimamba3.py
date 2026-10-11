# Copyright (c) 2026.
# biMamba-3 (V3): 基于官方 Mamba-3 的 ViM 风格双向化
# =========================================================
# 设计参考:
#   - Mamba-3 : mamba_ssm/modules/mamba3.py (Dao AI Lab / Goombalab, 2026)
#   - Vim     : Vision Mamba, arXiv:2401.09417 (ICML 2024) —— "bimamba_type=v2"
#
# 关键点:
#   1. 反向分支通过 Python 层 x.flip([1]) 实现, 完全复用 Mamba-3 的
#      mamba3_siso_combined / mamba3_mimo_combined 融合 kernel, 无需改 CUDA 代码.
#   2. RoPE 方向性自动满足: 序列翻转后, kernel 看到的就是一个 "正向" 序列,
#      angles 也随 in_proj_b 重新生成 (data-dependent), 无需手工共轭.
#   3. ViM v2 等价: 独立两套 in_proj + BC 规范化; 共享 out_proj 与 MIMO 权重.
#   4. step() 禁用 (双向非因果), 仅支持 forward 训练/并行推理.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

try:
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo as mamba3_mimo_combined
except ImportError:
    mamba3_mimo_combined = None


__all__ = ["BiMamba3"]


class BiMamba3(nn.Module):
    """
    biMamba-3: 双向 Mamba-3 block (ViM v2 风格).

    Args (与官方 Mamba3 完全对齐, 额外增加 4 个 V3 专属参数):
        d_model, d_state, expand, headdim, ngroups,
        rope_fraction, dt_min, dt_max, dt_init_floor, A_floor,
        is_outproj_norm, is_mimo, mimo_rank, chunk_size,
        layer_idx, device, dtype
        --- 双向相关 ---
        bimamba_type:
            - "none": 单向 (等价 Mamba3), 仅供 ablation
            - "v2"  : ViM v2 风格, 独立两套 in_proj + BC 规范化, 共享 out_proj (默认)
        fusion:
            - "add"         : y = y_f + y_b.flip(1)                 (默认, 与 ViM 一致)
            - "add_divide2" : y = (y_f + y_b.flip(1)) / 2
            - "concat_proj" : y = W([y_f, y_b.flip(1)])              # 拼接后 Linear(2d_inner -> d_inner)
            - "gated"       : g = sigmoid(W_g[y_f, y_b.flip(1)]);
                              y = g * y_f + (1 - g) * y_b.flip(1)
        share_mimo: MIMO 模式下是否共享 mimo_x/z/o 权重 (默认 True, 推荐)
    Forward I/O 与 Mamba3 保持兼容:
        u: (B, L, D) -> out: (B, L, D)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        # ---- Mamba-3 configs ----
        rope_fraction: float = 0.5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        is_outproj_norm: bool = False,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        chunk_size: int = 64,
        # ---- 双向专属 ----
        bimamba_type: str = "v2",
        fusion: str = "add",
        share_mimo: bool = True,
        # ---- 其他 ----
        dropout: float = 0.0,
        layer_idx: Optional[int] = None,
        n_layer: Optional[int] = None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        assert bimamba_type in ("none", "v2"), f"Unsupported bimamba_type={bimamba_type}"
        assert fusion in ("add", "add_divide2", "concat_proj", "gated"), \
            f"Unsupported fusion={fusion}"

        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.A_floor = A_floor
        self.is_outproj_norm = is_outproj_norm
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank if is_mimo else 1
        if is_mimo:
            assert mamba3_mimo_combined is not None, \
                "Mamba-3 MIMO kernel not available. Install TileLang or set is_mimo=False."

        self.bimamba_type = bimamba_type
        self.fusion = fusion
        self.share_mimo = share_mimo

        self.d_inner = int(self.expand * self.d_model)
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.num_bc_heads = ngroups

        # RoPE
        assert rope_fraction in [0.5, 1.0]
        self.rotary_dim_divisor = int(2 / rope_fraction)
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        assert self.num_rope_angles > 0

        # ---- in_proj 总维度 (与 Mamba3 完全一致的切分: [z, x, B, C, dd_dt, dd_A, trap, angles]) ----
        d_in_proj = (
            2 * self.d_inner
            + 2 * self.d_state * self.num_bc_heads * self.mimo_rank
            + 3 * self.nheads
            + self.num_rope_angles
        )

        # ==========================================================
        # 正向分支 (与 Mamba3 完全一致)
        # ==========================================================
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)
        self._init_dt_bias(dt_min, dt_max, dt_init_floor, device, prefix="")
        self._init_bc_params(device, prefix="")
        self._init_rmsnorm_bc(factory_kwargs, prefix="")
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # ==========================================================
        # 反向分支 (bimamba_type=="v2" 时创建)
        # ==========================================================
        if bimamba_type == "v2":
            self.in_proj_b = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)
            self._init_dt_bias(dt_min, dt_max, dt_init_floor, device, prefix="_b")
            self._init_bc_params(device, prefix="_b")
            self._init_rmsnorm_bc(factory_kwargs, prefix="_b")
            self.D_b = nn.Parameter(torch.ones(self.nheads, device=device))
            self.D_b._no_weight_decay = True

        # ==========================================================
        # MIMO 权重 (共享 or 独立)
        # ==========================================================
        if self.is_mimo:
            self.mimo_x = nn.Parameter(
                torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device) / self.mimo_rank)
            self.mimo_z = nn.Parameter(
                torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device))
            self.mimo_o = nn.Parameter(
                torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device) / self.mimo_rank)
            if bimamba_type == "v2" and not share_mimo:
                self.mimo_x_b = nn.Parameter(
                    torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device) / self.mimo_rank)
                self.mimo_z_b = nn.Parameter(
                    torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device))
                self.mimo_o_b = nn.Parameter(
                    torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device) / self.mimo_rank)

        # ==========================================================
        # 输出归一化 (可选) + 输出投影 (共享)
        # ==========================================================
        if self.is_outproj_norm:
            self.norm = RMSNormGated(
                self.d_inner, eps=1e-5, norm_before_gate=True,
                group_size=self.headdim, **factory_kwargs,
            )

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)

        # ==========================================================
        # 融合层 (fusion=="concat_proj"/"gated" 时需要)
        # ==========================================================
        if bimamba_type == "v2":
            if fusion == "concat_proj":
                self.fuse_proj = nn.Linear(2 * self.d_inner, self.d_inner, bias=False, **factory_kwargs)
            elif fusion == "gated":
                self.fuse_gate = nn.Linear(2 * self.d_inner, self.d_inner, bias=True, **factory_kwargs)

    # -----------------------------------------------------------------
    # helpers: 初始化与 Mamba3 完全一致的 dt_bias / B_bias / C_bias / RMSNorm
    # -----------------------------------------------------------------
    def _init_dt_bias(self, dt_min, dt_max, dt_init_floor, device, prefix=""):
        _dt = torch.exp(
            torch.rand(self.nheads, device=device, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        _dt = torch.clamp(_dt, min=dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        p = nn.Parameter(_dt_bias, requires_grad=True)
        p._no_weight_decay = True
        setattr(self, f"dt_bias{prefix}", p)

    def _init_bc_params(self, device, prefix=""):
        B_bias = nn.Parameter(
            1 + torch.zeros((self.nheads, self.mimo_rank, self.d_state),
                            dtype=torch.float32, device=device), requires_grad=True)
        C_bias = nn.Parameter(
            1 + torch.zeros((self.nheads, self.mimo_rank, self.d_state),
                            dtype=torch.float32, device=device), requires_grad=True)
        setattr(self, f"B_bias{prefix}", B_bias)
        setattr(self, f"C_bias{prefix}", C_bias)

    def _init_rmsnorm_bc(self, factory_kwargs, prefix=""):
        setattr(self, f"B_norm{prefix}", RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs))
        setattr(self, f"C_norm{prefix}", RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs))

    # -----------------------------------------------------------------
    # 内部: 单方向 SSM 扫描 (封装 Mamba3 原 forward 的 pre-kernel → kernel → post-kernel 流程)
    # -----------------------------------------------------------------
    def _scan_one_direction(self, u, *, in_proj, dt_bias, B_bias, C_bias,
                            B_norm, C_norm, D, mimo_x, mimo_z, mimo_o):
        """单方向前向, 返回未过 out_proj 的 y (B, L, d_inner)."""
        zxBCdtAtrap = in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [
                self.d_inner, self.d_inner,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.nheads, self.nheads, self.nheads,
                self.num_rope_angles,
            ], dim=-1)

        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")

        _A = -F.softplus(dd_A.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        # DT/ADT 必须 fp32 (kernel 要求 T.float32)
        DT = F.softplus(dd_dt.to(torch.float32) + dt_bias.to(torch.float32))
        ADT = _A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")

        # Angles 必须 fp32 (kernel 要求 T.float32)
        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32)
        # trap 必须保持与激活同 dtype (kernel: TRAP 类型为 dtype, 而非 fp32)

        B = B_norm(B)
        C = C_norm(C)

        if self.is_mimo:
            y = mamba3_mimo_combined(
                Q=C, K=B, V=x, ADT=ADT, DT=DT, Trap=trap,
                # MIMO TileLang kernel 严格要求以下参数为 fp32, 当 module .to(fp16/bf16)
                # 后 nn.Parameter 会被一起 cast, 这里显式还原为 fp32
                Q_bias=C_bias.to(torch.float32),
                K_bias=B_bias.to(torch.float32),
                MIMO_V=mimo_x.to(torch.float32) if mimo_x is not None else None,
                MIMO_Z=mimo_z.to(torch.float32) if mimo_z is not None else None,
                MIMO_Out=(mimo_o.to(torch.float32) if (mimo_o is not None and not self.is_outproj_norm) else None),
                Angles=angles, D=D.to(torch.float32),
                Z=z if not self.is_outproj_norm else None,
                chunk_size=self.chunk_size,
                rotary_dim_divisor=self.rotary_dim_divisor,
                dtype=x.dtype, return_state=False, cu_seqlens=None,
            )
            if self.is_outproj_norm:
                z_r = torch.einsum("blhp,hrp->blrhp", z.float(), mimo_z)
                z_r = rearrange(z_r, "b l r h p -> b l r (h p)")
                y = rearrange(y, "b l r h p -> b l r (h p)").float()
                y = self.norm(y, z_r)
                y = rearrange(y, "b l r (h p) -> b l r h p", p=self.headdim)
                y = torch.einsum("blrhp,hrp->blhp", y, mimo_o)
            y = rearrange(y, "b l h p -> b l (h p)")
        else:
            y = mamba3_siso_combined(
                Q=C.squeeze(2), K=B.squeeze(2), V=x,
                ADT=ADT, DT=DT, Trap=trap,
                # 与 MIMO 一致, kernel 期望 bias/D 为 fp32
                Q_bias=C_bias.squeeze(1).to(torch.float32),
                K_bias=B_bias.squeeze(1).to(torch.float32),
                Angles=angles, D=D.to(torch.float32),
                Z=z if not self.is_outproj_norm else None,
                chunk_size=self.chunk_size,
                Input_States=None, return_final_states=False, cu_seqlens=None,
            )
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.is_outproj_norm:
                z_flat = rearrange(z, "b l h p -> b l (h p)")
                y = self.norm(y, z_flat)
        return y

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------
    def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
        """u: (B, L, D)  ->  out: (B, L, D)"""
        assert cu_seqlens is None, "BiMamba3 does not support variable-length packed batch (cu_seqlens)."
        if inference_params is not None:
            assert inference_params.seqlen_offset == 0, \
                "BiMamba3 does not support incremental decoding (step). Bidirectional is non-causal."

        # ---- 正向 ----
        y_f = self._scan_one_direction(
            u,
            in_proj=self.in_proj, dt_bias=self.dt_bias,
            B_bias=self.B_bias, C_bias=self.C_bias,
            B_norm=self.B_norm, C_norm=self.C_norm, D=self.D,
            mimo_x=getattr(self, "mimo_x", None),
            mimo_z=getattr(self, "mimo_z", None),
            mimo_o=getattr(self, "mimo_o", None),
        )  # (B, L, d_inner)

        # ---- 单向模式 (ablation) ----
        if self.bimamba_type == "none":
            return self.out_proj(y_f.to(self.out_proj.weight.dtype))

        # ---- 反向: 翻序列 → 扫描 → 翻回 ----
        u_b = u.flip(dims=[1])
        y_b = self._scan_one_direction(
            u_b,
            in_proj=self.in_proj_b, dt_bias=self.dt_bias_b,
            B_bias=self.B_bias_b, C_bias=self.C_bias_b,
            B_norm=self.B_norm_b, C_norm=self.C_norm_b, D=self.D_b,
            mimo_x=(self.mimo_x if (self.is_mimo and self.share_mimo) else getattr(self, "mimo_x_b", None)),
            mimo_z=(self.mimo_z if (self.is_mimo and self.share_mimo) else getattr(self, "mimo_z_b", None)),
            mimo_o=(self.mimo_o if (self.is_mimo and self.share_mimo) else getattr(self, "mimo_o_b", None)),
        )
        y_b = y_b.flip(dims=[1])  # 翻回原序, 每个 t 的表征对应原位置

        # ---- 融合 ----
        # 注意: Mamba-3 kernel 内部可能把输出 cast 为 bf16, 而 fuse_proj/fuse_gate 的 weight
        # 跟随 module dtype (可能是 fp16), 因此对 concat_proj/gated 分支需手动统一 dtype.
        if self.fusion == "add":
            y = y_f + y_b
        elif self.fusion == "add_divide2":
            y = (y_f + y_b) * 0.5
        elif self.fusion == "concat_proj":
            target_dtype = self.fuse_proj.weight.dtype
            y = self.fuse_proj(torch.cat([y_f.to(target_dtype), y_b.to(target_dtype)], dim=-1))
        elif self.fusion == "gated":
            target_dtype = self.fuse_gate.weight.dtype
            g = torch.sigmoid(self.fuse_gate(torch.cat([y_f.to(target_dtype), y_b.to(target_dtype)], dim=-1)))
            y = g * y_f.to(target_dtype) + (1.0 - g) * y_b.to(target_dtype)
        else:
            raise ValueError(self.fusion)

        out = self.out_proj(y.to(self.out_proj.weight.dtype))
        return out

    # -----------------------------------------------------------------
    # 非因果: 不支持增量解码 cache
    # -----------------------------------------------------------------
    def allocate_inference_cache(self, *args, **kwargs):
        raise NotImplementedError(
            "BiMamba3 is non-causal (bidirectional) and does not support incremental decoding."
        )