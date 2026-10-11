"""
UMRM v3: 强基线（BERT 微调 + ISM 单模态建模 + 缺失鲁棒融合）
=============================================================

与 v2 的差异（性能来源）：
1. 文本走 BERT 微调编码（H-DCD 对齐，text_bert 三通道输入），而非冻结 768 特征
2. 每个模态先过 ISMEncoder（GLCE + BSSM 单模态序列建模，H-DCD 移植）
3. 保留 v2 的全部缺失机制：统一缺失编码 + 缺失感知增强 + CoupledMamba3Fork
   并行拼接 + 多粒度对比重建 + 课程调度

架构：
    文本(BERT) + 音频 + 视觉
        ↓ 投影 + 序列对齐
        ↓ ISMEncoder × 3（单模态建模）
        ↓ 缺失模拟（模态级+Token级）
        ↓ 统一缺失编码 + 缺失感知增强
        ↓ CoupledMamba3Fork（核心跨模态融合）
        ↓ 并行拼接 [enhanced, fused] → 池化 → 分类头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .coupled_mamba3_fork import CoupledMamba3Fork
from .text_encoder import TextPretrainedEncoder
from .layers.ism import ISMEncoder
from .umrm_v2 import (
    UnifiedMissingEncoder,
    MissingAwareFeatureEnhancer,
    MultiGranularityContrastiveReconstruction,
    CurriculumMissingScheduler,
)


class TextReconstructionLoss(nn.Module):
    """
    TF-Mamba 式文本重建损失（论文式12）

    Lrec = SmoothL1((Xt - R(Ht)) ⊙ (1 - Pt))

    - Xt: 原始文本特征 (B, L, text_dim)，作为重建目标（BERT 干净输出）
    - Ht: 缺失污染后的文本表示 (B, L, d_model)（经 ISM/缺失编码/增强）
    - R: 两层 Linear + ReLU 重建器（轻量解码头）
    - 只在 Token 缺失位置（且文本模态可用）计算损失，
      迫使模型从跨模态 SSM 上下文中恢复缺失文本语义。
    """

    def __init__(self, d_model: int, text_dim: int = 768):
        super().__init__()
        self.reconstructor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, text_dim),
        )

    def forward(self, Ht, Xt, token_mask_text, modality_mask_text):
        """
        Args:
            Ht: (B, L, d_model) 缺失污染后的文本表示
            Xt: (B, L, text_dim) 干净文本特征（重建目标）
            token_mask_text: (B, L) 1=保留, 0=缺失
            modality_mask_text: (B,) 文本模态可用性
        Returns:
            scalar loss（按缺失位置与特征维归一化，O(1) 量级）
        """
        B, L, D = Ht.shape
        # 只重建"文本模态可用"下的 token 缺失位置
        missing = (1.0 - token_mask_text) * modality_mask_text.unsqueeze(-1)  # (B, L)
        missing = missing.unsqueeze(-1)  # (B, L, 1)

        recon = self.reconstructor(Ht)  # (B, L, text_dim)
        err = F.smooth_l1_loss(recon, Xt, reduction='none')  # (B, L, text_dim)
        denom = missing.sum() * Xt.size(-1)
        if denom < 1:
            return torch.zeros((), device=Ht.device)
        return (err * missing).sum() / denom


class TextSimilarityInjection(nn.Module):
    """
    TF-Mamba 式相似度注入回路（论文式10-11）

    E_v = H_v + (M^{vt} ⊙ S^{vt}) · H_t
    - S^{vt}_{i,j} = exp(⟨v_i, t_j⟩/τ) / Σ_k exp(⟨v_i, t_k⟩/τ)  对文本 softmax
    - M = I[S > θ], θ = 1/L 硬阈值滤噪
    将高相似度的文本语义注入非文本模态，与重建损失形成闭环。
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, H_nontext, H_text, text_avail_mask):
        """
        Args:
            H_nontext: (B, L, D) 音/视觉表示
            H_text: (B, L, D) 文本表示（可能含缺失）
            text_avail_mask: (B,) 文本模态可用性
        Returns:
            注入后的 (B, L, D)
        """
        B, L, D = H_nontext.shape
        v = F.normalize(H_nontext, dim=-1)
        t = F.normalize(H_text, dim=-1)

        # 温度缩放相似度 + 对文本维度 softmax
        S = torch.bmm(v, t.transpose(1, 2)) / self.temperature  # (B, L, L)
        S = F.softmax(S, dim=-1)

        # 硬阈值滤噪：θ = 1/L（TF-Mamba 式）
        theta = 1.0 / max(L, 1)
        M = (S > theta).float()

        injected = torch.bmm(M * S, H_text)  # (B, L, D)
        # 文本模态缺失的样本跳过注入（防止零向量噪声）
        gate = text_avail_mask.float().unsqueeze(-1).unsqueeze(-1)
        return H_nontext + injected * gate


class UMRMv3(nn.Module):
    """BERT + ISM 强基线上的模态缺失鲁棒模型"""

    def __init__(self, args):
        super().__init__()
        self.args = args

        # ---- 基本配置 ----
        d_model = getattr(args, 'd_model', 128)
        num_layers = getattr(args, 'num_layers', 2)
        d_state = getattr(args, 'd_state', 64)
        expand = getattr(args, 'expand', 2)
        headdim = getattr(args, 'headdim', 32)
        ngroups = getattr(args, 'ngroups', 1)
        rope_fraction = getattr(args, 'rope_fraction', 0.5)
        is_mimo = getattr(args, 'is_mimo', False)
        mimo_rank = getattr(args, 'mimo_rank', 4)
        chunk_size = getattr(args, 'chunk_size', 64)
        is_outproj_norm = getattr(args, 'is_outproj_norm', False)
        v_self_ratio = getattr(args, 'v_self_ratio', 0.0)
        use_gcmn_gate = getattr(args, 'use_gcmn_gate', False)

        feature_dims = getattr(args, 'feature_dims', [768, 74, 20])

        # ---- 1) 文本编码：BERT 微调（可选） ----
        self.use_bert = getattr(args, 'use_bert', True)
        if self.use_bert:
            self.bert_encoder = TextPretrainedEncoder(
                pretrained=getattr(args, 'bert_pretrained', 'bert-base-uncased'),
                finetune=getattr(args, 'bert_finetune', True),
                pretokenized=getattr(args, 'bert_pretokenized', False),
                lora_r=getattr(args, 'lora_r', 8),
                lora_alpha=getattr(args, 'lora_alpha', 16),
            )
            text_feat_dim = self.bert_encoder.out_dim
        else:
            self.bert_encoder = None
            text_feat_dim = feature_dims[0]

        # ---- 2) 输入投影 ----
        self.proj_text = nn.Linear(text_feat_dim, d_model)
        self.proj_audio = nn.Linear(feature_dims[1], d_model)
        self.proj_video = nn.Linear(feature_dims[2], d_model)

        # ---- 3) ISM 单模态序列建模（H-DCD 对齐） ----
        self.ism_depth = getattr(args, 'ism_depth', 3)
        if self.ism_depth > 0:
            ism_kwargs = dict(
                d_model=d_model,
                seq_len=int(getattr(args, 'seq_lens', [50, 50, 50])[0]),
                depth=self.ism_depth,
                d_state=getattr(args, 'ism_d_state', 32),
                d_conv=4, expand=2,
                dropout=getattr(args, 'dropout', 0.5),
                mixer_type=getattr(args, 'ism_mixer_type', 'bimamba'),
                # ★ BSSM 门控 (CAGMamba 公式 15-16, 与 H-DCD 最优配置同款)
                use_bssm_gate=getattr(args, 'use_bssm_gate', False),
                bssm_gate_expand=getattr(args, 'bssm_gate_expand', 2),
            )
            self.ism_text = ISMEncoder(**ism_kwargs)
            self.ism_audio = ISMEncoder(**ism_kwargs)
            self.ism_video = ISMEncoder(**ism_kwargs)
        else:
            self.ism_text = self.ism_audio = self.ism_video = None

        # ---- 4) 缺失机制（与 v2 一致） ----
        self.missing_encoder = UnifiedMissingEncoder(d_model)
        self.missing_enhancers = nn.ModuleList([
            MissingAwareFeatureEnhancer(d_model=d_model) for _ in range(3)
        ])
        self.contrastive_reconstruction = MultiGranularityContrastiveReconstruction(d_model)
        self.curriculum_scheduler = CurriculumMissingScheduler(
            total_epochs=getattr(args, 'epochs', 100),
            max_token_rate=getattr(args, 'token_missing_rate_max', 0.6),
            modality_rate=getattr(args, 'modality_missing_rate', 0.3),
        )
        self.token_missing_rate = getattr(args, 'token_missing_rate', 0.3)
        self.modality_missing_rate = getattr(args, 'modality_missing_rate', 0.3)

        # ---- 4b) TF-Mamba 式文本重建损失（可选，方案①） ----
        self.use_text_recon = getattr(args, 'use_text_recon', False)
        if self.use_text_recon:
            self.text_recon = TextReconstructionLoss(d_model, text_feat_dim)
        else:
            self.text_recon = None

        # ---- 4c) TF-Mamba 式相似度注入回路（可选，方案②） ----
        self.use_text_injection = getattr(args, 'use_text_injection', False)
        if self.use_text_injection:
            self.inj_audio = TextSimilarityInjection(
                temperature=getattr(args, 'inj_temperature', 0.1))
            self.inj_vision = TextSimilarityInjection(
                temperature=getattr(args, 'inj_temperature', 0.1))
        else:
            self.inj_audio = self.inj_vision = None

        # ---- 5) 核心融合：CoupledMamba3Fork × num_layers 堆叠 ----
        # 对齐 H-DCD：多层 Fork 堆叠（每层内置残差+LN），缺失机制只作用于第一层入口
        self.num_fusion_layers = max(getattr(args, 'num_layers', 2), 1)
        self.fusion_layers = nn.ModuleList([
            CoupledMamba3Fork(
                d_model=d_model, d_state=d_state, expand=expand, headdim=headdim,
                ngroups=ngroups, rope_fraction=rope_fraction,
                is_outproj_norm=is_outproj_norm, is_mimo=is_mimo, mimo_rank=mimo_rank,
                chunk_size=chunk_size, v_self_ratio=v_self_ratio,
                use_gcmn_gate=use_gcmn_gate,
            )
            for _ in range(self.num_fusion_layers)
        ])
        # ---- 5b) 双向支路（VIM 思想转译，不改变 Fork 核心） ----
        # 反向扫描：flip 输入 → 第二个 Fork（独立参数） → flip 回相加
        self.use_bi_direction = getattr(args, 'use_bi_direction', False)
        if self.use_bi_direction:
            self.core_fusion_rev = CoupledMamba3Fork(
                d_model=d_model, d_state=d_state, expand=expand, headdim=headdim,
                ngroups=ngroups, rope_fraction=rope_fraction,
                is_outproj_norm=is_outproj_norm, is_mimo=is_mimo, mimo_rank=mimo_rank,
                chunk_size=chunk_size, v_self_ratio=v_self_ratio,
                use_gcmn_gate=use_gcmn_gate,
            )
        else:
            self.core_fusion_rev = None
        # ---- 6) 输出层（并行拼接 d_model*2*3） ----
        self.pool_type = getattr(args, 'pool_type', 'mean')
        dropout = getattr(args, 'dropout', 0.5)
        num_classes = getattr(args, 'num_classes', 1)
        self.fusion_norm = nn.LayerNorm(d_model * 2 * 3)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2 * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # ---- 6b) 7分类辅助头（aux-cls 直接优化 Acc7 / RCD 解耦蒸馏共用） ----
        self.use_aux_cls = getattr(args, 'use_aux_cls', False)
        self.use_rcd = getattr(args, 'use_rcd', False)
        if self.use_aux_cls or self.use_rcd:
            self.aux_cls_head = nn.Sequential(
                nn.Linear(d_model * 2 * 3, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 7),
            )
        else:
            self.aux_cls_head = None

        # ---- 6b2) 2分类辅助头（RCD2 二分类解耦蒸馏，Tier1） ----
        self.use_rcd2 = getattr(args, 'use_rcd2', False)
        if self.use_rcd2:
            self.rcd2_head = nn.Sequential(
                nn.Linear(d_model * 2 * 3, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 2),
            )
        else:
            self.rcd2_head = None

        # ---- 6c) UAF σ² 不确定性门控（MRUF 式逆方差调制，Tier1-1） ----
        # 输入：各支路池化特征 (B,d_model) + 缺失统计 (B,6: 模态可用性+token可用率)
        # 输出 log σ² → 归一化逆方差权重 ω 调制三支路进头特征
        self.use_uaf = getattr(args, 'use_uaf', False)
        if self.use_uaf:
            self.uaf_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model + 6, max(d_model // 2, 32)),
                    nn.GELU(),
                    nn.Linear(max(d_model // 2, 32), 1),
                ) for _ in range(3)
            ])
        else:
            self.uaf_heads = None

        # ---- 6d) 单模态预测头 + 能量平衡（EBMC 式，Tier1-2） ----
        # 每个模态的独立回归头，保证降级支路自身可预测；能量差正则在外层计算
        self.use_uni = getattr(args, 'use_uni', False)
        if self.use_uni:
            self.uni_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(3)])
        else:
            self.uni_heads = None

        # ---- 6e) GSDNet 式缺失 A/V 池化级生成重建（Tier2-15） ----
        # 缺失音频/视频时，用文本支路池化特征生成其融合池化表示的估计
        self.use_av_recon = getattr(args, 'use_av_recon', False)
        if self.use_av_recon:
            self.rec_audio = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(),
                nn.Linear(d_model, d_model * 2),
            )
            self.rec_vision = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(),
                nn.Linear(d_model, d_model * 2),
            )
        else:
            self.rec_audio = self.rec_vision = None

        # ---- 6f) 硬缺失掩码池化：缺失支路池化清零（SIEVE “repair 可能有害”观察） ----
        self.use_hard_mask_pool = getattr(args, 'use_hard_mask_pool', False)
        # ---- 6g) 组合均衡缺失采样：7 模式等权（对齐测试六组合协议） ----
        self.use_balanced_missing = getattr(args, 'use_balanced_missing', False)
        # ---- 6h) 缺失掩码条件化 head：modality_mask 拼接进 head 输入，使 head 感知组合模式 ----
        self.use_mask_cond_head = getattr(args, 'use_mask_cond_head', False)
        if self.use_mask_cond_head:
            self.head_cond = nn.Sequential(
                nn.Linear(d_model * 2 * 3 + 3, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )
        else:
            self.head_cond = None

        # ---- 6i) 组合级可学习偏置：校准 A/V/AV 等稀疏组合的恒定输出偏置 ----
        # 诊断：A/V/AV 组合 logits 恒正类（std≈0.01）→ 用 7 个标量参数校准
        self.use_combo_bias = getattr(args, 'use_combo_bias', False)
        self.use_combo_heads = getattr(args, 'use_combo_heads', False)
        if self.use_combo_bias or self.use_combo_heads:
            self.register_buffer('_combo_w', torch.tensor([4., 2., 1.]))
        if self.use_combo_bias:
            self.combo_bias = nn.Parameter(torch.zeros(7))
        else:
            self.combo_bias = None

        # ---- 6j) 7 组合独立 head（DVS 式，与六组合评测协议完全匹配） ----
        self.use_combo_heads = getattr(args, 'use_combo_heads', False)
        if self.use_combo_heads:
            self.combo_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model * 2 * 3, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, num_classes),
                ) for _ in range(7)
            ])
        else:
            self.combo_heads = None

    def _pool(self, x, mask=None):
        if self.pool_type == 'mean':
            if mask is not None:
                x = x * mask.unsqueeze(-1)
                return x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            return x.mean(dim=1)
        elif self.pool_type == 'last':
            return x[:, -1, :]
        elif self.pool_type == 'cls':
            return x[:, 0, :]
        return x.mean(dim=1)

    def _align_seq_length(self, x, target_len):
        if x.size(1) == target_len:
            return x
        return F.adaptive_avg_pool1d(x.transpose(1, 2), target_len).transpose(1, 2)

    def _generate_token_mask(self, B, L, missing_rate, device):
        if missing_rate <= 0:
            return torch.ones(B, L, device=device)
        mask = torch.rand(B, L, device=device) >= missing_rate
        mask[:, 0] = 1.0
        mask[:, -1] = 1.0
        return mask.float()

    def _generate_modality_mask(self, B, missing_rate, device):
        if missing_rate <= 0:
            return torch.ones(B, 3, device=device)
        # 组合均衡采样：7 模式等权（TAV/TA/TV/AV/T/A/V），对齐测试六组合协议
        # 解决“单模态训练样本仅 5% vs 测试 16.7%”的分布失配
        if getattr(self, 'use_balanced_missing', False):
            patterns = torch.tensor([
                [1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
                [1, 0, 0], [0, 1, 0], [0, 0, 1],
            ], device=device).float()
            idx = torch.randint(0, patterns.size(0), (B,), device=device)
            return patterns[idx]
        mask = torch.ones(B, 3, device=device)
        for b in range(B):
            if torch.rand(1) < missing_rate:
                num_missing = torch.randint(1, 3, (1,)).item()
                missing_idx = torch.randperm(3)[:num_missing]
                mask[b, missing_idx] = 0.0
        return mask

    def forward(self, text, audio, vision,
                audio_lengths=None, vision_lengths=None,
                cu_seqlens=None,
                token_mask=None, modality_mask=None,
                return_contrastive_loss=False,
                return_fused=False,
                return_pre_pools=False,
                return_text_target=False,
                text_reconstruction_target=None,
                return_reconstruction_targets=False,
                reconstruction_targets=None):
        device = text.device
        B = text.size(0)

        # ---- 1) 文本 BERT 编码 ----
        text_attention = None
        if self.use_bert and self.bert_encoder is not None:
            # text: (B, 3, L) 三通道 [ids, mask, segment] 或 (B, L, 768) 特征
            if text.dim() == 3 and text.size(1) == 3:
                text_feat, text_attention = self.bert_encoder(text)  # (B, L, 768)
            else:
                text_feat = text
        else:
            text_feat = text        # 干净文本特征作为重建目标（未受缺失污染）
        text_target = text_feat.detach() if (self.use_text_recon and self.training) else None
        # Input-corrupted students must reconstruct the *clean teacher* encoding.
        # The target is used only by the loss below, never by the forward features.
        if text_reconstruction_target is not None:
            if text_reconstruction_target.shape != text_feat.shape:
                raise ValueError('Clean text reconstruction target has incompatible shape')
            text_target = text_reconstruction_target.detach()
        # ---- 2) 投影 ----
        text_h = self.proj_text(text_feat)
        audio_h = self.proj_audio(audio)
        vision_h = self.proj_video(vision)

        # ---- 3) 序列对齐（对齐到文本长度） ----
        Lt = text_h.size(1)
        if audio_h.size(1) != Lt:
            audio_h = self._align_seq_length(audio_h, Lt)
        if vision_h.size(1) != Lt:
            vision_h = self._align_seq_length(vision_h, Lt)

        # ---- 4) ISM 单模态建模 ----
        if self.ism_text is not None:
            text_h = self.ism_text(text_h)
            audio_h = self.ism_audio(audio_h)
            vision_h = self.ism_video(vision_h)

        # ---- 4b) 无缺失污染的 A/V 池化目标（GSDNet 式重建监督用） ----
        if self.use_av_recon and self.training:
            pa_clean = self._pool(audio_h)
            pv_clean = self._pool(vision_h)
        else:
            pa_clean = pv_clean = None

        # ---- 5) 缺失 mask 生成 ----
        if token_mask is None:
            if not self.training:
                token_mask = torch.ones(B, 3, Lt, device=device)
            else:
                token_mask = torch.stack([
                    self._generate_token_mask(B, Lt, self.token_missing_rate, device)
                    for _ in range(3)
                ], dim=1)  # (B, 3, L)
                # Feature-level dropout excludes padding and preserves actual
                # valid endpoints. Explicit evaluation masks remain authoritative.
                from utils.token_protocol import validity, align_validity
                valid = align_validity(validity(text, audio, vision, audio_lengths, vision_lengths), Lt)
                if text_attention is not None:
                    valid[:, 0] = text_attention.bool()
                token_mask = torch.where(valid, token_mask, torch.ones_like(token_mask))
                indices = torch.arange(Lt, device=device).expand_as(valid)
                first = indices.masked_fill(~valid, Lt).amin(dim=-1)
                last = indices.masked_fill(~valid, -1).amax(dim=-1)
                endpoints = (indices == first[..., None]) | (indices == last[..., None])
                token_mask = torch.where(endpoints, torch.ones_like(token_mask), token_mask)

        if modality_mask is None:
            if not self.training:
                modality_mask = torch.ones(B, 3, device=device)
            else:
                modality_mask = self._generate_modality_mask(B, self.modality_missing_rate, device)

        # ---- 6) 应用缺失 mask ----
        text_h = text_h * token_mask[:, 0, :].unsqueeze(-1) * modality_mask[:, 0].unsqueeze(-1).unsqueeze(-1)
        audio_h = audio_h * token_mask[:, 1, :].unsqueeze(-1) * modality_mask[:, 1].unsqueeze(-1).unsqueeze(-1)
        vision_h = vision_h * token_mask[:, 2, :].unsqueeze(-1) * modality_mask[:, 2].unsqueeze(-1).unsqueeze(-1)

        # ---- 7) 统一缺失编码 + 缺失感知增强 ----
        text_enc, mi_t = self.missing_encoder(text_h, token_mask[:, 0], modality_mask[:, 0])
        audio_enc, mi_a = self.missing_encoder(audio_h, token_mask[:, 1], modality_mask[:, 1])
        vision_enc, mi_v = self.missing_encoder(vision_h, token_mask[:, 2], modality_mask[:, 2])

        text_enh = self.missing_enhancers[0](text_enc, mi_t)
        audio_enh = self.missing_enhancers[1](audio_enc, mi_a)
        vision_enh = self.missing_enhancers[2](vision_enc, mi_v)

        # ---- 7b) 方案②：文本语义相似度注入音/视觉（TME 回路） ----
        if self.inj_audio is not None:
            audio_enh = self.inj_audio(audio_enh, text_enh, modality_mask[:, 0])
            vision_enh = self.inj_vision(vision_enh, text_enh, modality_mask[:, 0])

        # ---- 7c) Tier1：预融合支路池化 + 缺失统计（UAF/单模态头/增强器级蒸馏共用） ----
        pre_pools = None
        miss_stat = None
        if self.use_uaf or self.use_uni or self.use_av_recon or return_pre_pools:
            pl_pre = self._pool(text_enh)
            pa_pre = self._pool(audio_enh)
            pv_pre = self._pool(vision_enh)
            pre_pools = (pl_pre, pa_pre, pv_pre)
            token_avail = token_mask.mean(dim=2)  # (B, 3) 每模态 token 可用率
            miss_stat = torch.cat([modality_mask, token_avail], dim=-1)  # (B, 6)

        # ---- 8) 核心融合：多层 Fork 堆叠 ----
        out_a, out_v, out_l = audio_enh, vision_enh, text_enh
        for layer in self.fusion_layers:
            out_a, out_v, out_l = layer(out_a, out_v, out_l, cu_seqlens=cu_seqlens)

        # 双向支路：反向扫描 + 相加（VIM 式，不改变 Fork 核心语义）
        if self.core_fusion_rev is not None:
            fa, fv, fl = audio_enh.flip(dims=[1]), vision_enh.flip(dims=[1]), text_enh.flip(dims=[1])
            out_a_b, out_v_b, out_l_b = self.core_fusion_rev(fa, fv, fl, cu_seqlens=cu_seqlens)
            out_a = out_a + out_a_b.flip(dims=[1])
            out_v = out_v + out_v_b.flip(dims=[1])
            out_l = out_l + out_l_b.flip(dims=[1])

        # ---- 9) 并行拼接 + 池化 + 分类头 ----
        text_fused = torch.cat([text_enh, out_l], dim=-1)
        audio_fused = torch.cat([audio_enh, out_a], dim=-1)
        vision_fused = torch.cat([vision_enh, out_v], dim=-1)
        pl, pa, pv = self._pool(text_fused), self._pool(audio_fused), self._pool(vision_fused)

        # ---- 9a-1) 硬缺失掩码池化：缺失支路池化清零，防止垃圾特征干扰 head ----
        if self.use_hard_mask_pool:
            mm = modality_mask.unsqueeze(-1)  # (B,3,1)
            pl = pl * mm[:, 0]
            pa = pa * mm[:, 1]
            pv = pv * mm[:, 2]

        # ---- 9a0) GSDNet 式：缺失 A/V 池化由文本支路生成替代（池化级重建） ----
        av_rec_loss_val = None
        if self.use_av_recon and pre_pools is not None:
            pa_rec = self.rec_audio(pre_pools[0])   # (B, 2d)
            pv_rec = self.rec_vision(pre_pools[0])
            ma = modality_mask[:, 1].unsqueeze(-1)
            mv = modality_mask[:, 2].unsqueeze(-1)
            pa = ma * pa + (1 - ma) * pa_rec
            pv = mv * pv + (1 - mv) * pv_rec
            if self.training and pa_clean is not None:
                target_a = torch.cat([pa_clean, pa_clean], dim=-1).detach()
                target_v = torch.cat([pv_clean, pv_clean], dim=-1).detach()
                la = ((1 - ma) * F.mse_loss(pa_rec, target_a, reduction='none').mean(dim=-1, keepdim=True)).sum()
                lv = ((1 - mv) * F.mse_loss(pv_rec, target_v, reduction='none').mean(dim=-1, keepdim=True)).sum()
                av_rec_loss_val = (la + lv) / ((1 - ma).sum() + (1 - mv).sum()).clamp(min=1)

        # ---- 9b) UAF σ² 逆方差门控：调制三支路进头特征 ----
        # v2：仅对"有模态缺失"的样本调制（完整样本保持原始信息流，避免伤害 TAV）
        if self.use_uaf and pre_pools is not None:
            sigma_logs = [self.uaf_heads[i](torch.cat([pre_pools[i], miss_stat], dim=-1))
                          for i in range(3)]
            var = torch.stack([torch.exp(s) for s in sigma_logs], dim=1)  # (B,3,1)
            w = 1.0 / (var + 1e-4)
            w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            full_avail = (modality_mask.min(dim=1).values > 0.5).float().unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
            w = full_avail + (1 - full_avail) * w
            pl = pl * w[:, 0]
            pa = pa * w[:, 1]
            pv = pv * w[:, 2]

        fused = self.fusion_norm(torch.cat([pl, pa, pv], dim=-1))
        # 缺失掩码条件化 head：拼接可用性掩码（B,3），使 head 感知当前组合模式
        if self.head_cond is not None:
            fused = torch.cat([fused, modality_mask], dim=-1)
            logits = self.head_cond(fused)
        else:
            logits = self.head(fused)
        # 组合级偏置校准：cid = miss(T)*4+miss(A)*2+miss(V)*1 ∈ [0,6]，每种模式一个标量偏置
        if self.combo_bias is not None:
            cid = ((~modality_mask.bool()).float() * self._combo_w).sum(dim=-1).long()
            logits = logits + self.combo_bias[cid].unsqueeze(-1)
        # 7 组合独立 head：按当前组合模式选择对应 head 输出
        if self.combo_heads is not None:
            cid = ((~modality_mask.bool()).float() * self._combo_w).sum(dim=-1).long()
            all_logits = torch.stack([h(fused) for h in self.combo_heads], dim=1)  # (B,7,1)
            logits = all_logits.gather(1, cid.view(-1, 1, 1).expand(-1, 1, all_logits.size(-1))).squeeze(1)

        result = {'logits': logits, 'token_mask': token_mask, 'modality_mask': modality_mask}
        if return_text_target:
            result['text_target'] = text_feat.detach()
        if return_reconstruction_targets:
            result['reconstruction_targets'] = [h.detach() for h in (text_enh, audio_enh, vision_enh)]
        if av_rec_loss_val is not None:
            result['av_recon_loss'] = av_rec_loss_val

        # ---- Tier1: 返回 Fork 前增强特征（CI-Der 式增强器级蒸馏用） ----
        if return_pre_pools and pre_pools is not None:
            result['pre_pools'] = pre_pools

        # ---- Tier1: 2分类 logits（RCD2 解耦蒸馏用） ----
        if self.rcd2_head is not None:
            result['rcd2_logits'] = self.rcd2_head(fused)

        # ---- 9c) Tier1：单模态预测 logits（EBMC 式，损失在外层计算） ----
        if self.use_uni and self.uni_heads is not None and pre_pools is not None:
            result['uni_logits'] = {
                'T': self.uni_heads[0](pre_pools[0]),
                'A': self.uni_heads[1](pre_pools[1]),
                'V': self.uni_heads[2](pre_pools[2]),
            }

        # 融合特征（特征层蒸馏用）
        if return_fused:
            result['fused'] = fused

        # 7 分类辅助 logits（训练时计算，直接优化 Acc7）
        if self.aux_cls_head is not None:
            result['aux_logits'] = self.aux_cls_head(fused)

        if return_contrastive_loss and self.training:
            targets = [text_enh, audio_enh, vision_enh]
            if reconstruction_targets is not None:
                if len(reconstruction_targets) != 3 or any(x.shape != y.shape for x,y in zip(reconstruction_targets, targets)):
                    raise ValueError('Clean reconstruction targets have incompatible shapes')
                targets = [h.detach() for h in reconstruction_targets]
            contrastive_loss = self.contrastive_reconstruction(
                out_l, targets, token_mask, modality_mask)
            result['contrastive_loss'] = contrastive_loss

            # 方案①：文本重建损失（TF-Mamba 式 12）
            if self.text_recon is not None and text_target is not None:
                result['text_recon_loss'] = self.text_recon(
                    text_enh, text_target, token_mask[:, 0], modality_mask[:, 0])

        return result

    def inference_with_missing(self, text, audio, vision, available_modalities):
        batch_size = text.size(0) if text is not None else (
            audio.size(0) if audio is not None else vision.size(0))
        device = text.device if text is not None else (
            audio.device if audio is not None else vision.device)
        seq_len = int(getattr(self.args, 'seq_lens', [50, 50, 50])[0])
        feature_dims = getattr(self.args, 'feature_dims', [768, 74, 20])

        modality_mask = torch.zeros(batch_size, 3, dtype=torch.bool, device=device)
        for m in available_modalities:
            modality_mask[:, {'text': 0, 'audio': 1, 'vision': 2}[m]] = True

        if text is None:
            if self.use_bert:
                text = torch.zeros(batch_size, 3, seq_len, device=device, dtype=torch.long)
            else:
                text = torch.zeros(batch_size, seq_len, feature_dims[0], device=device)
        if audio is None:
            audio = torch.zeros(batch_size, seq_len, feature_dims[1], device=device)
        if vision is None:
            vision = torch.zeros(batch_size, seq_len, feature_dims[2], device=device)

        self.eval()
        with torch.no_grad():
            return self.forward(text, audio, vision, modality_mask=modality_mask)
