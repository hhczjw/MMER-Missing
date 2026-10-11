"""
Missing-Modality Robust Multimodal Emotion Recognition
======================================================

核心架构：
    输入特征 → 特征投影 → [模态缺失处理] → CoupledMamba3Fork × N → 池化 → 分类头

核心融合模块：CoupledMamba3Fork（来自 coupled_mamba3_fork.py）
    - CrossMamba3Cell: Mamba-3 SSM + 跨模态 B/C 注入
    - 动态源权重网络：自适应学习源模态贡献
    - 目标主导控制，源提供内容

模态缺失处理策略：
    1. 训练时 Modality Dropout：随机丢弃模态，增强鲁棒性
    2. 缺失模态填充：zero / learned / reconstructed
    3. 模态可用性编码：告知模型哪些模态可用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math

from .coupled_mamba3_fork import CoupledMamba3Fork


# ============================================================================
# 特征投影层
# ============================================================================

class FeatureProjector(nn.Module):
    """将原始模态特征投影到统一的 d_model 维度"""

    def __init__(self, input_dims: list, d_model: int):
        super().__init__()
        self.proj_text = nn.Linear(input_dims[0], d_model)
        self.proj_audio = nn.Linear(input_dims[1], d_model)
        self.proj_video = nn.Linear(input_dims[2], d_model)

    def forward(self, text, audio, vision):
        return self.proj_text(text), self.proj_audio(audio), self.proj_video(vision)


# ============================================================================
# 模态缺失处理器
# ============================================================================

class ModalityMissingHandler(nn.Module):
    """
    处理模态缺失：训练时 Dropout + 缺失填充

    支持两种缺失级别：
    1. 模态级（Modality-level）：整个模态随机丢弃
    2. Token级（Token-level）：逐token随机丢失，模拟真实场景

    填充策略：
    - zero: 缺失模态置零（最简单）
    - learned: 使用可学习的缺失嵌入
    - reconstructed: 从可用模态重建缺失模态（最复杂）

    Token级特殊处理：
    - 文本：保留CLS/SEP token，缺失token替换为UNK(100)
    - 音频/视觉：缺失位置直接置零
    """

    def __init__(self, d_model: int, strategy: str = 'learned',
                 token_missing_rate: float = 0.3, use_token_missing: bool = True):
        """
        Args:
            d_model: 特征维度
            strategy: 填充策略 ('zero', 'learned', 'reconstructed')
            token_missing_rate: Token级缺失概率 (0~1)
            use_token_missing: 是否启用Token级随机丢失
        """
        super().__init__()
        self.d_model = d_model
        self.strategy = strategy
        self.token_missing_rate = token_missing_rate
        self.use_token_missing = use_token_missing

        if strategy == 'learned':
            # 可学习的缺失模态嵌入
            self.missing_text = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.missing_audio = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.missing_vision = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        elif strategy == 'reconstructed':
            # 从可用模态重建缺失模态
            self.reconstruct_text = nn.Sequential(
                nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
            )
            self.reconstruct_audio = nn.Sequential(
                nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
            )
            self.reconstruct_vision = nn.Sequential(
                nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
            )

    def _apply_token_missing(self, text, audio, vision, device):
        """
        Token级随机丢失：逐token以概率p丢失

        对文本：保留CLS/SEP，缺失token替换为UNK(100)
        对音频/视觉：缺失位置直接置零

        Returns:
            token_mask_text: (B, L) 文本token mask, True=保留
            token_mask_audio: (B, L) 音频token mask
            token_mask_vision: (B, L) 视觉token mask
        """
        B, L, _ = text.shape

        # 生成随机mask
        rand_text = torch.rand(B, L, device=device)
        rand_audio = torch.rand(B, L, device=device)
        rand_vision = torch.rand(B, L, device=device)

        # 文本：保留CLS(位置0)和SEP(位置-1)
        rand_text[:, 0] = 1.0   # CLS
        rand_text[:, -1] = 1.0  # SEP

        # 生成mask: True=保留, False=丢失
        token_mask_text = rand_text >= self.token_missing_rate
        token_mask_audio = rand_audio >= self.token_missing_rate
        token_mask_vision = rand_vision >= self.token_missing_rate

        return token_mask_text, token_mask_audio, token_mask_vision

    def _apply_token_mask(self, features, token_mask):
        """
        应用Token级mask：缺失位置置零

        Args:
            features: (B, L, d_model)
            token_mask: (B, L) 布尔张量，True=保留
        """
        return features * token_mask.unsqueeze(-1)

    def forward(self, text, audio, vision, modality_mask):
        """
        混合缺失处理：模态级 + Token级

        1. 模态级：整个模态随机丢弃（训练时由外部dropout控制）
        2. Token级：逐token随机丢失（本模块内部处理）

        Args:
            text, audio, vision: (B, L, d_model) 投影后的特征
            modality_mask: (B, 3) 布尔张量，True=可用, False=缺失

        Returns:
            处理后的 text, audio, vision
        """
        device = text.device

        text_missing = ~modality_mask[:, 0]  # (B,)
        audio_missing = ~modality_mask[:, 1]
        vision_missing = ~modality_mask[:, 2]

        # ---- Step 1: 模态级缺失处理 ----
        if self.strategy == 'zero':
            pass  # 模态级缺失默认已经是零

        elif self.strategy == 'learned':
            text = text.clone()
            audio = audio.clone()
            vision = vision.clone()

            if text_missing.any():
                B_miss = text_missing.sum()
                seq_len = text.size(1)
                text[text_missing] = self.missing_text.expand(B_miss, seq_len, -1)
            if audio_missing.any():
                B_miss = audio_missing.sum()
                seq_len = audio.size(1)
                audio[audio_missing] = self.missing_audio.expand(B_miss, seq_len, -1)
            if vision_missing.any():
                B_miss = vision_missing.sum()
                seq_len = vision.size(1)
                vision[vision_missing] = self.missing_vision.expand(B_miss, seq_len, -1)

        elif self.strategy == 'reconstructed':
            text = text.clone()
            audio = audio.clone()
            vision = vision.clone()

            # 从可用模态重建
            if text_missing.any():
                available = []
                if (~audio_missing).any():
                    available.append(audio.mean(dim=1))
                if (~vision_missing).any():
                    available.append(vision.mean(dim=1))
                if available:
                    concat = torch.cat(available, dim=-1)
                    if concat.size(-1) < self.d_model * 2:
                        concat = F.pad(concat, (0, self.d_model * 2 - concat.size(-1)))
                    recon = self.reconstruct_text(concat).unsqueeze(1).expand(-1, text.size(1), -1)
                    text[text_missing] = recon[text_missing]

            if audio_missing.any():
                available = []
                if (~text_missing).any():
                    available.append(text.mean(dim=1))
                if (~vision_missing).any():
                    available.append(vision.mean(dim=1))
                if available:
                    concat = torch.cat(available, dim=-1)
                    if concat.size(-1) < self.d_model * 2:
                        concat = F.pad(concat, (0, self.d_model * 2 - concat.size(-1)))
                    recon = self.reconstruct_audio(concat).unsqueeze(1).expand(-1, audio.size(1), -1)
                    audio[audio_missing] = recon[audio_missing]

            if vision_missing.any():
                available = []
                if (~text_missing).any():
                    available.append(text.mean(dim=1))
                if (~audio_missing).any():
                    available.append(audio.mean(dim=1))
                if available:
                    concat = torch.cat(available, dim=-1)
                    if concat.size(-1) < self.d_model * 2:
                        concat = F.pad(concat, (0, self.d_model * 2 - concat.size(-1)))
                    recon = self.reconstruct_vision(concat).unsqueeze(1).expand(-1, vision.size(1), -1)
                    vision[vision_missing] = recon[vision_missing]

        # ---- Step 2: Token级随机丢失（仅训练时启用）----
        if self.training and self.use_token_missing and self.token_missing_rate > 0:
            token_mask_text, token_mask_audio, token_mask_vision = \
                self._apply_token_missing(text, audio, vision, device)

            text = self._apply_token_mask(text, token_mask_text)
            audio = self._apply_token_mask(audio, token_mask_audio)
            vision = self._apply_token_mask(vision, token_mask_vision)

        return text, audio, vision


# ============================================================================
# 模态可用性编码器
# ============================================================================

class ModalityAvailabilityEncoder(nn.Module):
    """编码模态可用性信息，注入到特征中"""

    def __init__(self, d_model: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, modality_mask):
        """
        Args:
            modality_mask: (B, 3) 布尔张量
        Returns:
            (B, d_model) 模态可用性嵌入
        """
        return self.encoder(modality_mask.float())


# ============================================================================
# 主模型：MissingModalityMER
# ============================================================================

class MissingModalityMER(nn.Module):
    """
    模态缺失鲁棒的多模态情感识别模型

    核心：CoupledMamba3Fork（Mamba-3 SSM + 跨模态 B/C 注入）
    增强：模态缺失处理 + 模态可用性编码 + 可选 ISM 单模态建模
    """

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

        # ---- 特征投影 ----
        feature_dims = getattr(args, 'feature_dims', [768, 74, 20])
        self.projector = FeatureProjector(feature_dims, d_model)

        # ---- 序列对齐（音频/视频长度可能与文本不同）----
        self.target_seq_len = getattr(args, 'seq_lens', [50, 50, 50])[0]

        # ---- 模态缺失处理（模态级 + Token级混合策略）----
        missing_strategy = getattr(args, 'missing_strategy', 'learned')
        token_missing_rate = getattr(args, 'token_missing_rate', 0.3)
        use_token_missing = getattr(args, 'use_token_missing', True)
        self.missing_handler = ModalityMissingHandler(
            d_model,
            strategy=missing_strategy,
            token_missing_rate=token_missing_rate,
            use_token_missing=use_token_missing
        )

        # ---- 模态可用性编码（可选）----
        self.use_availability_encoding = getattr(args, 'use_availability_encoding', True)
        if self.use_availability_encoding:
            self.availability_encoder = ModalityAvailabilityEncoder(d_model)

        # ---- 核心融合：CoupledMamba3Fork × N ----
        self.fusion_layers = nn.ModuleList([
            CoupledMamba3Fork(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                headdim=headdim,
                ngroups=ngroups,
                rope_fraction=rope_fraction,
                is_outproj_norm=is_outproj_norm,
                is_mimo=is_mimo,
                mimo_rank=mimo_rank,
                chunk_size=chunk_size,
                v_self_ratio=v_self_ratio,
                use_gcmn_gate=use_gcmn_gate,
            )
            for _ in range(num_layers)
        ])

        # ---- 输出层 ----
        pool_type = getattr(args, 'pool_type', 'mean')
        self.pool_type = pool_type
        dropout = getattr(args, 'dropout', 0.5)

        task_type = getattr(args, 'task_type', 'regression')
        num_classes = getattr(args, 'num_classes', 1)

        self.fusion_norm = nn.LayerNorm(d_model * 3)
        self.head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # ---- 模态 Dropout 配置 ----
        self.drop_text_prob = getattr(args, 'drop_text_prob', 0.1)
        self.drop_audio_prob = getattr(args, 'drop_audio_prob', 0.3)
        self.drop_vision_prob = getattr(args, 'drop_vision_prob', 0.3)

    def _pool(self, x, mask=None):
        """池化：支持 mean / last / cls"""
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
        """对齐序列长度到目标长度"""
        if x.size(1) == target_len:
            return x
        return F.adaptive_avg_pool1d(x.transpose(1, 2), target_len).transpose(1, 2)

    def _apply_modality_dropout(self, text, audio, vision):
        """训练时随机丢弃模态"""
        if not self.training:
            batch_size = text.size(0)
            device = text.device
            return text, audio, vision, torch.ones(batch_size, 3, dtype=torch.bool, device=device)

        batch_size = text.size(0)
        device = text.device
        modality_mask = torch.ones(batch_size, 3, dtype=torch.bool, device=device)

        # 按概率独立丢弃
        if self.drop_text_prob > 0:
            drop = torch.rand(batch_size, device=device) < self.drop_text_prob
            text = text.clone()
            text[drop] = 0.0
            modality_mask[drop, 0] = False

        if self.drop_audio_prob > 0:
            drop = torch.rand(batch_size, device=device) < self.drop_audio_prob
            audio = audio.clone()
            audio[drop] = 0.0
            modality_mask[drop, 1] = False

        if self.drop_vision_prob > 0:
            drop = torch.rand(batch_size, device=device) < self.drop_vision_prob
            vision = vision.clone()
            vision[drop] = 0.0
            modality_mask[drop, 2] = False

        return text, audio, vision, modality_mask

    def forward(self, text, audio, vision,
                audio_lengths=None, vision_lengths=None,
                cu_seqlens=None,
                modality_mask=None):
        """
        Args:
            text: (B, L, D_text) 文本特征
            audio: (B, L, D_audio) 音频特征
            vision: (B, L, D_vision) 视觉特征
            modality_mask: (B, 3) 可选外部模态 mask

        Returns:
            dict with 'logits' and 'modality_mask'
        """
        # 1. 训练时模态 Dropout
        text, audio, vision, drop_mask = self._apply_modality_dropout(text, audio, vision)
        if modality_mask is None:
            modality_mask = drop_mask

        # 2. 特征投影
        text, audio, vision = self.projector(text, audio, vision)

        # 3. 序列对齐
        Lt = text.size(1)
        if audio.size(1) != Lt:
            audio = self._align_seq_length(audio, Lt)
        if vision.size(1) != Lt:
            vision = self._align_seq_length(vision, Lt)

        # 4. 模态缺失处理
        text, audio, vision = self.missing_handler(text, audio, vision, modality_mask)

        # 5. 模态可用性编码
        if self.use_availability_encoding:
            avail_embed = self.availability_encoder(modality_mask)  # (B, d_model)
            text = text + avail_embed.unsqueeze(1)
            audio = audio + avail_embed.unsqueeze(1)
            vision = vision + avail_embed.unsqueeze(1)

        # 6. 核心融合：CoupledMamba3Fork × N
        out_a, out_v, out_l = audio, vision, text
        for layer in self.fusion_layers:
            out_a, out_v, out_l = layer(out_a, out_v, out_l, cu_seqlens=cu_seqlens)

        # 8. 池化 + 拼接
        pl = self._pool(out_l)
        pa = self._pool(out_a)
        pv = self._pool(out_v)

        # 9. 分类头
        fused = self.fusion_norm(torch.cat([pl, pa, pv], dim=-1))
        logits = self.head(fused)

        return {
            'logits': logits,
            'modality_mask': modality_mask,
        }

    def inference_with_missing(self, text, audio, vision, available_modalities):
        """
        推理时指定可用模态

        Args:
            text, audio, vision: 输入特征（缺失的可以是 None）
            available_modalities: list of str, e.g. ['text', 'audio']

        Returns:
            dict with predictions
        """
        batch_size = text.size(0) if text is not None else (
            audio.size(0) if audio is not None else vision.size(0)
        )
        device = text.device if text is not None else (
            audio.device if audio is not None else vision.device
        )

        # 构建模态 mask
        modality_mask = torch.zeros(batch_size, 3, dtype=torch.bool, device=device)
        if 'text' in available_modalities:
            modality_mask[:, 0] = True
        if 'audio' in available_modalities:
            modality_mask[:, 1] = True
        if 'vision' in available_modalities:
            modality_mask[:, 2] = True

        # 处理 None 输入
        if text is None:
            text = torch.zeros(batch_size, self.target_seq_len, self.args.feature_dims[0], device=device)
        if audio is None:
            audio = torch.zeros(batch_size, self.target_seq_len, self.args.feature_dims[1], device=device)
        if vision is None:
            vision = torch.zeros(batch_size, self.target_seq_len, self.args.feature_dims[2], device=device)

        self.eval()
        with torch.no_grad():
            return self.forward(text, audio, vision, modality_mask=modality_mask)


# ============================================================================
# 模态 Dropout 率调度器
# ============================================================================

class ModalityDropoutScheduler:
    """
    模态 Dropout 率调度器

    支持：固定 / 线性递增 / 余弦退火 / 课程学习
    """

    def __init__(self, initial_rate=0.0, final_rate=0.3,
                 total_epochs=100, schedule_type='cosine'):
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1

    def get_rate(self):
        if self.schedule_type == 'fixed':
            return self.final_rate

        progress = min(self.current_epoch / self.total_epochs, 1.0)

        if self.schedule_type == 'linear':
            return self.initial_rate + (self.final_rate - self.initial_rate) * progress

        elif self.schedule_type == 'cosine':
            return self.initial_rate + (self.final_rate - self.initial_rate) * \
                   (1 - math.cos(math.pi * progress)) / 2

        elif self.schedule_type == 'curriculum':
            if progress < 0.3:
                return self.initial_rate
            elif progress < 0.7:
                return self.initial_rate + (self.final_rate - self.initial_rate) * \
                       (progress - 0.3) / 0.4
            else:
                return self.final_rate

        return self.final_rate
