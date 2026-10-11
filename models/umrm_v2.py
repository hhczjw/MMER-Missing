"""
UMRM v2: Unified Missing-Robust Mamba (改进版)
==============================================

改进点：
1. 将缺失感知能力集成到 CrossMamba3Cell
2. 并行设计：fusion_layers 和 core_fusion 并行执行
3. 修改对比损失实现（参考 SDUMC 和 CMAD）

架构：
输入特征 → 特征投影 → 统一缺失编码 → [并行融合] → 池化 → 分类头
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
            AdaptiveCrossMamba    CoupledMamba3Fork
                    ↓                   ↓
                    └─────────┬─────────┘
                              ↓
                         特征融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math

# 核心融合模块（来自 Coupled-BI-Mamba3）
from .coupled_mamba3_fork import CoupledMamba3Fork, CrossMamba3Cell


# ============================================================================
# 缺失感知的特征增强模块
# ============================================================================

class MissingAwareFeatureEnhancer(nn.Module):
    """
    缺失感知的特征增强模块

    作用：在 CoupledMamba3Fork 之前，根据缺失信息增强特征

    创新点：
    1. 根据缺失情况动态调整特征
    2. 自适应源权重：缺失模态的权重自动降低
    3. 保留 CoupledMamba3Fork 作为核心融合模块
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # 缺失感知的特征增强
        self.missing_aware_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, features, missing_info):
        """
        Args:
            features: (B, L, d_model) 模态特征
            missing_info: (B, d_model) 缺失信息嵌入

        Returns:
            enhanced_features: (B, L, d_model) 增强后的特征
        """
        B, L, D = features.shape

        # 1. 计算缺失感知的增强强度
        missing_info_expanded = missing_info.unsqueeze(1).expand(-1, L, -1)  # (B, L, D)
        gate_input = torch.cat([features, missing_info_expanded], dim=-1)  # (B, L, 2D)
        enhancement_gate = self.missing_aware_gate(gate_input)  # (B, L, D)

        # 2. 应用增强
        enhanced_features = features * enhancement_gate

        # 3. 残差连接 + 层归一化
        output = self.layer_norm(features + enhanced_features)

        return output


# ============================================================================
# 统一缺失编码器
# ============================================================================

class UnifiedMissingEncoder(nn.Module):
    """
    统一编码Token级和模态级缺失信息

    创新点：
    1. Token级缺失感知：为每个token生成缺失感知嵌入
    2. 模态级缺失感知：为每个模态生成缺失感知嵌入
    3. 多粒度融合：Token级+模态级信息融合
    """

    def __init__(self, d_model: int, num_modalities: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities

        # Token级缺失编码：为每个token生成缺失感知嵌入
        self.token_missing_embed = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model)
        )

        # 模态级缺失编码：为每个模态生成缺失感知嵌入
        self.modality_missing_embed = nn.Embedding(num_modalities, d_model)

        # 多粒度融合：Token级+模态级信息融合
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, features, token_mask, modality_mask):
        """
        Args:
            features: (B, L, d_model) 模态特征
            token_mask: (B, L) Token级mask，1=保留，0=缺失
            modality_mask: (B,) 模态级mask，1=可用，0=缺失

        Returns:
            encoded_features: (B, L, d_model) 缺失感知特征
            missing_info: (B, d_model) 缺失信息嵌入
        """
        B, L, D = features.shape

        # 1. Token级缺失编码
        token_missing_score = token_mask.unsqueeze(-1).float()  # (B, L, 1)
        token_missing_embed = self.token_missing_embed(token_missing_score)  # (B, L, D)

        # 2. 模态级缺失编码
        modality_missing_score = modality_mask.float().unsqueeze(-1)  # (B, 1)
        modality_missing_info = modality_missing_score * self.modality_missing_embed.weight.mean(dim=0)  # (B, D)

        # 3. 多粒度融合
        token_missing_info = token_missing_embed.mean(dim=1)  # (B, D)
        combined_info = torch.cat([token_missing_info, modality_missing_info], dim=-1)  # (B, 2D)
        gate = self.fusion_gate(combined_info)  # (B, D)
        missing_info = gate * token_missing_info + (1 - gate) * modality_missing_info  # (B, D)

        # 4. 将缺失信息注入到特征中
        encoded_features = features + missing_info.unsqueeze(1)  # (B, L, D)

        return encoded_features, missing_info


# ============================================================================
# 多粒度对比重建（改进版）
# ============================================================================

class MultiGranularityContrastiveReconstruction(nn.Module):
    """
    多粒度对比重建模块（改进版）

    参考：
    - SDUMC: 监督对比学习（SupConLoss）
    - CMAD: 相关性损失 + MSE 损失
    - TF-Mamba: 重建损失

    创新点：
    1. Token级对比：逐token对比学习
    2. 模态级对比：整体模态对比学习
    3. 重建损失：从缺失模态重建完整模态
    4. 相关性损失：保持特征结构一致性
    """

    def __init__(self, d_model: int, temperature: float = 0.07):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature

        # Token级对比投影头
        self.token_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2)
        )

        # 模态级对比投影头
        self.modality_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2)
        )

        # 重建解码器
        self.reconstruct_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

    def token_level_contrastive_loss(self, features, missing_mask):
        """
        Token级对比损失：保留的token应该与重建的token相似

        Args:
            features: (B, L, d_model) 特征
            missing_mask: (B, L) 缺失mask，1=保留，0=缺失
        """
        B, L, D = features.shape

        # 投影到对比空间
        projected = self.token_proj(features)  # (B, L, D/2)
        projected = F.normalize(projected, dim=-1)

        # 分离保留和缺失的token
        keep_mask = missing_mask >= 1.0  # fractional input availability: partial windows are impaired
        loss = 0.0
        count = 0

        for b in range(B):
            keep_idx = keep_mask[b].nonzero(as_tuple=True)[0]
            miss_idx = (~keep_mask[b]).nonzero(as_tuple=True)[0]

            if len(keep_idx) == 0 or len(miss_idx) == 0:
                continue

            # 正样本：保留token之间的相似度
            pos_sim = torch.mm(projected[b, keep_idx], projected[b, keep_idx].t()) / self.temperature

            # 负样本：保留token与缺失token之间的相似度
            neg_sim = torch.mm(projected[b, keep_idx], projected[b, miss_idx].t()) / self.temperature

            # InfoNCE损失
            logits = torch.cat([pos_sim, neg_sim], dim=-1)
            labels = torch.arange(len(keep_idx), device=features.device)
            loss += F.cross_entropy(logits, labels)
            count += 1

        return loss / max(count, 1)

    def modality_level_contrastive_loss(self, modality_features, missing_mask):
        """
        模态级对比损失：不同模态应该学习模态不变特征

        参考 SDUMC 的 SupConLoss 实现

        Args:
            modality_features: list of (B, d_model) 不同模态的特征
            missing_mask: (B, 3) 模态缺失mask
        """
        B = modality_features[0].shape[0]
        loss = 0.0
        count = 0

        for i in range(len(modality_features)):
            for j in range(i + 1, len(modality_features)):
                # 只对比两个模态都可用的样本
                both_available = (missing_mask[:, i] * missing_mask[:, j]).bool()

                if both_available.sum() == 0:
                    continue

                # 投影到对比空间
                feat_i = self.modality_proj(modality_features[i][both_available])
                feat_j = self.modality_proj(modality_features[j][both_available])

                feat_i = F.normalize(feat_i, dim=-1)
                feat_j = F.normalize(feat_j, dim=-1)

                # 正样本：同一样本的不同模态
                pos_sim = (feat_i * feat_j).sum(dim=-1) / self.temperature

                # 负样本：不同样本的模态
                neg_sim = torch.mm(feat_i, feat_j.t()) / self.temperature

                # InfoNCE损失
                logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=-1)
                labels = torch.zeros(len(feat_i), dtype=torch.long, device=feat_i.device)
                loss += F.cross_entropy(logits, labels)
                count += 1

        return loss / max(count, 1)

    def reconstruction_loss(self, reconstructed, original, missing_mask):
        """
        重建损失：从缺失模态重建完整模态

        参考 TF-Mamba 的 ReconLoss 实现

        Args:
            reconstructed: (B, L, d_model) 重建的特征
            original: (B, L, d_model) 原始特征
            missing_mask: (B, L) 缺失mask
        """
        # 只计算缺失位置的重建损失
        missing_positions = (1.0 - missing_mask.float()).clamp(0, 1)  # binary and fractional availability
        missing_positions = missing_positions.unsqueeze(-1)  # (B, L, 1)

        # 计算重建误差（按特征维度归一化，避免量级随 d_model 膨胀）
        recon_error = (reconstructed - original) ** 2  # (B, L, D)
        D = reconstructed.size(-1)
        recon_loss = (recon_error * missing_positions).sum() / (
            missing_positions.sum() * D).clamp(min=1)

        return recon_loss

    def correlation_loss(self, features_i, features_j):
        """
        相关性损失：保持特征结构一致性

        参考 CMAD 的相关性损失实现

        Args:
            features_i: (B, d_model) 特征 i
            features_j: (B, d_model) 特征 j
        """
        # 计算相似度矩阵
        sim_i = F.cosine_similarity(features_i.unsqueeze(1), features_i.unsqueeze(0), dim=2)
        sim_j = F.cosine_similarity(features_j.unsqueeze(1), features_j.unsqueeze(0), dim=2)

        # 计算相似度矩阵的差异
        sim_diff = torch.abs(sim_i - sim_j)
        diagonal_sum = torch.diagonal(sim_diff).sum()
        total_sum = sim_diff.sum()

        # 归一化
        loss = diagonal_sum / total_sum.clamp(min=1e-6)

        return loss

    def forward(self, fused_features, original_features, token_mask, modality_mask):
        """
        Args:
            fused_features: (B, L, d_model) 融合后的特征（单个模态）
            original_features: list of (B, L, d_model) 原始模态特征
            token_mask: (B, 3, L) Token级mask
            modality_mask: (B, 3) 模态级mask
        """
        # 1. Token级对比损失
        token_loss = 0.0
        for i in range(3):
            if token_mask[:, i].sum() > 0:
                token_loss += self.token_level_contrastive_loss(fused_features, token_mask[:, i])

        # 2. 模态级对比损失
        modality_features = [feat.mean(dim=1) for feat in original_features]
        modality_loss = self.modality_level_contrastive_loss(modality_features, modality_mask)

        # 3. 重建损失
        reconstructed = self.reconstruct_decoder(fused_features)
        recon_loss = 0.0
        for i in range(3):
            recon_loss += self.reconstruction_loss(reconstructed, original_features[i], token_mask[:, i])

        # 4. 相关性损失
        corr_loss = 0.0
        fused_pooled = fused_features.mean(dim=1)  # (B, D)
        for i in range(3):
            original_pooled = original_features[i].mean(dim=1)  # (B, D)
            corr_loss += self.correlation_loss(fused_pooled, original_pooled)

        return token_loss + modality_loss + recon_loss + corr_loss


# ============================================================================
# 课程训练调度器
# ============================================================================

class CurriculumMissingScheduler:
    """
    课程训练调度器：从易到难的缺失模式（模态级 + Token级联合训练）

    模态级缺失：全程恒定（默认 30%，随机丢 1-2 个模态）
    Token级缺失：课程递增到 max_token_rate（默认 0.6，对齐 TF-Mamba 采样范围）

    阶段1：只训练模态级缺失（较易）
    阶段2：加入轻量 Token级缺失
    阶段3：中等 Token级缺失
    阶段4：高 Token级缺失（模态级 + Token级联合缺失）
    """

    def __init__(self, total_epochs: int, warmup_epochs: int = 10,
                 max_token_rate: float = 0.6, modality_rate: float = 0.3):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.max_token_rate = max_token_rate
        self.modality_rate = modality_rate

        # 缺失率调度（token 级课程上升，模态级恒定）
        self.token_missing_rates = [
            0.0,
            max_token_rate * 0.25,
            max_token_rate * 0.5,
            max_token_rate,
        ]
        self.modality_missing_rates = [modality_rate] * 4

    def step(self):
        self.current_epoch += 1

    def get_missing_rates(self):
        """获取当前epoch的缺失率（模态级 + Token级联合）"""
        progress = self.current_epoch / self.total_epochs

        if progress < 0.25:
            # 阶段1：只训练模态级缺失
            return {
                'token_missing_rate': self.token_missing_rates[0],
                'modality_missing_rate': self.modality_missing_rates[0],
                'stage': 'modality_only'
            }
        elif progress < 0.5:
            # 阶段2：加入轻量Token级缺失
            return {
                'token_missing_rate': self.token_missing_rates[1],
                'modality_missing_rate': self.modality_missing_rates[1],
                'stage': 'token_light'
            }
        elif progress < 0.75:
            # 阶段3：中等Token级缺失
            return {
                'token_missing_rate': self.token_missing_rates[2],
                'modality_missing_rate': self.modality_missing_rates[2],
                'stage': 'token_medium'
            }
        else:
            # 阶段4：高Token级缺失（联合缺失）
            return {
                'token_missing_rate': self.token_missing_rates[3],
                'modality_missing_rate': self.modality_missing_rates[3],
                'stage': 'full'
            }


# ============================================================================
# 主模型：UMRM v2
# ============================================================================

class UMRMv2(nn.Module):
    """
    Unified Missing-Robust Mamba v2

    改进点：
    1. 将缺失感知能力集成到 CrossMamba3Cell
    2. 并行设计：fusion_layers 和 core_fusion 并行执行
    3. 修改对比损失实现（参考 SDUMC 和 CMAD）
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # 基本配置
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

        # 特征投影
        feature_dims = getattr(args, 'feature_dims', [768, 74, 20])
        self.proj_text = nn.Linear(feature_dims[0], d_model)
        self.proj_audio = nn.Linear(feature_dims[1], d_model)
        self.proj_video = nn.Linear(feature_dims[2], d_model)

        # 序列对齐
        self.target_seq_len = getattr(args, 'seq_lens', [50, 50, 50])[0]

        # 统一缺失编码器
        self.missing_encoder = UnifiedMissingEncoder(d_model)

        # 缺失感知的特征增强层（在 CoupledMamba3Fork 之前）
        # 每模态一个增强器（按模态数，而非层数）
        self.missing_enhancers = nn.ModuleList([
            MissingAwareFeatureEnhancer(d_model=d_model)
            for _ in range(3)
        ])

        # 核心融合层：CoupledMamba3Fork（并行设计）
        self.core_fusion = CoupledMamba3Fork(
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

        # 多粒度对比重建
        self.contrastive_reconstruction = MultiGranularityContrastiveReconstruction(d_model)

        # 课程训练调度器
        self.curriculum_scheduler = CurriculumMissingScheduler(
            total_epochs=getattr(args, 'epochs', 100),
            max_token_rate=getattr(args, 'token_missing_rate_max', 0.6),
            modality_rate=getattr(args, 'modality_missing_rate', 0.3),
        )

        # 输出层
        pool_type = getattr(args, 'pool_type', 'mean')
        self.pool_type = pool_type
        dropout = getattr(args, 'dropout', 0.5)
        task_type = getattr(args, 'task_type', 'regression')
        num_classes = getattr(args, 'num_classes', 1)

        # 并行融合后的特征维度：d_model * 2（fusion_layers + core_fusion）
        self.fusion_norm = nn.LayerNorm(d_model * 2 * 3)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2 * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # 缺失率配置
        self.token_missing_rate = getattr(args, 'token_missing_rate', 0.3)
        self.modality_missing_rate = getattr(args, 'modality_missing_rate', 0.3)

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

    def _generate_token_mask(self, B, L, missing_rate, device):
        """生成Token级随机mask"""
        if missing_rate <= 0:
            return torch.ones(B, L, device=device)

        # 生成随机mask
        mask = torch.rand(B, L, device=device) >= missing_rate

        # 保留CLS和SEP token（如果有）
        mask[:, 0] = 1.0  # CLS
        mask[:, -1] = 1.0  # SEP

        return mask.float()

    def _generate_modality_mask(self, B, missing_rate, device):
        """生成模态级随机mask"""
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

        # 随机选择要缺失的模态
        mask = torch.ones(B, 3, device=device)
        for b in range(B):
            if torch.rand(1) < missing_rate:
                # 随机选择1-2个模态缺失
                num_missing = torch.randint(1, 3, (1,)).item()
                missing_idx = torch.randperm(3)[:num_missing]
                mask[b, missing_idx] = 0.0

        return mask

    def forward(self, text, audio, vision,
                audio_lengths=None, vision_lengths=None,
                cu_seqlens=None,
                token_mask=None, modality_mask=None,
                return_contrastive_loss=False):
        """
        Args:
            text: (B, L, D_text) 文本特征
            audio: (B, L, D_audio) 音频特征
            vision: (B, L, D_vision) 视觉特征
            token_mask: (B, 3, L) 可选Token级mask
            modality_mask: (B, 3) 可选模态级mask
            return_contrastive_loss: 是否返回对比损失

        Returns:
            dict with 'logits', 'token_mask', 'modality_mask', and optionally 'contrastive_loss'
        """
        device = text.device
        B = text.size(0)

        # 1. 特征投影
        text = self.proj_text(text)
        audio = self.proj_audio(audio)
        vision = self.proj_video(vision)

        # 2. 序列对齐
        Lt = text.size(1)
        if audio.size(1) != Lt:
            audio = self._align_seq_length(audio, Lt)
        if vision.size(1) != Lt:
            vision = self._align_seq_length(vision, Lt)

        # 3. 生成缺失mask（如果未提供）
        if token_mask is None:
            if not self.training:
                # 评估模式：不随机丢失，保持全 1
                token_mask = torch.ones(B, 3, Lt, device=device)
            else:
                # 生成Token级mask
                token_mask_text = self._generate_token_mask(B, Lt, self.token_missing_rate, device)
                token_mask_audio = self._generate_token_mask(B, Lt, self.token_missing_rate, device)
                token_mask_vision = self._generate_token_mask(B, Lt, self.token_missing_rate, device)
                token_mask = torch.stack([token_mask_text, token_mask_audio, token_mask_vision], dim=1)  # (B, 3, L)

        if modality_mask is None:
            if not self.training:
                # 评估模式：默认全模态可用（缺失组合由外部传入 mask 控制）
                modality_mask = torch.ones(B, 3, device=device)
            else:
                # 生成模态级mask
                modality_mask = self._generate_modality_mask(B, self.modality_missing_rate, device)  # (B, 3)

        # 4. 应用缺失mask
        # Token级：逐token置零
        text = text * token_mask[:, 0, :].unsqueeze(-1)
        audio = audio * token_mask[:, 1, :].unsqueeze(-1)
        vision = vision * token_mask[:, 2, :].unsqueeze(-1)

        # 模态级：整个模态置零
        text = text * modality_mask[:, 0].unsqueeze(-1).unsqueeze(-1)
        audio = audio * modality_mask[:, 1].unsqueeze(-1).unsqueeze(-1)
        vision = vision * modality_mask[:, 2].unsqueeze(-1).unsqueeze(-1)

        # 5. 统一缺失编码
        text_encoded, missing_info_text = self.missing_encoder(text, token_mask[:, 0], modality_mask[:, 0])
        audio_encoded, missing_info_audio = self.missing_encoder(audio, token_mask[:, 1], modality_mask[:, 1])
        vision_encoded, missing_info_vision = self.missing_encoder(vision, token_mask[:, 2], modality_mask[:, 2])

        # 6. 缺失感知的特征增强（在 CoupledMamba3Fork 之前）
        # 每模态使用各自的增强器，缺失信息增强对应模态特征
        text_enhanced = self.missing_enhancers[0](text_encoded, missing_info_text)
        audio_enhanced = self.missing_enhancers[1](audio_encoded, missing_info_audio)
        vision_enhanced = self.missing_enhancers[2](vision_encoded, missing_info_vision)

        # 7. 核心融合：CoupledMamba3Fork（使用增强后的特征）
        out_a, out_v, out_l = self.core_fusion(audio_enhanced, vision_enhanced, text_enhanced, cu_seqlens=cu_seqlens)

        # 8. 并行设计：增强后特征 + CoupledMamba3Fork 输出 拼接
        text_fused = torch.cat([text_enhanced, out_l], dim=-1)  # (B, L, d_model * 2)
        audio_fused = torch.cat([audio_enhanced, out_a], dim=-1)  # (B, L, d_model * 2)
        vision_fused = torch.cat([vision_enhanced, out_v], dim=-1)  # (B, L, d_model * 2)

        # 7. 池化 + 拼接
        pl = self._pool(text_fused)
        pa = self._pool(audio_fused)
        pv = self._pool(vision_fused)

        # 8. 分类头
        fused = self.fusion_norm(torch.cat([pl, pa, pv], dim=-1))
        logits = self.head(fused)

        result = {
            'logits': logits,
            'token_mask': token_mask,
            'modality_mask': modality_mask,
        }

        # 9. 对比损失（训练时）
        if return_contrastive_loss and self.training:
            # 使用 CoupledMamba3Fork 的输出进行对比学习
            contrastive_loss = self.contrastive_reconstruction(
                out_l, [text_enhanced, audio_enhanced, vision_enhanced], token_mask, modality_mask
            )
            result['contrastive_loss'] = contrastive_loss

        return result

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

        # 构建模态mask
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
