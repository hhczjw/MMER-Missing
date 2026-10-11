"""
UMRM: Unified Missing-Robust Mamba
===================================

统一处理Token级随机丢失和模态级缺失的架构

核心创新：
1. 统一缺失编码器：同时编码Token级和模态级缺失信息
2. 自适应跨模态Mamba：根据缺失情况动态调整B/C注入
3. 多粒度对比重建：Token级+模态级的对比学习
4. 课程训练策略：从易到难的缺失模式

架构：
输入特征 → 统一缺失编码 → 自适应跨模态Mamba × N → 多粒度池化 → 对比学习头 → 分类头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math

# 核心融合模块（来自 Coupled-BI-Mamba3）
from .coupled_mamba3_fork import CoupledMamba3Fork, CrossMamba3Cell


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
        # 将token_mask转换为连续值，表示缺失程度
        token_missing_score = token_mask.unsqueeze(-1).float()  # (B, L, 1)
        token_missing_embed = self.token_missing_embed(token_missing_score)  # (B, L, D)

        # 2. 模态级缺失编码
        # 为当前模态生成缺失嵌入
        # modality_mask是标量（单个模态的mask），需要转换为嵌入
        modality_missing_score = modality_mask.float().unsqueeze(-1)  # (B, 1)
        modality_missing_info = modality_missing_score * self.modality_missing_embed.weight.mean(dim=0)  # (B, D)

        # 3. 多粒度融合
        # 将Token级和模态级信息融合
        token_missing_info = token_missing_embed.mean(dim=1)  # (B, D)
        combined_info = torch.cat([token_missing_info, modality_missing_info], dim=-1)  # (B, 2D)
        gate = self.fusion_gate(combined_info)  # (B, D)
        missing_info = gate * token_missing_info + (1 - gate) * modality_missing_info  # (B, D)

        # 4. 将缺失信息注入到特征中
        encoded_features = features + missing_info.unsqueeze(1)  # (B, L, D)

        return encoded_features, missing_info


# ============================================================================
# 自适应跨模态Mamba
# ============================================================================

class AdaptiveCrossMamba3Cell(nn.Module):
    """
    自适应跨模态Mamba单元

    创新点：
    1. 缺失感知的B/C注入：根据缺失情况动态调整注入强度
    2. 自适应源权重：缺失模态的权重自动降低
    3. 鲁棒的状态转移：对缺失模态更鲁棒
    """

    def __init__(self, d_model: int, d_state: int = 64, expand: int = 2,
                 headdim: int = 32, ngroups: int = 1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # 基础Mamba-3 SSM
        self.ssm = nn.GRU(d_model, d_state)  # 简化的SSM，实际应该用Mamba-3

        # 跨模态B/C注入
        self.cross_modal_proj = nn.Linear(d_model * 2, d_model)

        # 缺失感知的注入强度调节
        self.missing_aware_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # 自适应源权重
        self.source_weight_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, target, source, missing_info_target, missing_info_source):
        """
        Args:
            target: (B, L, d_model) 目标模态特征
            source: (B, L, d_model) 源模态特征
            missing_info_target: (B, d_model) 目标模态缺失信息
            missing_info_source: (B, d_model) 源模态缺失信息

        Returns:
            output: (B, L, d_model) 融合后的特征
        """
        B, L, D = target.shape

        # 1. 计算缺失感知的注入强度
        missing_info = torch.cat([missing_info_target, missing_info_source], dim=-1)  # (B, 2D)
        injection_gate = self.missing_aware_gate(missing_info)  # (B, D)

        # 2. 计算自适应源权重
        source_info = torch.cat([target.mean(dim=1), source.mean(dim=1)], dim=-1)  # (B, 2D)
        source_weight = self.source_weight_net(source_info)  # (B, 1)

        # 3. 跨模态信息注入
        cross_modal_info = self.cross_modal_proj(torch.cat([target, source], dim=-1))  # (B, L, D)

        # 4. 自适应融合
        # 根据缺失情况和源权重调整融合
        output = target + injection_gate.unsqueeze(1) * source_weight.unsqueeze(1) * cross_modal_info

        return output


# ============================================================================
# 多粒度对比重建
# ============================================================================

class MultiGranularityContrastiveReconstruction(nn.Module):
    """
    多粒度对比重建模块

    创新点：
    1. Token级对比：逐token对比学习
    2. 模态级对比：整体模态对比学习
    3. 重建损失：从缺失模态重建完整模态
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
        keep_mask = missing_mask.bool()  # (B, L)
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

        Args:
            reconstructed: (B, L, d_model) 重建的特征
            original: (B, L, d_model) 原始特征
            missing_mask: (B, L) 缺失mask
        """
        # 只计算缺失位置的重建损失
        missing_positions = (~missing_mask.bool()).float()  # (B, L)
        missing_positions = missing_positions.unsqueeze(-1)  # (B, L, 1)

        # 计算重建误差（按特征维度归一化，避免量级随 d_model 膨胀）
        recon_error = (reconstructed - original) ** 2  # (B, L, D)
        D = reconstructed.size(-1)
        recon_loss = (recon_error * missing_positions).sum() / (
            missing_positions.sum() * D).clamp(min=1)

        return recon_loss

    def forward(self, features, original_features, token_mask, modality_mask):
        """
        Args:
            features: (B, L, d_model) 融合后的特征
            original_features: list of (B, L, d_model) 原始模态特征
            token_mask: (B, 3, L) Token级mask
            modality_mask: (B, 3) 模态级mask
        """
        # 1. Token级对比损失
        token_loss = 0.0
        for i in range(3):
            if token_mask[:, i].sum() > 0:
                token_loss += self.token_level_contrastive_loss(features, token_mask[:, i])

        # 2. 模态级对比损失
        modality_features = [feat.mean(dim=1) for feat in original_features]
        modality_loss = self.modality_level_contrastive_loss(modality_features, modality_mask)

        # 3. 重建损失
        reconstructed = self.reconstruct_decoder(features)
        recon_loss = 0.0
        for i in range(3):
            recon_loss += self.reconstruction_loss(reconstructed, original_features[i], token_mask[:, i])

        return token_loss + modality_loss + recon_loss


# ============================================================================
# 课程训练调度器
# ============================================================================

class CurriculumMissingScheduler:
    """
    课程训练调度器：从易到难的缺失模式

    阶段1：只训练模态级缺失（较易）
    阶段2：加入Token级缺失（中等）
    阶段3：同时训练两种缺失（较难）
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
# 主模型：UMRM
# ============================================================================

class UMRM(nn.Module):
    """
    Unified Missing-Robust Mamba

    统一处理Token级随机丢失和模态级缺失的架构

    核心创新：
    1. 统一缺失编码器：同时编码Token级和模态级缺失信息
    2. 自适应跨模态Mamba：根据缺失情况动态调整B/C注入
    3. 多粒度对比重建：Token级+模态级的对比学习
    4. 课程训练策略：从易到难的缺失模式
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

        # 自适应跨模态Mamba层
        self.fusion_layers = nn.ModuleList([
            AdaptiveCrossMamba3Cell(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                headdim=headdim,
                ngroups=ngroups
            )
            for _ in range(num_layers)
        ])

        # 核心融合层：CoupledMamba3Fork
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

        self.fusion_norm = nn.LayerNorm(d_model * 3)
        self.head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
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

        # 6. 自适应跨模态Mamba融合
        # 文本-音频融合
        text_audio = self.fusion_layers[0](
            text_encoded, audio_encoded,
            missing_info_text, missing_info_audio
        )

        # 文本-视觉融合
        text_vision = self.fusion_layers[0](
            text_encoded, vision_encoded,
            missing_info_text, missing_info_vision
        )

        # 音频-视觉融合
        audio_vision = self.fusion_layers[0](
            audio_encoded, vision_encoded,
            missing_info_audio, missing_info_vision
        )

        # 7. 核心融合：CoupledMamba3Fork
        out_a, out_v, out_l = audio_encoded, vision_encoded, text_encoded
        out_a, out_v, out_l = self.core_fusion(out_a, out_v, out_l, cu_seqlens=cu_seqlens)

        # 8. 池化 + 拼接
        pl = self._pool(out_l)
        pa = self._pool(out_a)
        pv = self._pool(out_v)

        # 9. 分类头
        fused = self.fusion_norm(torch.cat([pl, pa, pv], dim=-1))
        logits = self.head(fused)

        result = {
            'logits': logits,
            'token_mask': token_mask,
            'modality_mask': modality_mask,
        }

        # 10. 对比损失（训练时）
        if return_contrastive_loss and self.training:
            # 使用文本特征作为参考，计算对比损失
            contrastive_loss = self.contrastive_reconstruction(
                out_l, [text, audio, vision], token_mask, modality_mask
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
