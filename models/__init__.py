# Models for Modality-Missing Multimodal Emotion Recognition

# 核心融合模块（来自 Coupled-BI-Mamba3）
from .coupled_mamba3_fork import CoupledMamba3Fork, CrossMamba3Cell

# 模态缺失鲁棒模型（支持模态级+Token级混合缺失策略）
from .missing_modality_model import (
    MissingModalityMER,
    FeatureProjector,
    ModalityMissingHandler,
    ModalityAvailabilityEncoder,
    ModalityDropoutScheduler,
)

# UMRM: Unified Missing-Robust Mamba（统一处理Token级和模态级缺失）
from .umrm import (
    UMRM,
    UnifiedMissingEncoder,
    AdaptiveCrossMamba3Cell,
    MultiGranularityContrastiveReconstruction,
    CurriculumMissingScheduler,
)

# UMRM v2: 改进版（缺失感知增强 + CoupledMamba3Fork + 改进对比损失）
from .umrm_v2 import (
    UMRMv2,
    MissingAwareFeatureEnhancer,
    UnifiedMissingEncoder as UnifiedMissingEncoderV2,
    MultiGranularityContrastiveReconstruction as MultiGranularityContrastiveReconstructionV2,
    CurriculumMissingScheduler as CurriculumMissingSchedulerV2,
)

# UMRM v3: 强基线（BERT 微调 + ISM 单模态建模 + 缺失鲁棒融合）
from .umrm_v3 import UMRMv3
