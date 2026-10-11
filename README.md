# MMER-Missing: Modality-Missing Multimodal Emotion Recognition

> 面向模态缺失场景的多模态情感识别研究

---

## 研究动机

现有多模态情感识别模型（如 Coupled-BI-Mamba3、CAGMamba 等）在训练和推理时假设所有模态（Text / Audio / Vision）均完整可用。然而在真实场景中，模态缺失是普遍存在的问题：

- **传感器故障**：摄像头遮挡、麦克风静音
- **数据采集不完整**：历史数据缺少某些模态
- **隐私限制**：某些场景不允许采集视频/音频
- **传输丢包**：实时通信中部分模态数据丢失
- **异构设备**：不同终端具备不同的传感器能力

当模态缺失时，现有模型性能会急剧下降，因为其跨模态融合机制依赖于所有模态的完整输入。

---

## 研究目标

设计一个对模态缺失具有鲁棒性的多模态情感识别框架，能够在任意模态组合（单模态、双模态、三模态）下保持稳定且高性能的情感预测。

### 核心问题

1. **训练阶段**：如何让模型学习到对模态缺失鲁棒的表征？
2. **推理阶段**：如何在模态缺失时自适应调整融合策略？
3. **知识迁移**：如何将完整模态的知识迁移到缺失模态场景？

---

## 技术方案：混合模态缺失策略

采用**模态级 + Token级**混合缺失策略，模拟真实场景中的模态缺失情况。

### 模态级随机丢弃（Modality-level）

训练时按概率随机丢弃整个模态：

| 模态 | 丢弃概率 | 说明 |
|------|----------|------|
| 文本 | 10% | 文本通常最可靠，概率最低 |
| 音频 | 30% | 音频易受环境噪声影响 |
| 视觉 | 30% | 视觉易受遮挡、光照影响 |

### Token级随机丢失（Token-level）

逐token以概率p丢失，模拟更细粒度的缺失：

- **文本**：保留CLS/SEP token，缺失token替换为UNK(100)
- **音频/视觉**：缺失位置直接置零
- **缺失率**：30%（可配置）

### 填充策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| **Zero** | 缺失模态置零 | 快速实验 |
| **Learned** | 可学习的缺失嵌入 | 默认选择 |
| **Reconstructed** | 从可用模态重建 | 最复杂但可能最有效 |

### 其他技术方向（作为对比）

| 方向 | 核心思路 | 代表工作 |
|------|----------|----------|
| **知识蒸馏** | 完整模态模型作为教师，缺失模态学生模型学习 | MM-DistillNet |
| **模态生成/补全** | 从已有模态生成缺失模态的特征 | MCTN, SSMT |
| **自适应融合** | 根据可用模态动态调整融合权重 | MulT, MFM-CL |
| **对比学习** | 通过跨模态对比学习共享表征空间 | CLMLF, MMIM |
| **图神经网络** | 将模态关系建模为图，缺失节点自然处理 | MAGNET |

---

## 目录结构

```
MMER-Missing/
├── README.md               # 本文件
├── models/                 # 模型定义
│   ├── __init__.py
│   ├── coupled_mamba3_fork.py    # 核心融合模块（CoupledMamba3Fork）
│   └── missing_modality_model.py # 模态缺失鲁棒模型
├── dataset/                # 数据加载与处理
│   └── data_loader.py
├── configs/                # 实验配置
│   └── missing_modality_config.json
├── scripts/                # 训练/测试脚本
├── logs/                   # 训练日志
├── checkpoints/            # 模型权重
├── results/                # 实验结果
├── docs/                   # 文档与分析
│   ├── hybrid_missing_strategy.md  # 混合缺失策略说明
│   ├── research_plan.md            # 研究计划
│   └── modality_missing_analysis.md # 模态缺失分析
└── references/             # 参考文献
    ├── papers/             # 已下载论文（31篇）
    └── codebases/          # 相关代码仓库（11个）
```

---

## 数据集

计划使用与 Coupled-BI-Mamba3 相同的数据集：

| 数据集 | 任务 | 模态 | 缺失场景 |
|--------|------|------|----------|
| CMU-MOSI | 回归 | T+A+V | 单模态/双模态缺失 |
| CMU-MOSEI | 回归 | T+A+V | 单模态/双模态缺失 |
| IEMOCAP | 分类 | T+A+V | 单模态/双模态缺失 |
| MELD | 分类 | T+A+V | 单模态/双模态缺失 |
| SIMS | 回归 | T+A+V | 单模态/双模态缺失 |

### 模态缺失模拟策略

在测试时模拟以下 7 种模态组合：

| 组合 | Text | Audio | Vision |
|------|:----:|:-----:|:------:|
| TAV | ✅ | ✅ | ✅ |
| TA | ✅ | ✅ | ❌ |
| TV | ✅ | ❌ | ✅ |
| AV | ❌ | ✅ | ✅ |
| T | ✅ | ❌ | ❌ |
| A | ❌ | ✅ | ❌ |
| V | ❌ | ❌ | ✅ |

---

## 评估指标

- **Acc2 Non0（主口径）**：剔除真实标签为 0 的样本，按预测值与标签的正负号计算准确率
- **Acc2 Has0**：保留全部样本，以 `label >= 0`、`pred >= 0` 计算准确率
- **F1 Non0 / Has0**：与上述两种样本口径一致的 weighted F1
- **Acc7**：全测试样本按七个情感区间分箱，不区分 Has0 / Non0
- **MAE / Corr**：全测试样本上的回归指标
- **完整模态性能**：TAV 全部可用时的指标（作为上界）
- **缺失模态性能**：各缺失组合下的指标
- **性能下降率**：`(完整 - 缺失) / 完整 × 100%`
- **平均鲁棒性**：所有缺失组合的平均性能
- **最差情况**：性能最差的缺失组合

---

## 与 Coupled-BI-Mamba3 的关系

本项目可以基于 Coupled-BI-Mamba3 进行扩展，也可以独立设计：

- **如果基于 Coupled-BI-Mamba3**：利用其已有的跨模态 B/C 注入机制，增加模态缺失的鲁棒性处理
- **如果独立设计**：参考 Coupled-BI-Mamba3 的实验配置和数据集，进行公平对比

---

## 环境依赖

（待确定技术路线后补充）

---

## 快速开始

### 环境配置

```bash
# 使用 coupled_mamba_3 环境
conda activate coupled_mamba_3

# 安装 mamba_ssm（必需）
pip install mamba-ssm
```

**注意**：mamba_ssm 需要与 PyTorch 和 CUDA 版本匹配。如果遇到导入错误，请检查版本兼容性。

### 模型使用

```python
import torch
from models import MissingModalityMER

# 加载配置
args = load_config('configs/missing_modality_config.json')

# 创建模型
model = MissingModalityMER(args)

# 训练模式（自动应用混合缺失策略）
model.train()
text = torch.randn(B, L, 768)   # 文本特征
audio = torch.randn(B, L, 74)   # 音频特征
vision = torch.randn(B, L, 20)  # 视觉特征

outputs = model(text, audio, vision)
logits = outputs['logits']
modality_mask = outputs['modality_mask']  # (B, 3) 布尔张量

# 推理模式（指定可用模态）
outputs = model.inference_with_missing(
    text, audio, vision,
    available_modalities=['text', 'audio']  # 缺失视觉
)
```

### 详细文档

- 混合缺失策略：`docs/hybrid_missing_strategy.md`
- 研究计划：`docs/research_plan.md`
- 模态缺失分析：`references/modality_missing_construction.md`

---

## 参考资源

- [Coupled-BI-Mamba3](../H-DCD/coupled_BI_Mamba3/) — 基线模型
- [H-DCD](../H-DCD/) — 层次化解耦对比蒸馏框架
- 相关工作梳理见 `docs/related_work.md`
