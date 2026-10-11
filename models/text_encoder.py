"""
TextPretrainedEncoder: 通用文本预训练编码器
===========================================

支持 BERT / RoBERTa / DeBERTa 等 HuggingFace 模型。
自动检测模型类型, 适配不同的输入格式。
当输入是 BERT token ID 但模型非 BERT 时, 自动解码→重编码 (CAGMamba 对齐)。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAWrapper(nn.Module):
    """低秩旁路包装：冻结基础 Linear，只训 lora_A/lora_B"""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        h = self.base(x)
        h = h + (x @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scale
        return h


def apply_lora_to_roberta(model, r: int = 8, alpha: int = 16):
    """对 RoBERTa 每层 attention 的 q/k/v/o 加 LoRA 旁路（其余参数冻结）"""
    n_replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        short = name.split('.')[-1]
        if short in ('query', 'key', 'value', 'dense') and 'attention' in name:
            parent = model
            parts = name.split('.')[:-1]
            for p in parts:
                parent = getattr(parent, p)
            setattr(parent, short, LoRAWrapper(module, r=r, alpha=alpha))
            n_replaced += 1
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad = True
    print(f'[LoRA] 已注入 {n_replaced} 个低秩旁路（r={r}, alpha={alpha}），基础参数冻结')
    return n_replaced


class TextPretrainedEncoder(nn.Module):
    """
    通用文本预训练编码器, 支持 BERT / RoBERTa / DeBERTa 等 HuggingFace 模型.

    与原有 BertTextEncoder 的差异:
    1. 支持 RoBERTa (无 token_type_ids, 使用 <s>/</s> 分隔符)
    2. 支持 DeBERTa (相对位置编码)
    3. 自动检测模型类型, 适配不同的输入格式
    4. ★ CAGMamba 对齐: 当输入是 BERT token ID 但模型非 BERT 时, 自动解码→重编码
       (decode BERT ids → raw text → re-encode with current tokenizer)
    """

    # 预定义模型配置映射
    SUPPORTED_MODELS = {
        "bert-base-uncased":    {"dim": 768, "type": "bert"},
        "bert-large-uncased":   {"dim": 1024, "type": "bert"},
        "roberta-base":         {"dim": 768, "type": "roberta"},
        "roberta-large":        {"dim": 1024, "type": "roberta"},
        "microsoft/deberta-base": {"dim": 768, "type": "deberta"},
    }

    def __init__(
        self,
        pretrained: str = "roberta-base",
        finetune: bool = True,
        strict: bool = True,
        pretokenized: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
    ):
        super().__init__()
        self.pretrained_name = pretrained
        self.use_hf = False
        # ★ 当三通道 ids 已由当前 tokenizer 编码时，跳过 BERT 解码→重编码
        self.pretokenized = pretokenized

        # 自动检测模型类型
        self.model_type = "bert"  # fallback
        for key, info in self.SUPPORTED_MODELS.items():
            if key in pretrained.lower():
                self.model_type = info["type"]
                self.out_dim = info["dim"]
                break

        # ★ 安全默认值: 防止 SUPPORTED_MODELS 未匹配时 out_dim 未定义
        if not hasattr(self, 'out_dim'):
            self.out_dim = 768

        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
            self.transformer = AutoModel.from_pretrained(pretrained)
            self.out_dim = self.transformer.config.hidden_size
            self.use_hf = True
        except Exception as e:
            if strict:
                raise RuntimeError(f"加载 {pretrained} 失败: {e}")
            print(f"[WARN] 回退到 Embedding: {e}")
            self.transformer = nn.Embedding(50265, self.out_dim, padding_idx=1)
            self.tokenizer = None

        # ★ CAGMamba 对齐: 当模型非 BERT 时, 预加载 BERT tokenizer 用于解码旧的 .pkl 数据
        self._bert_tokenizer = None
        if self.model_type != "bert" and self.use_hf:
            try:
                from transformers import BertTokenizer
                self._bert_tokenizer = BertTokenizer.from_pretrained(
                    "bert-base-uncased"
                )
            except Exception:
                pass  # 静默失败: 回退到旧行为

        if self.use_hf and not finetune:
            for p in self.transformer.parameters():
                p.requires_grad = False

        # LoRA 微调：冻结基础参数，仅训低秩旁路
        if self.use_hf and finetune == 'lora':
            apply_lora_to_roberta(self.transformer, r=lora_r, alpha=lora_alpha)

    def forward(
        self, text_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_input:
                - 离线三通道模式 (兼容旧 BERT .pkl): (B, 3, L)
                - 在线模式: (B, L) 仅 input_ids
        Returns:
            hidden:         (B, L, out_dim)
            attention_mask: (B, L) long
        """
        # ==== 检测输入格式 ====
        if text_input.dim() == 3 and text_input.size(1) == 3:
            # ─── 旧 BERT 三通道格式: [ids, mask, segment] ───
            input_ids = text_input[:, 0].long()
            attention_mask = text_input[:, 1].long()
            old_max_len = text_input.size(2)

            # ★ CAGMamba 对齐: BERT token ID → 模型 token ID 的自动转换
            if (self._bert_tokenizer is not None and self.model_type != "bert"
                    and not self.pretokenized):
                raw_texts = self._bert_tokenizer.batch_decode(
                    input_ids, skip_special_tokens=True
                )
                encoded = self.tokenizer(
                    raw_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=old_max_len,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(text_input.device)
                attention_mask = encoded["attention_mask"].to(text_input.device)

        elif text_input.dim() == 2:
            # ─── 纯 token ID 模式 ───
            input_ids = text_input.long()
            attention_mask = (
                input_ids != self.tokenizer.pad_token_id
            ).long() if self.tokenizer else (input_ids != 0).long()
        else:
            input_ids = text_input.squeeze(1).long()
            attention_mask = (input_ids != 0).long()

        if not self.use_hf:
            return self.transformer(input_ids), attention_mask

        kw = {"input_ids": input_ids, "attention_mask": attention_mask}

        out = self.transformer(**kw)
        return out.last_hidden_state, attention_mask
