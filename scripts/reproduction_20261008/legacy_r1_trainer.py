"""
通用 Trainer: 训练 / 验证 / 测试 单循环, 与具体模型解耦.

改进:
    - 添加 linear warmup + cosine decay 调度
    - 添加 gradient accumulation 支持
    - 添加 InfoNCE 对比损失 (模态对齐)
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from losses import build_loss, MultiTaskLoss, RegressionWithDiscreteCE
from utils.metrics import eval_regression, eval_classification


class ModelEMA:
    """模型权重指数移动平均 (Polyak averaging).
    维护一份影子权重 shadow_state, 每次 update() 后:
        shadow = decay * shadow + (1 - decay) * current
    apply_shadow() / restore() 实现 evaluate 时无缝切换.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        # 仅对 float 参数做 EMA (跳过 int buffer 如 BN num_batches_tracked)
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self._backup = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                v_f = v.detach().float()
                # 跳过 NaN/Inf 参数, 防止 EMA 影子被污染后无法恢复
                if not torch.isfinite(v_f).all():
                    continue
                self.shadow[k].mul_(self.decay).add_(v_f, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module):
        """临时切换到影子权重 (evaluate 用)."""
        self._backup = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if k in self.shadow
        }
        sd = model.state_dict()
        for k in self.shadow:
            sd[k].copy_(self.shadow[k].to(sd[k].dtype))

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        if self._backup is None:
            return
        sd = model.state_dict()
        for k, v in self._backup.items():
            sd[k].copy_(v)
        self._backup = None

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.shadow = {k: v for k, v in sd["shadow"].items()}


class Trainer:
    def __init__(self, args: Any, model: torch.nn.Module, logger):
        self.args = args
        self.model = model
        self.base_model = model.module if hasattr(model, "module") else model
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.task_type = args.task_type
        self.is_multi_task = bool(getattr(args, "multi_task", False))

        # --- 对比损失 ---
        self.contrastive_weight = float(getattr(args, "contrastive_weight", 0.0))
        self.contrastive_temp = float(getattr(args, "contrastive_temp", 0.07))

        # --- gradient accumulation ---
        self.grad_accum_steps = int(getattr(args, "grad_accum_steps", 1))

        # --- gradient clipping (训练后期数值稳定性) ---
        # Mamba3 长训末期容易出现梯度尖峰, 适当收紧 max_norm 可显著降低 NaN 概率
        self.grad_clip = float(getattr(args, "grad_clip", 0.5))

        # --- 复合损失 (回归 + 离散 CE 辅助头, 仅回归任务启用) ---
        self.aux_cls_weight = float(getattr(args, "aux_cls_weight", 0.0))
        self.aux_num_classes = int(getattr(args, "aux_num_classes", 0))
        self.sub_loss_lambda = float(getattr(args, "sub_loss_lambda", 0.0))
        self.use_aux_cls = (
            self.task_type == "regression"
            and not self.is_multi_task
            and self.aux_cls_weight > 0.0
            and self.aux_num_classes > 0
        )
        # use_composite_loss: 任意辅助分支启用就走 RegressionWithDiscreteCE
        self.use_composite_loss = (
            self.task_type == "regression"
            and not self.is_multi_task
            and (self.use_aux_cls or self.sub_loss_lambda > 0.0)
        )

        if self.is_multi_task:
            self.criterion = MultiTaskLoss(
                task_weights=getattr(args, "task_weights", {"M": 1.0}),
                task_type=self.task_type,
            )
        elif self.use_composite_loss:
            self.criterion = RegressionWithDiscreteCE(
                alpha=self.aux_cls_weight,
                num_aux_classes=max(self.aux_num_classes, 7),
                clip_range=float(getattr(args, "aux_clip_range", 3.0)),
                label_smoothing=float(getattr(args, "aux_label_smoothing", 0.05)),
                regression_beta=0.5,
                sub_loss_lambda=self.sub_loss_lambda,
            )
        else:
            self.criterion = build_loss(self.task_type)

        # ------------------------------------------------------------------
        # 优化器: BERT 用小 lr, 其他模块用大 lr
        # ------------------------------------------------------------------
        bert_lr = float(getattr(args, "bert_learning_rate", 2e-5))
        main_lr = float(args.learning_rate)
        wd = float(args.weight_decay)
        bert_params, other_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "text_encoder" in n:
                bert_params.append(p)
            else:
                other_params.append(p)
        param_groups = []
        if bert_params:
            param_groups.append({"params": bert_params, "lr": bert_lr, "weight_decay": wd})
        param_groups.append({"params": other_params, "lr": main_lr, "weight_decay": wd})
        self.optimizer = torch.optim.AdamW(param_groups)
        self.logger.info(
            f"Optimizer: AdamW | bert_lr={bert_lr} ({len(bert_params)} params) | "
            f"main_lr={main_lr} ({len(other_params)} params) | wd={wd} | "
            f"grad_accum={self.grad_accum_steps} | grad_clip={self.grad_clip}"
        )
        if self.use_composite_loss:
            self.logger.info(
                f"Loss: RegressionWithDiscreteCE | alpha={self.aux_cls_weight} | "
                f"aux_num_classes={self.aux_num_classes} | sub_loss_lambda={self.sub_loss_lambda}"
            )
        else:
            self.logger.info(f"Loss: {type(self.criterion).__name__}")

        # --- warmup + cosine decay 调度 ---
        total_epochs = int(getattr(args, "epochs", 30))
        warmup_ratio = float(getattr(args, "warmup_ratio", 0.1))
        train_samples = int(getattr(args, "train_samples", 1000))
        batch_size = int(getattr(args, "batch_size", 32))
        steps_per_epoch = math.ceil(train_samples / batch_size)
        total_steps = steps_per_epoch * total_epochs
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.total_steps = total_steps
        self.global_step = 0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._lr_lambda
        )
        self.logger.info(
            f"Scheduler: warmup {self.warmup_steps} steps + cosine decay | "
            f"total_steps={total_steps}"
        )

        # ------------------------------------------------------------------
        # EMA 影子权重 (可选)
        # ------------------------------------------------------------------
        ema_decay = float(getattr(args, "ema_decay", 0.0) or 0.0)
        self.ema = ModelEMA(self.model, decay=ema_decay) if ema_decay > 0 else None
        if self.ema is not None:
            self.logger.info(f"EMA enabled | decay={ema_decay}")
        else:
            self.logger.info("EMA disabled")

    def _lr_lambda(self, step: int) -> float:
        """线性 warmup + cosine decay"""
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    # ------------------------------------------------------------------
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            elif isinstance(v, dict):
                out[k] = {kk: vv.to(self.device, non_blocking=True) for kk, vv in v.items()}
            else:
                out[k] = v
        return out

    def _forward_pred(self, batch: Dict[str, Any]):
        kw = dict(
            text=batch["text"],
            audio=batch["audio"],
            video=batch["vision"],
            audio_lengths=batch.get("audio_lengths", None),
            vision_lengths=batch.get("vision_lengths", None),
        )
        # ★ Phase 5: 上下文传入
        if "context_text" in batch:
            kw["context_text"] = batch["context_text"]
            kw["context_audio"] = batch["context_audio"]
            kw["context_video"] = batch["context_video"]
        return self.model(**kw)

    @staticmethod
    def _split_outputs(out):
        """模型 forward 可能返回 Tensor 或 dict, 统一拆解为 (logits, aux_logits, sub_outputs).

        sub_outputs: tuple (sub_T, sub_A, sub_V) 或 None
        """
        if isinstance(out, dict):
            logits = out.get("logits")
            aux_logits = out.get("aux_logits", None)
            sub_t = out.get("sub_T")
            sub_a = out.get("sub_A")
            sub_v = out.get("sub_V")
            sub_outputs = (sub_t, sub_a, sub_v) if (sub_t is not None) else None
            return logits, aux_logits, sub_outputs
        return out, None, None

    # 兼容旧调用方 (返回前两项)
    @classmethod
    def _split_logits(cls, out):
        l, a, _ = cls._split_outputs(out)
        return l, a

    def _forward_with_contrastive(self, batch: Dict[str, Any]):
        """前向 + 对比损失 (共享 _encode, 避免重复计算)"""
        model = self.base_model
        audio_lengths = batch.get("audio_lengths", None)
        vision_lengths = batch.get("vision_lengths", None)
        ctx_kw = {}
        if "context_text" in batch:
            ctx_kw = {
                "context_text": batch["context_text"],
                "context_audio": batch["context_audio"],
                "context_video": batch["context_video"],
            }
        # 需提取 CLS 的情况：使用了 sub_loss，或者使用了 aux_head(因为现在 aux_head 需要纯净端)
        need_cls = getattr(model, "use_sub_loss", False) or getattr(model, "aux_head", None) is not None
        if need_cls:
            (out_l, out_a, out_v,
             mask_t, mask_a, mask_v,
             c_t, c_a, c_v) = model._encode(
                batch["text"], batch["audio"], batch["vision"], return_ism_cls=True,
                audio_lengths=audio_lengths, vision_lengths=vision_lengths,
            )
        else:
            out_l, out_a, out_v, mask_t, mask_a, mask_v = model._encode(
                batch["text"], batch["audio"], batch["vision"],
                audio_lengths=audio_lengths, vision_lengths=vision_lengths,
            )
            c_t = c_a = c_v = None
        # 池化 (带 mask)
        pl = model._pool(out_l, model.pool_text  if model.pool_type == "attention" else None, mask=mask_t)
        pa = model._pool(out_a, model.pool_audio if model.pool_type == "attention" else None, mask=mask_a)
        pv = model._pool(out_v, model.pool_video if model.pool_type == "attention" else None, mask=mask_v)
        # 分类头
        fused = model.fusion_norm(torch.cat([pl, pa, pv], dim=-1))
        logits = model.head(fused)
        
        aux_logits = None
        if getattr(model, "aux_head", None) is not None:
            # 双通解耦：将独立提纯的 Audio 和 Video 加入 aux_head
            pure_a = model.aux_a_proj(c_a)
            pure_v = model.aux_v_proj(c_v)
            aux_feat = torch.cat([fused, pure_a, pure_v], dim=-1)
            aux_logits = model.aux_head(aux_feat)
            
        sub_outputs = None
        if getattr(model, "use_sub_loss", False):
            sub_outputs = (
                model.sub_fc_T(c_t),
                model.sub_fc_A(c_a),
                model.sub_fc_V(c_v),
            )
        # 对比损失
        cl = (_info_nce(pl, pa, self.contrastive_temp) +
              _info_nce(pl, pv, self.contrastive_temp) +
              _info_nce(pa, pv, self.contrastive_temp)) / 3.0
        return logits, aux_logits, sub_outputs, cl

    # ------------------------------------------------------------------
    def train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        self.optimizer.zero_grad()
        use_cl = self.contrastive_weight > 0

        nan_skipped = 0
        for step_i, batch in enumerate(loader):
            batch = self._to_device(batch)

            if use_cl:
                logits, aux_logits, sub_outputs, cl = self._forward_with_contrastive(batch)
            else:
                out = self._forward_pred(batch)
                logits, aux_logits, sub_outputs = self._split_outputs(out)
                cl = torch.tensor(0.0, device=self.device)

            label = batch["labels"]["M"] if not self.is_multi_task else None

            if self.is_multi_task:
                # 审查3 修复: 原来只把 M 传给损失, T/A/V 被丢弃, task_weights 形同虚设.
                # 现在利用 sub_fc_T/A/V (模型端的模态级回归头) 作为 T/A/V preds.
                # 要求: model 需设 sub_loss_lambda > 0 (或由 multi_task 自动启用) 使 sub_fc_* 生效.
                mt_preds = {"M": logits.squeeze(-1) if logits.ndim > 1 else logits}
                if sub_outputs is not None:
                    s_t, s_a, s_v = sub_outputs
                    if s_t is not None: mt_preds["T"] = s_t.squeeze(-1) if s_t.ndim > 1 else s_t
                    if s_a is not None: mt_preds["A"] = s_a.squeeze(-1) if s_a.ndim > 1 else s_a
                    if s_v is not None: mt_preds["V"] = s_v.squeeze(-1) if s_v.ndim > 1 else s_v
                loss = self.criterion(mt_preds, batch["labels"])
            elif self.use_composite_loss:
                loss = self.criterion(logits, aux_logits, label, sub_outputs=sub_outputs)
            elif self.task_type == "regression":
                loss = self.criterion(logits.squeeze(-1), label)
            else:
                loss = self.criterion(logits, label)

            # 对比损失
            loss = loss + self.contrastive_weight * cl

            # === NaN 看门狗: 出现非有限值就跳过该 step ===
            if not torch.isfinite(loss):
                nan_skipped += 1
                self.logger.warning(
                    f"[Train] Epoch {epoch} step {step_i}: non-finite loss, skipping batch"
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            loss = loss / self.grad_accum_steps
            loss.backward()

            if (step_i + 1) % self.grad_accum_steps == 0 or (step_i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                if self.ema is not None:
                    self.ema.update(self.model)

            bs = logits.size(0)
            total_loss += float(loss.item()) * bs * self.grad_accum_steps
            n += bs

        avg = total_loss / max(n, 1)
        lrs = [f"{g['lr']:.2e}" for g in self.optimizer.param_groups]
        skip_msg = f" | nan_skipped={nan_skipped}" if nan_skipped > 0 else ""
        self.logger.info(f"[Train] Epoch {epoch} | loss={avg:.4f} | lr={lrs}{skip_msg}")
        return avg

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str = "valid", use_ema: bool = True) -> Dict[str, float]:
        # use_ema=True 且启用了 EMA 时, 切换到影子权重做评估
        ema_active = use_ema and (self.ema is not None)
        if ema_active:
            self.ema.apply_shadow(self.model)
        try:
            self.model.eval()
            all_p, all_t = [], []
            total_loss, total_n = 0.0, 0
            for batch in loader:
                batch = self._to_device(batch)
                out = self._forward_pred(batch)
                logits, _ = self._split_logits(out)
                aux_logits = out.get("aux_logits") if isinstance(out, dict) else None
                sub_outputs = None
                if isinstance(out, dict) and out.get("sub_T") is not None:
                    sub_outputs = (out.get("sub_T"), out.get("sub_A"), out.get("sub_V"))

                label = batch["labels"]["M"]
                if self.task_type == "regression":
                    if self.is_multi_task:
                        mt_preds = {"M": logits.squeeze(-1) if logits.ndim > 1 else logits}
                        if sub_outputs is not None:
                            s_t, s_a, s_v = sub_outputs
                            if s_t is not None: mt_preds["T"] = s_t.squeeze(-1) if s_t.ndim > 1 else s_t
                            if s_a is not None: mt_preds["A"] = s_a.squeeze(-1) if s_a.ndim > 1 else s_a
                            if s_v is not None: mt_preds["V"] = s_v.squeeze(-1) if s_v.ndim > 1 else s_v
                        loss = self.criterion(mt_preds, batch["labels"])
                    elif self.use_composite_loss:
                        loss = self.criterion(logits, aux_logits, label, sub_outputs=sub_outputs)
                    else:
                        loss = self.criterion(logits.squeeze(-1), label)
                else:
                    loss = self.criterion(logits, label)

                bs = logits.size(0)
                total_loss += float(loss.item()) * bs
                total_n += bs
                if self.task_type == "regression":
                    all_p.append(logits.squeeze(-1).cpu().numpy())
                else:
                    all_p.append(logits.cpu().numpy())
                all_t.append(batch["labels"]["M"].cpu().numpy())
            preds = np.concatenate(all_p, axis=0)
            truths = np.concatenate(all_t, axis=0)
            metrics = eval_regression(preds, truths) if self.task_type == "regression" else eval_classification(preds, truths)
            metrics["Loss"] = total_loss / max(total_n, 1)
        finally:
            if ema_active:
                self.ema.restore(self.model)
        msg = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        ema_tag = "+ema" if ema_active else ""
        self.logger.info(f"[{split}{ema_tag}] {msg}")
        return metrics

    def save(self, path: str, use_ema: bool = True) -> None:
        """保存 ckpt. use_ema=True 时存 EMA 权重 (若启用), 否则存 raw 权重.

        新增: 保存前校验权重有限性, 若含 NaN/Inf 则拒绝保存 (保留上一次干净的 ckpt).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if use_ema and self.ema is not None:
            self.ema.apply_shadow(self.model)
            try:
                # 有限性校验 (EMA 影子权重)
                bad = [k for k, v in self.model.state_dict().items()
                       if torch.is_tensor(v) and v.dtype.is_floating_point and not torch.isfinite(v).all()]
                if bad:
                    self.logger.warning(
                        f"[save] EMA weights contain NaN/Inf in {len(bad)} tensors; "
                        f"SKIP saving {path} to preserve previous clean ckpt"
                    )
                    return
                state = {"model": self.model.state_dict(), "args": vars(self.args), "is_ema": True}
                torch.save(state, path)
            finally:
                self.ema.restore(self.model)
        else:
            bad = [k for k, v in self.model.state_dict().items()
                   if torch.is_tensor(v) and v.dtype.is_floating_point and not torch.isfinite(v).all()]
            if bad:
                self.logger.warning(
                    f"[save] Raw weights contain NaN/Inf in {len(bad)} tensors; "
                    f"SKIP saving {path}"
                )
                return
            torch.save({"model": self.model.state_dict(), "args": vars(self.args), "is_ema": False}, path)
        self.logger.info(f"Checkpoint saved: {path}{' [EMA]' if use_ema and self.ema is not None else ''}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        # 关键修复: 若 ckpt 存的是 EMA 权重, 同步刷新 self.ema.shadow,
        # 否则后续 evaluate(use_ema=True) 会用训练末态影子(可能含 NaN) 覆盖刚加载的干净权重.
        if self.ema is not None:
            is_ema_ckpt = bool(ckpt.get("is_ema", False))
            sd = self.model.state_dict()
            for k in list(self.ema.shadow.keys()):
                if k in sd:
                    self.ema.shadow[k] = sd[k].detach().clone().float()
            self.logger.info(
                f"Checkpoint loaded: {path} | EMA shadow refreshed (ckpt is_ema={is_ema_ckpt})"
            )
        else:
            self.logger.info(f"Checkpoint loaded: {path}")


def _info_nce(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    """对称 InfoNCE: z1, z2 shape (B, D)"""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temp        # (B, B)
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
    return loss
