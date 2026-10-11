"""
MMER-Missing 训练 + 7 种模态组合评估脚本
=========================================

用法:
    # 训练（默认 umrmv2 + MOSI）
    python scripts/train.py --config configs/umrm_v2_config.json --model umrmv2 --dataset MOSI

    # 其他架构
    python scripts/train.py --model missing --config configs/missing_modality_config.json --dataset MOSI
    python scripts/train.py --model umrm --config configs/umrm_config.json --dataset MOSI

    # 仅评估（加载已有 checkpoint）
    python scripts/train.py --model umrmv2 --dataset MOSI --eval-only \
        --checkpoint checkpoints/umrmv2_MOSI_seed42_best.pth

评估：测试集 7 种模态组合（TAV/TA/TV/AV/T/A/V）+ 平均鲁棒性 + 最差 + 下降率
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# 将项目根目录加入 sys.path（脚本位于 scripts/ 子目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _path_setup  # noqa: F401,E402  必须在所有 MMER-Missing import 之前

from utils.seed import set_seed
from config import load_config
from dataset.data_loader import MMDataLoader
from losses.task_losses import TaskLoss
from utils.metrics import eval_regression, eval_classification
from models import MissingModalityMER, UMRM, UMRMv2, UMRMv3

MODEL_CLS = {
    'missing': MissingModalityMER,
    'umrm': UMRM,
    'umrmv2': UMRMv2,
    'umrmv3': UMRMv3,
}

# 7 种评估组合（TAV + 6 缺失）
COMBOS = [
    ('TAV', ['text', 'audio', 'vision']),
    ('TA',  ['text', 'audio']),
    ('TV',  ['text', 'vision']),
    ('AV',  ['audio', 'vision']),
    ('T',   ['text']),
    ('A',   ['audio']),
    ('V',   ['vision']),
]
MOD_IDX = {'text': 0, 'audio': 1, 'vision': 2}


def _text_missing_input(text, model):
    """构造缺失文本输入：三通道 ids 全用 tokenizer 的 pad_token_id（BERT=0，RoBERTa=1）"""
    tok = getattr(model.bert_encoder, 'tokenizer', None)
    pad_id = tok.pad_token_id if (tok is not None and tok.pad_token_id is not None) else 0
    return torch.full(text.size(), pad_id, device=text.device, dtype=torch.long)


# ============================================================================
# 评估
# ============================================================================

@torch.no_grad()
def evaluate_combos(model, loader, args, device):
    """在测试集上评估 7 种模态组合"""
    model.eval()
    task_type = args.task_type

    preds = {name: [] for name, _ in COMBOS}
    labels = []

    for batch in loader:
        text = batch['text'].to(device)
        audio = batch['audio'].to(device)
        vision = batch['vision'].to(device)
        label = batch['label'].to(device)
        B = text.size(0)

        labels.append(label.cpu().numpy())

        for name, mods in COMBOS:
            # 缺失模态置零 + 显式 mask
            use_bert = getattr(model, 'use_bert', False)
            if 'text' in mods:
                t = text
            else:
                t = (_text_missing_input(text, model) if use_bert else torch.zeros_like(text))
            a = audio if 'audio' in mods else torch.zeros_like(audio)
            v = vision if 'vision' in mods else torch.zeros_like(vision)

            modality_mask = torch.zeros(B, 3, device=device, dtype=torch.bool)
            for m in mods:
                modality_mask[:, MOD_IDX[m]] = True

            if args.model_name == 'missing':
                out = model(t, a, v, modality_mask=modality_mask)
            else:
                out = model(t, a, v, modality_mask=modality_mask, token_mask=None)

            logits = out['logits']
            if task_type == 'regression':
                p = logits.squeeze(-1).cpu().numpy()
            else:
                p = logits.argmax(dim=-1).cpu().numpy()
            preds[name].append(p)

    labels = np.concatenate(labels).flatten()
    results = {}
    for name, _ in COMBOS:
        p = np.concatenate(preds[name]).flatten()
        if task_type == 'regression':
            m = eval_regression(p, labels)
            results[name] = {k: round(float(v), 4) for k, v in m.items()}
        else:
            m = eval_classification(p, labels, num_classes=args.num_classes)
            results[name] = {k: round(float(v), 4) for k, v in m.items()}

    # ---- 汇总：缺失组合（不含 TAV）的全指标统计 ----------------
    # 与 MPLMM / HARDY-MER 的 6 缺失组合平均协议对齐，便于直接对标
    missing_names = [n for n, _ in COMBOS if n != 'TAV']
    metric_keys = list(results['TAV'].keys())
    LOWER_BETTER = ('MAE', 'Loss')
    summary = {}
    for k in metric_keys:
        vals = [results[n][k] for n in missing_names]
        summary[f'missing_avg_{k}'] = round(float(np.mean(vals)), 4)
        summary[f'missing_worst_{k}'] = round(float(max(vals) if k in LOWER_BETTER else min(vals)), 4)
        summary[f'gap_vs_TAV_{k}'] = round(float(abs(np.mean(vals) - results['TAV'][k])), 4)

    # 兼容旧字段：avg_robustness / worst_case / gap_vs_TAV（用主指标）
    key = args.KeyEval if hasattr(args, 'KeyEval') else 'MAE'
    key_upper = {'mae': 'MAE', 'loss': 'MAE', 'f1': 'F1_nonzero', 'acc': 'Accuracy',
                 'acc2': 'Acc2_nonzero', 'acc2_nonzero': 'Acc2_nonzero',
                 'acc2_has0': 'Acc2_has0', 'f1_nonzero': 'F1_nonzero',
                 'f1_has0': 'F1_has0', 'acc7': 'Acc7', 'corr': 'Corr'}.get(key.lower(), key)
    if key_upper in results['TAV']:
        missing_vals = [results[n][key_upper] for n in missing_names]
        lower_better = key_upper in LOWER_BETTER
        summary['avg_robustness'] = round(float(np.mean(missing_vals)), 4)
        summary['worst_case'] = round(float(max(missing_vals) if lower_better else min(missing_vals)), 4)
        summary['gap_vs_TAV'] = round(float(np.mean([abs(v - results['TAV'][key_upper]) for v in missing_vals])), 4)
    results['summary'] = summary
    return results


def evaluate_token_missing(model, loader, args, device, rates=None):
    """
    Token 缺失曲线：UMRMv3 默认 feature_valid（ISM 后缺失，与训练位置一致）。
    设置 args.token_eval_protocol='input_valid_aware' 可评测输入级缺失。
    两种缺失位置分别标注，不将特征级结果作为输入级论文对比。

    对测试集逐 token 随机丢失，缺失率 r ∈ {0, 0.1, ..., 0.9}：
    - 文本：保留 CLS/SEP，缺失 token 替换为 UNK（BERT 输入时替换 ids）
    - 音频/视觉：缺失位置置零
    每个缺失率报告四指标，并输出全率平均（对标 TF-Mamba Table 3）。
    """
    rates = rates or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if args.model_name == 'umrmv3':
        from utils.token_evaluation import evaluate_curve
        return evaluate_curve(model, loader, args, device, rates)
    model.eval()
    task_type = args.task_type
    use_bert = getattr(model, 'use_bert', False)
    unk_id = 100
    if use_bert:
        tok = getattr(model.bert_encoder, 'tokenizer', None)
        if tok is not None and tok.unk_token_id is not None:
            unk_id = tok.unk_token_id

    def _metrics(preds_list, labels_list):
        p = np.concatenate(preds_list).flatten()
        y = np.concatenate(labels_list).flatten()
        if task_type == 'regression':
            return {k: round(float(v), 4) for k, v in eval_regression(p, y).items()}
        return {k: round(float(v), 4) for k, v in eval_classification(p, y, num_classes=args.num_classes).items()}

    curve = {}
    for r in rates:
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                text = batch['text'].to(device)
                audio = batch['audio'].to(device)
                vision = batch['vision'].to(device)
                label = batch['label'].to(device)
                B = text.size(0)
                gen = torch.Generator(device=device).manual_seed(42)

                # ---- Token 级随机丢失 ----
                if use_bert:
                    L = text.size(-1)
                    rand = torch.rand(B, L, device=device, generator=gen)
                    m = rand >= r
                    m[:, 0] = 1.0  # CLS
                    m[:, -1] = 1.0  # SEP
                    t = text.clone()
                    t[:, 0][~m] = unk_id  # 缺失 token 替换 UNK
                else:
                    rand = torch.rand(B, text.size(1), device=device, generator=gen)
                    m = (rand >= r).float()
                    m[:, 0] = 1.0
                    m[:, -1] = 1.0
                    t = text * m.unsqueeze(-1)

                # 音频/视觉逐 token 置零
                ra = torch.rand(B, audio.size(1), device=device, generator=gen) >= r
                ra[:, 0] = 1.0; ra[:, -1] = 1.0
                a = audio * ra.unsqueeze(-1)
                rv = torch.rand(B, vision.size(1), device=device, generator=gen) >= r
                rv[:, 0] = 1.0; rv[:, -1] = 1.0
                v = vision * rv.unsqueeze(-1)

                if args.model_name == 'missing':
                    out = model(t, a, v)
                else:
                    out = model(t, a, v, token_mask=None, modality_mask=None)
                logits = out['logits']
                if task_type == 'regression':
                    preds.append(logits.squeeze(-1).cpu().numpy())
                else:
                    preds.append(logits.argmax(dim=-1).cpu().numpy())
                labels.append(label.cpu().numpy())

        curve[f'{r:.1f}'] = _metrics(preds, labels)

    # ---- 全率平均（对标 TF-Mamba 的 missing-setting 平均） ----
    keys = list(curve[f'{rates[0]:.1f}'].keys())
    avg = {}
    for k in keys:
        avg[k] = round(float(np.mean([curve[f'{r:.1f}'][k] for r in rates])), 4)
    return {'per_rate': curve, 'avg_over_rates': avg}


def evaluate_joint_grid(model, loader, args, device, rates=(0.0, 0.3, 0.6)):
    """
    模态缺失组合 × Token缺失率 二维网格评测

    对每个模态组合（TAV/TA/TV/AV/T/A/V）在多个 Token 缺失率下分别评估，
    输出 (7 组合 × N 率) 四指标矩阵 + 汇总，用于同时对比两种缺失维度。
    """
    if args.model_name == 'umrmv3':
        from utils.token_evaluation import evaluate_curve
        results = {}
        for name, mods in COMBOS:
            curve = evaluate_curve(model, loader, args, device, rates, mods)
            results[name] = dict(curve['per_rate'], avg=curve['avg_over_rates'])
        keys = list(results['TAV']['avg'])
        results['summary'] = {f'r{r:.1f}_{k}': float(np.mean([results[n][f'{r:.1f}'][k] for n,_ in COMBOS]))
                              for r in rates for k in keys}
        results['protocol'] = curve['protocol']
        return results
    task_type = args.task_type
    use_bert = getattr(model, 'use_bert', False)
    unk_id = 100
    if use_bert:
        tok = getattr(model.bert_encoder, 'tokenizer', None)
        if tok is not None and tok.unk_token_id is not None:
            unk_id = tok.unk_token_id

    def _metrics(preds_list, labels_list):
        p = np.concatenate(preds_list).flatten()
        y = np.concatenate(labels_list).flatten()
        if task_type == 'regression':
            return {k: round(float(v), 4) for k, v in eval_regression(p, y).items()}
        return {k: round(float(v), 4) for k, v in eval_classification(p, y, num_classes=args.num_classes).items()}

    results = {}
    for name, mods in COMBOS:
        combo_res = {}
        for r in rates:
            preds, labels = [], []
            with torch.no_grad():
                for batch in loader:
                    text = batch['text'].to(device)
                    audio = batch['audio'].to(device)
                    vision = batch['vision'].to(device)
                    label = batch['label'].to(device)
                    B = text.size(0)
                    gen = torch.Generator(device=device).manual_seed(42)

                    # ---- 模态级缺失：缺失模态置零 ----------------
                    if use_bert:
                        t_full = text if 'text' in mods else _text_missing_input(text, model)
                    else:
                        t_full = text if 'text' in mods else torch.zeros_like(text)
                    a_full = audio if 'audio' in mods else torch.zeros_like(audio)
                    v_full = vision if 'vision' in mods else torch.zeros_like(vision)

                    # ---- Token级缺失（仅对可用模态） ----
                    if use_bert:
                        L = text.size(-1)
                        m = torch.rand(B, L, device=device, generator=gen) >= r
                        m[:, 0] = 1.0
                        m[:, -1] = 1.0
                        t = t_full.clone()
                        if 'text' in mods:
                            t[:, 0][~m] = unk_id
                    else:
                        m = (torch.rand(B, text.size(1), device=device, generator=gen) >= r).float()
                        m[:, 0] = 1.0
                        m[:, -1] = 1.0
                        t = t_full * (m.unsqueeze(-1) if 'text' in mods else 1.0)

                    ra = torch.rand(B, audio.size(1), device=device, generator=gen) >= r
                    ra[:, 0] = 1.0; ra[:, -1] = 1.0
                    a = a_full * (ra.unsqueeze(-1) if 'audio' in mods else 1.0)
                    rv = torch.rand(B, vision.size(1), device=device, generator=gen) >= r
                    rv[:, 0] = 1.0; rv[:, -1] = 1.0
                    v = v_full * (rv.unsqueeze(-1) if 'vision' in mods else 1.0)

                    mm = torch.zeros(B, 3, device=device, dtype=torch.bool)
                    for mname in mods:
                        mm[:, MOD_IDX[mname]] = True

                    if args.model_name == 'missing':
                        out = model(t, a, v, modality_mask=mm)
                    else:
                        out = model(t, a, v, modality_mask=mm, token_mask=None)
                    logits = out['logits']
                    if task_type == 'regression':
                        preds.append(logits.squeeze(-1).cpu().numpy())
                    else:
                        preds.append(logits.argmax(dim=-1).cpu().numpy())
                    labels.append(label.cpu().numpy())

            combo_res[f'{r:.1f}'] = _metrics(preds, labels)

        # 该组合跨 token 率的平均
        keys = list(combo_res[f'{rates[0]:.1f}'].keys())
        combo_res['avg'] = {k: round(float(np.mean([combo_res[f'{r:.1f}'][k] for r in rates])), 4)
                            for k in keys}
        results[name] = combo_res

    # 全局汇总：每个 token 率下所有组合的平均 MAE（双维度鲁棒性总览）
    summary = {}
    for r in rates:
        for k in keys:
            vals = [results[n][f'{r:.1f}'][k] for n, _ in COMBOS]
            summary[f'r{r:.1f}_{k}'] = round(float(np.mean(vals)), 4)
    results['summary'] = summary
    return results


def evaluate(model, loader, args, device):
    """标准评估（TAV 完整模态，验证集用）"""
    model.eval()
    task_type = args.task_type
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            vision = batch['vision'].to(device)
            label = batch['label'].to(device)
            if args.model_name == 'missing':
                out = model(text, audio, vision)
            else:
                out = model(text, audio, vision, token_mask=None, modality_mask=None)
            logits = out['logits']
            if task_type == 'regression':
                preds.append(logits.squeeze(-1).cpu().numpy())
            else:
                preds.append(logits.argmax(dim=-1).cpu().numpy())
            labels.append(label.cpu().numpy())
    preds = np.concatenate(preds).flatten()
    labels = np.concatenate(labels).flatten()
    if task_type == 'regression':
        return eval_regression(preds, labels)
    return eval_classification(preds, labels, num_classes=args.num_classes)


# ============================================================================
# 训练
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, args, device, epoch, total_epochs):
    model.train()
    total_loss, steps = 0.0, 0
    use_kd = getattr(args, 'use_kd', False) and args.model_name != 'missing'
    kd_weight = getattr(args, 'kd_weight', 0.3)
    kd_feat_weight = getattr(args, 'kd_feat_weight', 0.0)
    use_enh_kd = getattr(args, 'use_enh_kd', False)
    enh_kd_w = getattr(args, 'enh_kd_weight', 0.1)
    use_rcd2 = getattr(args, 'use_rcd2', False)
    rcd2_w = getattr(args, 'rcd2_weight', 0.15)
    use_dual_mask = getattr(args, 'use_dual_mask', False)
    dual_mask_w = getattr(args, 'dual_mask_weight', 0.2)
    input_token_training = getattr(args, 'token_train_protocol', 'feature_valid') == 'input_valid_aware'
    if input_token_training and args.model_name != 'umrmv3':
        raise ValueError('Input Token training is currently implemented for UMRMv3 only')
    if input_token_training and use_dual_mask:
        raise ValueError('Legacy dual-mask path is incompatible with input Token training')
    if input_token_training:
        from utils.token_protocol import validity, native_corruption, aligned_availability
        corruption_rng = torch.Generator(device=device).manual_seed(args.seed + epoch * 1_000_003)
        sampled_rates = []

    # 课程调度（v1/v2/v3）
    if not input_token_training and args.model_name in ('umrm', 'umrmv2', 'umrmv3') and getattr(args, 'use_curriculum', False):
        rates = model.curriculum_scheduler.get_missing_rates()
        model.token_missing_rate = rates['token_missing_rate']
        model.modality_missing_rate = rates['modality_missing_rate']

    # Tier1-4：课程耦合 KD 权重（CMAD 式）：缺失越重 KD 越强（早期保护全模态拟合）
    if use_kd and getattr(args, 'use_cd_kd', False):
        tmax = max(getattr(args, 'token_missing_rate_max', 0.6), 1e-6)
        prog = min(getattr(model, 'token_missing_rate', 0.0) / tmax, 1.0)
        kd_weight = kd_weight * (0.5 + 0.5 * prog)

    for batch in loader:
        text = batch['text'].to(device)
        audio = batch['audio'].to(device)
        vision = batch['vision'].to(device)
        label = batch['label'].to(device)

        # ---- 教师路径：完整模态前向（stop-gradient） ----
        if use_kd or input_token_training:
            B = text.size(0)
            # 序列长度：BERT 三通道 (B,3,L) 取最后一维；普通特征 (B,L,D) 取中间维
            use_bert = getattr(model, 'use_bert', False)
            if use_bert:
                Lt = text.size(-1)
            else:
                Lt = text.size(1)
            full_mm = torch.ones(B, 3, device=device)
            full_tm = torch.ones(B, 3, Lt, device=device)
            feat_kd_on = (kd_feat_weight > 0 and args.model_name == 'umrmv3')
            # Deterministic clean view for the corrected input-level contract.
            if input_token_training:
                model.eval()
            try:
                with torch.no_grad():
                    out_teacher = model(text, audio, vision,
                                        token_mask=full_tm, modality_mask=full_mm,
                                        return_fused=feat_kd_on,
                                        return_pre_pools=use_enh_kd,
                                        **({'return_text_target': True, 'return_reconstruction_targets': True}
                                           if input_token_training else {}))
            finally:
                if input_token_training:
                    model.train()

        # ---- 学生路径：随机缺失前向 ----
        feat_kd_on = (use_kd and kd_feat_weight > 0 and args.model_name == 'umrmv3')
        need_fused2 = getattr(args, 'use_dual_mask', False)
        if args.model_name == 'missing':
            out = model(text, audio, vision)
            loss = criterion(out['logits'], label)
        else:
            length_kwargs = {}
            if args.model_name == 'umrmv3':
                length_kwargs = {
                    'audio_lengths': batch.get('audio_valid_lengths', batch.get('audio_lengths')),
                    'vision_lengths': batch.get('vision_valid_lengths', batch.get('vision_lengths')),
                }
            if input_token_training:
                valid = validity(text, audio, vision, length_kwargs['audio_lengths'], length_kwargs['vision_lengths'])
                cap = int(round(getattr(args, 'token_train_rate_cap', 0.9) * 10))
                rate = getattr(args, 'token_train_fixed_rate', None)
                if rate is None:
                    rate = int(torch.randint(cap + 1, (), device=device, generator=corruption_rng)) / 10
                sampled_rates.append(rate)
                model.token_missing_rate = rate
                unk_id = model.bert_encoder.tokenizer.unk_token_id
                (text, audio, vision), keeps, _ = native_corruption(
                    text, audio, vision, valid, rate, corruption_rng, unk_id)
                length_kwargs.update(
                    token_mask=aligned_availability(keeps, text.shape[-1], valid),
                    modality_mask=torch.ones(text.shape[0], 3, device=device),
                    text_reconstruction_target=out_teacher['text_target'] if model.use_text_recon else None,
                    reconstruction_targets=out_teacher['reconstruction_targets'])
            out = model(text, audio, vision, return_contrastive_loss=True,
                        return_fused=(feat_kd_on or need_fused2),
                        return_pre_pools=use_enh_kd, **length_kwargs)
            loss = criterion(out['logits'], label)
            cl = out.get('contrastive_loss', None)
            if cl is not None:
                w = getattr(args, 'contrastive_loss_weight', 0.1)
                loss = loss + w * cl
            # 方案①：TF-Mamba 式文本重建损失
            trl = out.get('text_recon_loss', None)
            if trl is not None:
                loss = loss + getattr(args, 'text_recon_weight', 0.5) * trl
            # 7 分类辅助损失（仅 --aux-cls 时启用；RCD 用解耦蒸馏而非 CE）
            aux_logits = out.get('aux_logits', None)
            if aux_logits is not None and getattr(args, 'use_aux_cls', False) and args.task_type == 'regression':
                label_cls = (torch.round(label.squeeze(-1).clamp(-3, 3)) + 3).long()
                aux_loss = F.cross_entropy(aux_logits, label_cls)
                loss = loss + getattr(args, 'aux_cls_weight', 0.3) * aux_loss
            # Tier1-2：EBMC 式单模态头 + 能量差平衡
            uni = out.get('uni_logits', None)
            if uni is not None and getattr(args, 'use_uni', False):
                mm = out['modality_mask']
                y = label.squeeze(-1)
                per_mod = [F.smooth_l1_loss(uni[k].squeeze(-1), y, reduction='none')
                           for k in ('T', 'A', 'V')]
                uni_w = getattr(args, 'uni_weight', 0.3)
                uni_loss = sum((l * mm[:, i]).sum() / mm[:, i].sum().clamp(min=1)
                               for i, l in enumerate(per_mod))
                loss = loss + uni_w * uni_loss
                # 能量差平衡（仅三模态全可用样本，抑制文本支路梯度主导）
                gap_w = getattr(args, 'gap_weight', 0.1)
                full = (mm.min(dim=1).values > 0.5)
                if full.sum() > 1 and gap_w > 0:
                    gap = sum(((per_mod[i][full] - per_mod[j][full]) ** 2).mean()
                              for i, j in ((0, 1), (0, 2), (1, 2)))
                    loss = loss + gap_w * gap

        # ---- 蒸馏损失：学生 logits 对齐完整模态教师 ----
        if use_kd:
            if args.task_type == 'regression':
                kd_loss = F.smooth_l1_loss(out['logits'], out_teacher['logits'].detach())
            else:
                kd_loss = F.kl_div(
                    F.log_softmax(out['logits'], dim=-1),
                    F.softmax(out_teacher['logits'].detach() / getattr(args, 'kd_temp', 2.0), dim=-1),
                    reduction='batchmean')
            loss = loss + kd_weight * kd_loss
            # 特征层蒸馏：融合特征对齐（仅 v3）
            if feat_kd_on and 'fused' in out and 'fused' in out_teacher:
                kd_feat = F.smooth_l1_loss(out['fused'], out_teacher['fused'].detach())
                loss = loss + kd_feat_weight * kd_feat
            # Tier1: 增强器级特征蒸馏（CI-Der 式：Fork 前 pre_pools 对齐）
            if use_enh_kd and 'pre_pools' in out and 'pre_pools' in out_teacher:
                pp_s = torch.cat(out['pre_pools'], dim=-1)
                pp_t = torch.cat(out_teacher['pre_pools'], dim=-1)
                loss = loss + enh_kd_w * F.smooth_l1_loss(pp_s, pp_t.detach())
            # Tier1: 2分类解耦蒸馏 RCD2（TCR + NTCR，二分类版）
            rcd2_logits = out.get('rcd2_logits', None)
            t2 = out_teacher.get('rcd2_logits', None) if use_kd else None
            if use_rcd2 and rcd2_logits is not None and t2 is not None:
                lb2 = (label.squeeze(-1) >= 0).long()
                pt = F.softmax(t2.detach(), dim=-1)
                ps = F.softmax(rcd2_logits, dim=-1)
                idx = lb2.unsqueeze(1)
                tcr = F.smooth_l1_loss(ps.gather(1, idx), pt.gather(1, idx))
                mask = torch.ones_like(pt)
                mask.scatter_(1, idx, 0.0)
                ps_nt = (ps * mask); pt_nt = (pt * mask)
                ps_nt = ps_nt / ps_nt.sum(-1, keepdim=True).clamp(min=1e-6)
                pt_nt = pt_nt / pt_nt.sum(-1, keepdim=True).clamp(min=1e-6)
                ntcr = F.kl_div(torch.log(ps_nt.clamp(min=1e-8)), pt_nt, reduction='batchmean')
                loss = loss + rcd2_w * (tcr + 0.5 * ntcr)
            # Tier1-3：CorrKD 式 7 类解耦蒸馏（TCR 目标类 + NTCR 非目标类）
            if (getattr(args, 'use_rcd', False)
                    and 'aux_logits' in out and 'aux_logits' in out_teacher):
                label_cls = (torch.round(label.squeeze(-1).clamp(-3, 3)) + 3).long()
                pt = F.softmax(out_teacher['aux_logits'].detach(), dim=-1)
                ps = F.softmax(out['aux_logits'], dim=-1)
                idx = label_cls.unsqueeze(1)
                tcr = F.smooth_l1_loss(ps.gather(1, idx), pt.gather(1, idx))
                mask = torch.ones_like(pt)
                mask.scatter_(1, idx, 0.0)
                ps_nt = (ps * mask); pt_nt = (pt * mask)
                ps_nt = ps_nt / ps_nt.sum(-1, keepdim=True).clamp(min=1e-6)
                pt_nt = pt_nt / pt_nt.sum(-1, keepdim=True).clamp(min=1e-6)
                ntcr = F.kl_div(torch.log(ps_nt.clamp(min=1e-8)), pt_nt, reduction='batchmean')
                loss = loss + getattr(args, 'rcd_weight', 0.2) * (tcr + 0.5 * ntcr)

        # ---- Tier2: GSDNet 式缺失 A/V 生成重建损失 ----
        if getattr(args, 'use_av_recon', False) and 'av_recon_loss' in out:
            avr_w = getattr(args, 'av_recon_weight', None) or 0.5
            loss = loss + avr_w * out['av_recon_loss']

        # ---- Tier2-14: FSRF 式双掩码视图一致性（第二套缺失掩码前向） ----
        if use_dual_mask:
            out2 = model(text, audio, vision, return_fused=True)
            fused1 = out.get('fused', None)
            fused2 = out2.get('fused', None)
            if fused1 is not None and fused2 is not None:
                l_f = F.smooth_l1_loss(fused1, fused2.detach())
                l_y = F.smooth_l1_loss(out['logits'], out2['logits'].detach())
                loss = loss + dual_mask_w * (l_f + 0.5 * l_y)

        optimizer.zero_grad()
        if input_token_training and not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite input Token loss at epoch {epoch}, batch {steps}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(args, 'grad_clip', 1.0),
                                      error_if_nonfinite=input_token_training)
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    if input_token_training:
        args.token_train_last_stats = {'protocol': 'input_valid_aware', 'steps': steps,
                                      'rate_histogram': {f'{r/10:.1f}': sampled_rates.count(r/10) for r in range(10)}}
    return total_loss / max(steps, 1)


# ============================================================================
# 辅助：训练/评估后的三维评测（7组合 + Token曲线 + 联合网格）
# ============================================================================

def _run_extra_evals(model, test_loader, args, device, tag, log, skip=False):
    """训练完成后自动执行 Token 缺失曲线 + 模态×Token 联合网格评测"""
    if skip:
        log('已跳过额外评测（--skip-extra-eval）')
        return

    # 1. Token 级缺失率曲线（TF-Mamba / LNLN 协议）
    tmr = evaluate_token_missing(model, test_loader, args, device)
    out_path2 = f'results/{tag}_tokenmissing.json'
    with open(out_path2, 'w') as f:
        json.dump(tmr, f, indent=2)
    log(f'[额外评测1/2] Token缺失曲线 -> {out_path2}')
    log(f'  全率平均: {tmr["avg_over_rates"]}')

    # 2. 模态组合 × Token 缺失率联合网格
    jg = evaluate_joint_grid(model, test_loader, args, device)
    out_path3 = f'results/{tag}_jointgrid.json'
    with open(out_path3, 'w') as f:
        json.dump(jg, f, indent=2)
    log(f'[额外评测2/2] 模态×Token联合网格 -> {out_path3}')
    for name, combo_res in jg.items():
        if name == 'summary':
            log(f'  汇总: {combo_res}')
            continue
        avg = combo_res.get('avg', {})
        log(f'  {name}: r0.0={combo_res.get("0.0", {})} | avg={avg}')


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/umrm_v2_config.json')
    parser.add_argument('--model', type=str, default='umrmv2',
                        choices=['missing', 'umrm', 'umrmv2', 'umrmv3'])
    parser.add_argument('--dataset', type=str, default='MOSI')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--track-metrics', nargs='+', default=None,
                        choices=['MAE', 'Corr', 'Acc2_nonzero', 'Acc2_has0',
                                 'F1_nonzero', 'F1_has0', 'Acc7'],
                        help='仅保存这些辅助指标的最佳 checkpoint；主指标仍独立保存')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data-root', type=str, default=None, help='覆盖 dataset_root_dir')
    parser.add_argument('--kd', action='store_true', help='启用缺失感知自蒸馏')
    parser.add_argument('--kd-weight', type=float, default=None, help='蒸馏损失权重')
    parser.add_argument('--kd-temp', type=float, default=None, help='蒸馏温度')
    parser.add_argument('--kd-feat-weight', type=float, default=None, help='特征层蒸馏权重（>0 启用）')
    parser.add_argument('--text-recon', action='store_true', help='启用 TF-Mamba 式文本重建损失')
    parser.add_argument('--text-recon-weight', type=float, default=None, help='文本重建损失权重')
    parser.add_argument('--text-inject', action='store_true', help='启用相似度注入回路（方案②，与重建配合）')
    parser.add_argument('--aux-cls', action='store_true', help='启用 7 分类辅助头（直接优化 Acc7）')
    parser.add_argument('--aux-cls-weight', type=float, default=None, help='辅助分类损失权重')
    parser.add_argument('--token-max', type=float, default=None, help='覆盖 token 缺失率课程上限 token_missing_rate_max')
    parser.add_argument('--modality-rate', type=float, default=None, help='覆盖训练模态缺失率 modality_missing_rate')
    parser.add_argument('--no-bert', action='store_true', help='不用 BERT（用 pkl 离线 768 特征）')
    parser.add_argument('--freeze-bert', action='store_true', help='使用 BERT 但冻结参数不微调')
    parser.add_argument('--bert-model', type=str, default=None,
                        help='文本预训练模型（如 FacebookAI/roberta-base），覆盖配置 bert_pretrained')
    parser.add_argument('--bert-pretokenized', action='store_true',
                        help='text_bert 三通道已由当前 tokenizer 离线编码（跳过 BERT 解码→重编码）')
    parser.add_argument('--feature-path', type=str, default=None, help='覆盖数据 pkl 路径 featurePath（相对 dataset_root_dir）')
    parser.add_argument('--bi', action='store_true', help='启用双向支路（VIM 反向扫描 + 相加）')
    parser.add_argument('--uaf', action='store_true', help='Tier1-1: MRUF 式 σ² 不确定性门控')
    parser.add_argument('--uni', action='store_true', help='Tier1-2: EBMC 式单模态头+能量平衡')
    parser.add_argument('--uni-weight', type=float, default=None, help='单模态损失权重（默认 0.3）')
    parser.add_argument('--gap-weight', type=float, default=None, help='能量平衡损失权重（默认 0.1）')
    parser.add_argument('--rcd', action='store_true', help='Tier1-3: CorrKD 式 7 类解耦蒸馏（需 --kd）')
    parser.add_argument('--rcd-weight', type=float, default=None, help='解耦蒸馏权重（默认 0.2）')
    parser.add_argument('--cd-kd', action='store_true', help='Tier1-4: 课程耦合 KD 权重调度（需 --kd）')
    parser.add_argument('--rcd2', action='store_true', help='Tier1: 2分类解耦蒸馏（CorrKD 二分类版，需 --kd）')
    parser.add_argument('--rcd2-weight', type=float, default=None, help='2分类解耦蒸馏权重（默认 0.15）')
    parser.add_argument('--enh-kd', action='store_true', help='Tier1: 增强器级特征蒸馏（CI-Der 式，Fork 前对齐，需 --kd）')
    parser.add_argument('--enh-kd-weight', type=float, default=None, help='增强器级特征蒸馏权重（默认 0.1）')
    parser.add_argument('--av-recon', action='store_true', help='Tier2: GSDNet 式缺失 A/V 池化级生成重建')
    parser.add_argument('--av-recon-weight', type=float, default=None, help='A/V 重建损失权重（默认 0.5）')
    parser.add_argument('--dual-mask', action='store_true', help='Tier2-14: FSRF 式双掩码视图一致性（融合+logits）')
    parser.add_argument('--dual-mask-weight', type=float, default=None, help='双掩码一致性权重（默认 0.2）')
    parser.add_argument('--hard-mask-pool', action='store_true', help='Tier2-15: 缺失支路池化硬掩码清零（SIEVE 观察）')
    parser.add_argument('--balanced-missing', action='store_true', help='Tier3-16: 训练组合均衡缺失采样（7 模式等权）')
    parser.add_argument('--mask-cond-head', action='store_true', help='Tier3-17: 缺失掩码条件化 head（拼 modality_mask）')
    parser.add_argument('--combo-bias', action='store_true', help='Tier3-18: 组合级可学习偏置校准（7 标量）')
    parser.add_argument('--combo-heads', action='store_true', help='Tier3-19: 7 组合独立 head（DVS 式）')
    parser.add_argument('--bssm-gate', action='store_true', help='ISM BSSM 门控（CAGMamba 公式 15-16: SiLU(MLP) 逐元素乘）')
    parser.add_argument('--audio-dim', type=int, default=None, help='覆盖音频特征维度（wav2vec 1024 等）')
    parser.add_argument('--lora-bert', action='store_true', help='RoBERTa 用 LoRA 低秩微调（省显存，large 可用）')
    parser.add_argument('--opt-8bit', action='store_true', help='用 8-bit AdamW（large 全微调省显存）')
    parser.add_argument('--num-layers', type=int, default=None, help='覆盖 Fork 堆叠层数')
    parser.add_argument('--key-eval', type=str, default=None,
                        choices=['MAE', 'Acc2', 'Acc2_nonzero', 'Acc2_has0',
                                 'F1', 'F1_nonzero', 'F1_has0', 'Acc7', 'Corr'],
                        help='checkpoint 选优/早停主指标（默认用配置 KeyEval）')
    parser.add_argument('--eval-token-missing', action='store_true',
                        help='训练后/评估时追加 Token 级缺失率曲线评测（TF-Mamba 协议）')
    parser.add_argument('--skip-extra-eval', action='store_true',
                        help='跳过训练后的 Token 缺失曲线与联合网格评测（默认自动执行）')
    args_cli = parser.parse_args()

    # 加载配置
    args = load_config(args_cli.config, args_cli.dataset)
    args.model_name = args_cli.model
    if args_cli.epochs: args.epochs = args_cli.epochs
    if args_cli.batch_size: args.batch_size = args_cli.batch_size
    if args_cli.lr: args.learning_rate = args_cli.lr
    if args_cli.data_root: args.dataset_root_dir = args_cli.data_root
    args.seed = args_cli.seed
    if args_cli.kd: args.use_kd = True
    if args_cli.kd_weight is not None: args.kd_weight = args_cli.kd_weight
    if args_cli.kd_temp is not None: args.kd_temp = args_cli.kd_temp
    if args_cli.kd_feat_weight is not None: args.kd_feat_weight = args_cli.kd_feat_weight
    if args_cli.text_recon: args.use_text_recon = True
    if args_cli.text_recon_weight is not None: args.text_recon_weight = args_cli.text_recon_weight
    if args_cli.text_inject: args.use_text_injection = True
    if args_cli.aux_cls: args.use_aux_cls = True
    if args_cli.aux_cls_weight is not None: args.aux_cls_weight = args_cli.aux_cls_weight
    if args_cli.token_max is not None: args.token_missing_rate_max = args_cli.token_max
    if args_cli.modality_rate is not None: args.modality_missing_rate = args_cli.modality_rate
    if args_cli.no_bert: args.use_bert = False
    if args_cli.freeze_bert:
        args.use_bert = True
        args.bert_finetune = False
    if args_cli.lora_bert:
        args.use_bert = True
        args.bert_finetune = 'lora'
    if args_cli.opt_8bit:
        args.use_opt_8bit = True
    if args_cli.bert_model: args.bert_pretrained = args_cli.bert_model
    if args_cli.bert_pretokenized: args.bert_pretokenized = True
    if args_cli.feature_path: args.featurePath = args_cli.feature_path
    if args_cli.bi: args.use_bi_direction = True
    if args_cli.uaf: args.use_uaf = True
    if args_cli.uni: args.use_uni = True
    if args_cli.uni_weight is not None: args.uni_weight = args_cli.uni_weight
    if args_cli.gap_weight is not None: args.gap_weight = args_cli.gap_weight
    if args_cli.rcd: args.use_rcd = True
    if args_cli.rcd_weight is not None: args.rcd_weight = args_cli.rcd_weight
    if args_cli.cd_kd: args.use_cd_kd = True
    if args_cli.rcd2: args.use_rcd2 = True
    if args_cli.rcd2_weight is not None: args.rcd2_weight = args_cli.rcd2_weight
    if args_cli.enh_kd: args.use_enh_kd = True
    if args_cli.enh_kd_weight is not None: args.enh_kd_weight = args_cli.enh_kd_weight
    if args_cli.av_recon: args.use_av_recon = True
    if args_cli.av_recon_weight is not None: args.av_recon_weight = args_cli.av_recon_weight
    if args_cli.dual_mask: args.use_dual_mask = True
    if args_cli.dual_mask_weight is not None: args.dual_mask_weight = args_cli.dual_mask_weight
    if args_cli.hard_mask_pool: args.use_hard_mask_pool = True
    if args_cli.balanced_missing: args.use_balanced_missing = True
    if args_cli.mask_cond_head: args.use_mask_cond_head = True
    if args_cli.combo_bias: args.use_combo_bias = True
    if args_cli.combo_heads: args.use_combo_heads = True
    if args_cli.bssm_gate: args.use_bssm_gate = True
    if args_cli.audio_dim is not None:
        fd = list(getattr(args, 'feature_dims', [768, 5, 20]))
        fd[1] = args_cli.audio_dim
        args.feature_dims = fd
    if args_cli.num_layers is not None: args.num_layers = args_cli.num_layers
    if args_cli.key_eval: args.KeyEval = args_cli.key_eval

    set_seed(args.seed)
    device = torch.device(args_cli.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        # Triton kernels use the current CUDA device; keep it aligned with model tensors.
        torch.cuda.set_device(device)

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    kd_tag = f'_kd{getattr(args, "kd_weight", 0.3)}' if getattr(args, 'use_kd', False) else ''
    recon_tag = '_recon' if getattr(args, 'use_text_recon', False) else ''
    inj_tag = '_inj' if getattr(args, 'use_text_injection', False) else ''
    aux_tag = '_aux' if getattr(args, 'use_aux_cls', False) else ''
    tkmax = getattr(args, 'token_missing_rate_max', 0.6)
    tk_tag = f'_tk{tkmax}' if tkmax != 0.6 else ''
    if getattr(args, 'use_bert', True):
        bp = getattr(args, 'bert_pretrained', 'bert-base-uncased')
        BERT_SHORT = {'bert-base-uncased': '', 'roberta-base': '_rb',
                      'FacebookAI/roberta-base': '_rb', 'roberta-large': '_rbl',
                      'FacebookAI/roberta-large': '_rbl', 'bert-large-uncased': '_bl'}
        bert_tag = BERT_SHORT.get(bp, '_' + bp.split('/')[-1].replace('-', '')[:10])
        if getattr(args, 'bert_pretokenized', False):
            bert_tag += '_rbp'
        if not getattr(args, 'bert_finetune', True):
            bert_tag += '_frzbert'
        if getattr(args, 'bert_finetune', True) == 'lora':
            bert_tag += '_lora'
    else:
        bert_tag = '_offline'
    feat_tag = f'_feat{getattr(args, "kd_feat_weight", 0.0)}' if getattr(args, 'kd_feat_weight', 0.0) > 0 else ''
    bi_tag = '_bi' if getattr(args, 'use_bi_direction', False) else ''
    uaf_tag = '_uaf' if getattr(args, 'use_uaf', False) else ''
    uni_tag = f'_uni{getattr(args, "uni_weight", 0.3):g}' if getattr(args, 'use_uni', False) else ''
    if getattr(args, 'use_uni', False) and getattr(args, 'gap_weight', 0.1) != 0.1:
        uni_tag += f'_g{getattr(args, "gap_weight", 0.1):g}'
    rcd_tag = f'_rcd{getattr(args, "rcd_weight", 0.2):g}' if getattr(args, 'use_rcd', False) else ''
    cdkd_tag = '_cdkd' if getattr(args, 'use_cd_kd', False) else ''
    rcd2_tag = '_rcd2' if getattr(args, 'use_rcd2', False) else ''
    enhkd_tag = '_enhkd' if getattr(args, 'use_enh_kd', False) else ''
    avr_tag = '_avr' if getattr(args, 'use_av_recon', False) else ''
    dm_tag = '_dm' if getattr(args, 'use_dual_mask', False) else ''
    hmp_tag = '_hmp' if getattr(args, 'use_hard_mask_pool', False) else ''
    bal_tag = '_bal' if getattr(args, 'use_balanced_missing', False) else ''
    mch_tag = '_mch' if getattr(args, 'use_mask_cond_head', False) else ''
    cb_tag = '_cb' if getattr(args, 'use_combo_bias', False) else ''
    ch_tag = '_ch' if getattr(args, 'use_combo_heads', False) else ''
    bssm_tag = '_bssm' if getattr(args, 'use_bssm_gate', False) else ''
    w2v_tag = '_w2v' if getattr(args, 'feature_dims', [768, 5, 20])[1] == 1024 else ''
    b8_tag = '_8bit' if getattr(args, 'use_opt_8bit', False) else ''
    nl_tag = f'_nl{args.num_layers}' if getattr(args, 'num_layers', 2) != 2 else ''
    mr_tag = f'_mr{getattr(args, "modality_missing_rate", 0.3)}' if getattr(args, 'modality_missing_rate', 0.3) != 0.3 else ''
    tag = f'{args.model_name}{kd_tag}{recon_tag}{inj_tag}{aux_tag}{tk_tag}{bert_tag}{feat_tag}{bi_tag}{uaf_tag}{uni_tag}{rcd_tag}{cdkd_tag}{rcd2_tag}{enhkd_tag}{avr_tag}{dm_tag}{hmp_tag}{bal_tag}{mch_tag}{cb_tag}{ch_tag}{bssm_tag}{w2v_tag}{b8_tag}{nl_tag}{mr_tag}_{args.dataset}_seed{args.seed}'
    log_path = f'logs/{tag}.log'

    def log(msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        with open(log_path, 'a') as f:
            f.write(line + '\n')

    log(f'=== 训练配置 ===')
    log(f'模型: {args.model_name} | 数据集: {args.dataset} | device: {device}')
    log(f'epochs={args.epochs} batch_size={args.batch_size} lr={args.learning_rate} seed={args.seed}')
    log(f'featurePath: {args.featurePath}')

    # 模型
    model = MODEL_CLS[args.model_name](args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'参数量: {n_params / 1e6:.2f}M')

    criterion = TaskLoss(task_type=args.task_type, num_classes=args.num_classes)

    # BERT 微调：低学习率参数组（对齐 H-DCD：bert_lr=1e-5, main_lr=3e-4~5e-4）
    # LoRA 旁路参数（名字含 lora_）归主参数组（需正常 lr）
    bert_lr = getattr(args, 'bert_lr', None)
    bert_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and n.startswith('bert_encoder') and 'lora_' not in n]
    main_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not (n.startswith('bert_encoder') and 'lora_' not in n)]
    if getattr(args, 'use_opt_8bit', False):
        import bitsandbytes as bnb
        if bert_lr is not None and bert_params:
            optimizer = bnb.optim.AdamW8bit(
                [{'params': main_params}, {'params': bert_params, 'lr': bert_lr}],
                lr=args.learning_rate,
                weight_decay=getattr(args, 'weight_decay', 1e-4),
            )
        else:
            optimizer = bnb.optim.AdamW8bit(
                main_params + bert_params, lr=args.learning_rate,
                weight_decay=getattr(args, 'weight_decay', 1e-4),
            )
    elif bert_lr is not None and bert_params:
        optimizer = torch.optim.AdamW(
            [{'params': main_params}, {'params': bert_params, 'lr': bert_lr}],
            lr=args.learning_rate,
            weight_decay=getattr(args, 'weight_decay', 1e-4),
        )
    else:
        optimizer = torch.optim.AdamW(
            main_params + bert_params, lr=args.learning_rate,
            weight_decay=getattr(args, 'weight_decay', 1e-4),
        )

    # checkpoint 路径
    ckpt_path = args_cli.checkpoint or f'checkpoints/{tag}_best.pth'

    if args_cli.eval_only:
        assert os.path.exists(ckpt_path), f'checkpoint 不存在: {ckpt_path}'
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model'] if 'model' in state else state)
        log(f'已加载 checkpoint: {ckpt_path}')
        test_loader = MMDataLoader(args, mode='test')
        results = evaluate_combos(model, test_loader, args, device)
        out_path = f'results/{tag}_eval.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        log(f'评估完成 -> {out_path}')
        for name, m in results.items():
            log(f'  {name}: {m}')
        _run_extra_evals(model, test_loader, args, device, tag, log, args_cli.skip_extra_eval)
        return

    # 数据
    train_loader = MMDataLoader(args, mode='train')
    valid_loader = MMDataLoader(args, mode='valid')
    test_loader = MMDataLoader(args, mode='test')
    log(f'train={len(train_loader.dataset)} valid={len(valid_loader.dataset)} test={len(test_loader.dataset)}')

    key = getattr(args, 'KeyEval', 'MAE')
    key_upper = {'mae': 'MAE', 'loss': 'MAE', 'f1': 'F1_nonzero', 'acc': 'Accuracy',
                 'acc2': 'Acc2_nonzero', 'acc2_nonzero': 'Acc2_nonzero',
                 'acc2_has0': 'Acc2_has0', 'f1_nonzero': 'F1_nonzero',
                 'f1_has0': 'F1_has0', 'acc7': 'Acc7', 'corr': 'Corr'}.get(key.lower(), key)
    lower_better = key_upper in ('MAE', 'Loss')
    best_metric = float('inf') if lower_better else -float('inf')

    # ---- 学习率调度（对齐 H-DCD 配方：warmup + ReduceLROnPlateau） ----
    from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
    patience = getattr(args, 'patience', 10)
    warmup_epochs = getattr(args, 'warmup_epochs', 5)

    if warmup_epochs > 0 and args.epochs > warmup_epochs:
        def lr_lambda(ep):  # ep 从 0 开始
            if ep < warmup_epochs:
                return 0.1 + 0.9 * (ep + 1) / warmup_epochs
            return 1.0
        warmup = LambdaLR(optimizer, lr_lambda)
        plateau = ReduceLROnPlateau(
            optimizer, mode='min' if lower_better else 'max',
            factor=0.5, patience=max(patience // 2, 3))
    else:
        warmup, plateau = None, None

    # ---- 早停 ----
    patience_counter = 0

    # ---- 多指标追踪：回归任务显式保存双口径 Acc2/F1 ----
    if args.task_type == 'regression':
        TRACK_METRICS = ['MAE', 'Corr', 'Acc2_nonzero', 'Acc2_has0',
                         'F1_nonzero', 'F1_has0', 'Acc7']
    else:
        TRACK_METRICS = ['Accuracy', 'F1']
    if args_cli.track_metrics is not None:
        if args.task_type != 'regression':
            raise ValueError('--track-metrics 仅适用于回归实验')
        TRACK_METRICS = args_cli.track_metrics
    best_per_metric = {k: (float('inf') if k in ('MAE', 'Loss') else -float('inf'))
                       for k in TRACK_METRICS}
    best_epoch_per_metric = {k: 0 for k in TRACK_METRICS}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, train_loader, criterion, optimizer, args, device, epoch, args.epochs)

        # 课程调度 step
        if args.model_name in ('umrm', 'umrmv2', 'umrmv3') and getattr(args, 'use_curriculum', False):
            model.curriculum_scheduler.step()

        v = evaluate(model, valid_loader, args, device)
        vm = v[key_upper]
        dt = time.time() - t0

        # 更新各指标最佳（并独立保存该指标最优 checkpoint）
        for k in TRACK_METRICS:
            if k in v:
                if k in ('MAE', 'Loss'):
                    if v[k] < best_per_metric[k]:
                        best_per_metric[k] = v[k]
                        best_epoch_per_metric[k] = epoch
                        payload = {'model': model.state_dict(), 'args': vars(args)}
                        torch.save(payload, f'checkpoints/{tag}_best_{k}.pth')
                        if k == 'Acc2_nonzero':
                            torch.save(payload, f'checkpoints/{tag}_best_Acc2.pth')
                        elif k == 'F1_nonzero':
                            torch.save(payload, f'checkpoints/{tag}_best_F1.pth')
                else:
                    if v[k] > best_per_metric[k]:
                        best_per_metric[k] = v[k]
                        best_epoch_per_metric[k] = epoch
                        payload = {'model': model.state_dict(), 'args': vars(args)}
                        torch.save(payload, f'checkpoints/{tag}_best_{k}.pth')
                        if k == 'Acc2_nonzero':
                            torch.save(payload, f'checkpoints/{tag}_best_Acc2.pth')
                        elif k == 'F1_nonzero':
                            torch.save(payload, f'checkpoints/{tag}_best_F1.pth')

        # 主指标（KeyEval）选优 + 保存
        improved = (lower_better and vm < best_metric) or (not lower_better and vm > best_metric)
        if improved:
            best_metric = vm
            patience_counter = 0
            torch.save({'model': model.state_dict(), 'args': vars(args)}, ckpt_path)
        else:
            patience_counter += 1

        # LR 调度
        if warmup is not None and epoch <= warmup_epochs:
            warmup.step()
        elif plateau is not None:
            plateau.step(vm)

        lr_now = optimizer.param_groups[0]['lr']
        # 多指标日志
        extra = ' '.join(f'{k}={v[k]:.4f}' for k in TRACK_METRICS if k in v and k != key_upper)
        log(f'Epoch {epoch:3d}/{args.epochs} | loss {loss:.4f} | val {key_upper} {vm:.4f} | {extra}'
            f'{" *" if improved else ""} | lr {lr_now:.2e} | {dt:.1f}s')

        if patience_counter >= patience:
            log(f'早停触发（{patience} epochs 无提升），终止训练')
            break

    log(f'最佳 val {key_upper}: {best_metric:.4f} -> {ckpt_path}')
    log('验证集各指标最佳: ' + ' | '.join(
        f'{k}={best_per_metric[k]:.4f}@ep{best_epoch_per_metric[k]}' for k in TRACK_METRICS))

    # 测试集 7 组合评估（主指标最优 checkpoint）
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'])
    results = evaluate_combos(model, test_loader, args, device)
    out_path = f'results/{tag}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    log(f'测试集 7 组合结果（ckpt={key_upper} 最优） -> {out_path}')
    for name, m in results.items():
        log(f'  {name}: {m}')

    # ---- 各指标最优 checkpoint 分别验证（回答"用哪个 ckpt 验证最好"） ----
    for metric in TRACK_METRICS:
        if metric == key_upper:
            continue
        p = f'checkpoints/{tag}_best_{metric}.pth'
        if not os.path.exists(p):
            continue
        state = torch.load(p, map_location=device)
        model.load_state_dict(state['model'])
        res = evaluate_combos(model, test_loader, args, device)
        with open(f'results/{tag}_ckpt_{metric}.json', 'w') as f:
            json.dump(res, f, indent=2)
        s = res.get('summary', {})
        log(f'--- ckpt 按 {metric} 最优（val {metric}={best_per_metric[metric]:.4f}@ep{best_epoch_per_metric[metric]}） ---')
        for nm, mm in res.items():
            if nm == 'summary':
                log(f'  summary: {mm}')
                break
        log(f'  TAV: {res.get("TAV", {})}')

    # 各指标评测循环会依次加载其他 checkpoint；额外评测前必须恢复主 checkpoint，
    # 避免 Token 缺失与联合网格意外沿用循环中最后一个（通常为 Acc7）模型。
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'] if 'model' in state else state)
    log(f'额外评测已恢复主 checkpoint: {ckpt_path}')

    # Token 缺失曲线 + 模态×Token 联合网格（默认自动，用主 checkpoint）
    _run_extra_evals(model, test_loader, args, device, tag, log, args_cli.skip_extra_eval)


if __name__ == '__main__':
    main()
