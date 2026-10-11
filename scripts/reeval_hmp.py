#!/usr/bin/env python3
"""
Tier2-15 验证: 推理侧硬缺失掩码池化（训练未启用，仅评测时清零缺失支路池化）
=============================================================================
用已有 checkpoint，评测时对缺失模态支路池化清零，看 A/V/AV 组合是否回升。
用法:
    python scripts/reeval_hmp.py --config configs/umrm_v3_config.json \
        --dataset MOSI --checkpoint checkpoints/<tag>_best_Acc2.pth \
        --bert-model FacebookAI/roberta-base --bert-pretokenized \
        --feature-path CMU-MOSI/Processed/unaligned_50_roberta.pkl \
        --uni --rcd --cd-kd --bi
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import _path_setup  # noqa: F401,E402
from config import load_config
from dataset.data_loader import MMDataLoader
from models import UMRMv3
from utils.metrics import eval_regression
from utils.seed import set_seed

COMBOS = [
    ('TAV', ['text', 'audio', 'vision']),
    ('TA', ['text', 'audio']),
    ('TV', ['text', 'vision']),
    ('AV', ['audio', 'vision']),
    ('T', ['text']),
    ('A', ['audio']),
    ('V', ['vision']),
]
MOD_IDX = {'text': 0, 'audio': 1, 'vision': 2}


@torch.no_grad()
def collect_preds(model, loader, args, device):
    model.eval()
    preds = {name: [] for name, _ in COMBOS}
    labels = []
    use_bert = getattr(model, 'use_bert', False)
    for batch in loader:
        text = batch['text'].to(device)
        audio = batch['audio'].to(device)
        vision = batch['vision'].to(device)
        label = batch['label'].to(device)
        B = text.size(0)
        labels.append(label.cpu().numpy())
        for name, mods in COMBOS:
            if 'text' in mods:
                t = text
            else:
                tok = getattr(model.bert_encoder, 'tokenizer', None)
                pad_id = tok.pad_token_id if (tok is not None and tok.pad_token_id is not None) else 0
                t = (torch.full(text.size(), pad_id, device=device, dtype=torch.long)
                     if use_bert else torch.zeros_like(text))
            a = audio if 'audio' in mods else torch.zeros_like(audio)
            v = vision if 'vision' in mods else torch.zeros_like(vision)
            mm = torch.zeros(B, 3, device=device, dtype=torch.bool)
            for m in mods:
                mm[:, MOD_IDX[m]] = True
            out = model(t, a, v, modality_mask=mm, token_mask=None)
            preds[name].append(out['logits'].squeeze(-1).cpu().numpy())
    labels = np.concatenate(labels).flatten()
    return {k: np.concatenate(v).flatten() for k, v in preds.items()}, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='MOSI')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bert-model', type=str, default=None)
    parser.add_argument('--bert-pretokenized', action='store_true')
    parser.add_argument('--feature-path', type=str, default=None)
    parser.add_argument('--uni', action='store_true')
    parser.add_argument('--uni-weight', type=float, default=0.1)
    parser.add_argument('--gap-weight', type=float, default=0.05)
    parser.add_argument('--rcd', action='store_true')
    parser.add_argument('--rcd-weight', type=float, default=0.1)
    parser.add_argument('--cd-kd', action='store_true')
    parser.add_argument('--bi', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config, args.dataset)
    cfg.seed = args.seed
    if args.bert_model:
        cfg.bert_pretrained = args.bert_model
    if args.bert_pretokenized:
        cfg.bert_pretokenized = True
    if args.feature_path:
        cfg.featurePath = args.feature_path
    if args.uni:
        cfg.use_uni = True
        cfg.uni_weight = args.uni_weight
        cfg.gap_weight = args.gap_weight
    if args.rcd:
        cfg.use_rcd = True
        cfg.rcd_weight = args.rcd_weight
    if args.cd_kd:
        cfg.use_cd_kd = True
    if args.bi:
        cfg.use_bi_direction = True
    cfg.use_text_recon = True
    cfg.use_text_injection = True
    cfg.use_kd = True
    cfg.kd_weight = 0.3
    cfg.use_hard_mask_pool = True  # 推理侧硬掩码

    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UMRMv3(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state['model'] if 'model' in state else state)
    model.eval()
    print(f'已加载 {args.checkpoint}')

    test_loader = MMDataLoader(cfg, mode='test')
    preds, labels = collect_preds(model, test_loader, cfg, device)

    acc = {k: eval_regression(preds[k], labels) for k, _ in COMBOS}
    miss_names = [k for k, _ in COMBOS if k != 'TAV']
    miss_acc2 = float(np.mean([acc[k]['Acc2_nonzero'] for k in miss_names]))
    miss_acc2_has0 = float(np.mean([acc[k]['Acc2_has0'] for k in miss_names]))
    miss_f1 = float(np.mean([acc[k]['F1_nonzero'] for k in miss_names]))
    miss_f1_has0 = float(np.mean([acc[k]['F1_has0'] for k in miss_names]))
    miss_acc7 = float(np.mean([acc[k]['Acc7'] for k in miss_names]))
    miss_acc7_bins = float(np.mean([acc[k]['Acc7_bins'] for k in miss_names]))
    miss_mae = float(np.mean([acc[k]['MAE'] for k in miss_names]))
    worst = min(acc[k]['Acc2_nonzero'] for k in miss_names)
    print('组合      Acc2 Non0  Acc2 Has0   F1 Non0    F1 Has0    Acc7     MAE')
    for k, _ in COMBOS:
        a = acc[k]
        print(f'{k:5s}    {a["Acc2_nonzero"]:.4f}      {a["Acc2_has0"]:.4f}      '
              f'{a["F1_nonzero"]:.4f}      {a["F1_has0"]:.4f}      '
              f'{a["Acc7"]:.4f}  {a["MAE"]:.4f}')
    print(f'缺失均值: Acc2 Non0 {miss_acc2:.4f} | Acc2 Has0 {miss_acc2_has0:.4f} | '
          f'F1 Non0 {miss_f1:.4f} | F1 Has0 {miss_f1_has0:.4f} | '
          f'Acc7 {miss_acc7:.4f} | MAE {miss_mae:.4f} | worst Non0 {worst:.4f}')
    out_path = args.checkpoint.replace('.pth', '_hmp_reeval.json').replace('checkpoints/', 'results/')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({k: acc[k] for k, _ in COMBOS} | {
        'missing_avg_Acc2': miss_acc2,
        'missing_avg_Acc2_nonzero': miss_acc2,
        'missing_avg_Acc2_has0': miss_acc2_has0,
        'missing_avg_F1_nonzero': miss_f1,
        'missing_avg_F1_has0': miss_f1_has0,
        'missing_avg_Acc7': miss_acc7,
        'missing_avg_Acc7_bins': miss_acc7_bins,
        'missing_worst_Acc2': worst,
        'missing_worst_Acc2_nonzero': worst,
        'missing_avg_MAE': miss_mae},
        open(out_path, 'w'), indent=2)
    print('已保存', out_path)


if __name__ == '__main__':
    main()
