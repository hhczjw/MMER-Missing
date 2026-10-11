#!/usr/bin/env python3
"""
多 seed 集成评测：加载多个 checkpoint，平均预测后评测七组合。
用法:
    python scripts/reeval_ensemble.py --config configs/umrm_v3_config.json \
        --dataset MOSI --bert-model FacebookAI/roberta-large --bert-pretokenized \
        --feature-path CMU-MOSI/Processed/unaligned_50_roberta.pkl --lora-bert \
        --uni --uni-weight 0.1 --gap-weight 0.05 --rcd --rcd-weight 0.1 --cd-kd --balanced-missing --combo-heads --bi \
        --checkpoints ckpt1.pth ckpt2.pth ckpt3.pth
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--dataset', default='MOSI')
    ap.add_argument('--checkpoints', nargs='+', required=True)
    ap.add_argument('--bert-model', default=None)
    ap.add_argument('--bert-pretokenized', action='store_true')
    ap.add_argument('--feature-path', default=None)
    ap.add_argument('--lora-bert', action='store_true')
    ap.add_argument('--uni', action='store_true')
    ap.add_argument('--gap-weight', type=float, default=0.05)
    ap.add_argument('--rcd', action='store_true')
    ap.add_argument('--cd-kd', action='store_true')
    ap.add_argument('--balanced-missing', action='store_true')
    ap.add_argument('--combo-heads', action='store_true')
    ap.add_argument('--bi', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--output', default=None, help='输出 JSON 路径；默认另存为 clip-round 版本，不覆盖历史结果')
    cli = ap.parse_args()

    args = load_config(cli.config, cli.dataset)
    args.seed = cli.seed
    if cli.bert_model: args.bert_pretrained = cli.bert_model
    if cli.bert_pretokenized: args.bert_pretokenized = True
    if cli.feature_path: args.featurePath = cli.feature_path
    if cli.lora_bert: args.bert_finetune = 'lora'
    if cli.uni: args.use_uni = True; args.uni_weight = 0.1; args.gap_weight = cli.gap_weight
    if cli.rcd: args.use_rcd = True; args.rcd_weight = 0.1
    if cli.cd_kd: args.use_cd_kd = True
    if cli.balanced_missing: args.use_balanced_missing = True
    if cli.combo_heads: args.use_combo_heads = True
    if cli.bi: args.use_bi_direction = True
    args.use_text_recon = True
    args.use_text_injection = True
    args.use_kd = True
    args.kd_weight = 0.3

    set_seed(args.seed)
    device = torch.device(cli.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    tl = MMDataLoader(args, mode='test')
    labels = []
    batches = []
    for batch in tl:
        batches.append((batch['text'].to(device), batch['audio'].to(device),
                        batch['vision'].to(device), batch['label'].to(device)))
        labels.append(batch['label'].cpu().numpy())
    labels = np.concatenate(labels).flatten()

    tok = None
    acc = {name: None for name, _ in COMBOS}  # 累加器
    n_models = len(cli.checkpoints)
    for cp in cli.checkpoints:
        m = UMRMv3(args).to(device)
        st = torch.load(cp, map_location=device)
        m.load_state_dict(st['model'] if 'model' in st else st)
        m.eval()
        if tok is None:
            tok = m.bert_encoder.tokenizer
        print(f'已加载 {cp}')
        pad = tok.pad_token_id if tok else 0
        cur = {name: [] for name, _ in COMBOS}
        with torch.no_grad():
            for (t, a, v, _y) in batches:
                B = t.size(0)
                for name, mods in COMBOS:
                    tt = t if 'text' in mods else torch.full(t.size(), pad, dtype=torch.long, device=device)
                    aa = a if 'audio' in mods else torch.zeros_like(a)
                    vv = v if 'vision' in mods else torch.zeros_like(v)
                    mm = torch.zeros(B, 3, dtype=torch.bool, device=device)
                    for mod in mods:
                        mm[:, MOD_IDX[mod]] = True
                    o = m(tt, aa, vv, modality_mask=mm, token_mask=None)
                    cur[name].append(o['logits'].squeeze(-1).cpu().numpy())
        for name, _ in COMBOS:
            arr = np.concatenate(cur[name]).flatten()
            acc[name] = arr if acc[name] is None else acc[name] + arr
        del m
        torch.cuda.empty_cache()

    preds = {k: v / n_models for k, v in acc.items()}

    acc = {k: eval_regression(preds[k], labels) for k, _ in COMBOS}
    miss = [k for k, _ in COMBOS if k != 'TAV']
    m_acc2_nonzero = float(np.mean([acc[k]['Acc2_nonzero'] for k in miss]))
    m_acc2_has0 = float(np.mean([acc[k]['Acc2_has0'] for k in miss]))
    m_f1_nonzero = float(np.mean([acc[k]['F1_nonzero'] for k in miss]))
    m_f1_has0 = float(np.mean([acc[k]['F1_has0'] for k in miss]))
    m_acc7 = float(np.mean([acc[k]['Acc7'] for k in miss]))
    m_acc7_bins = float(np.mean([acc[k]['Acc7_bins'] for k in miss]))
    m_mae = float(np.mean([acc[k]['MAE'] for k in miss]))
    worst = min(acc[k]['Acc2_nonzero'] for k in miss)
    print('组合      Acc2 Non0  Acc2 Has0   F1 Non0    F1 Has0    Acc7      MAE      Corr')
    for k, _ in COMBOS:
        a = acc[k]
        print(f'{k:5s}    {a["Acc2_nonzero"]:.4f}      {a["Acc2_has0"]:.4f}      '
              f'{a["F1_nonzero"]:.4f}      {a["F1_has0"]:.4f}      '
              f'{a["Acc7"]:.4f}  {a["MAE"]:.4f}  {a["Corr"]:.4f}')
    print(f'缺失均值: Acc2 Non0 {m_acc2_nonzero:.4f} | Acc2 Has0 {m_acc2_has0:.4f} | '
          f'F1 Non0 {m_f1_nonzero:.4f} | F1 Has0 {m_f1_has0:.4f} | '
          f'Acc7 {m_acc7:.4f} | MAE {m_mae:.4f} | worst Non0 {worst:.4f}')
    out_path = cli.output or ('results/ensemble_' + os.path.basename(cli.checkpoints[0]).split('_seed')[0]
                              + f'_n{n_models}_acc7_clip_round.json')
    json.dump({k: acc[k] for k, _ in COMBOS} | {
        'missing_avg_Acc2': m_acc2_nonzero,
        'missing_avg_Acc2_nonzero': m_acc2_nonzero,
        'missing_avg_Acc2_has0': m_acc2_has0,
        'missing_avg_F1_nonzero': m_f1_nonzero,
        'missing_avg_F1_has0': m_f1_has0,
        'missing_avg_Acc7': m_acc7,
        'missing_avg_Acc7_bins': m_acc7_bins,
        'missing_worst_Acc2': worst,
        'missing_worst_Acc2_nonzero': worst,
        'missing_avg_MAE': m_mae,
        'checkpoints': cli.checkpoints},
        open(out_path, 'w'), indent=2)
    print('已保存', out_path)


if __name__ == '__main__':
    main()
