#!/usr/bin/env python3
"""
Tier0-1: Acc2 二分类阈值后处理扫描
====================================

对已有 checkpoint：验证集上扫描分类阈值 τ ∈ [-0.4, 0.4]（步长 0.01），
以"6 缺失组合均值 Acc2"最优为目标选 τ*，测试集上报告 τ=0 与 τ* 的对比。

用法（与 train.py 同配置口径）:
    python scripts/reeval_threshold.py --config configs/umrm_v3_config.json \
        --dataset MOSI --checkpoint checkpoints/<tag>_best_Acc2.pth \
        --bert-model FacebookAI/roberta-base --bert-pretokenized \
        --feature-path CMU-MOSI/Processed/unaligned_50_roberta.pkl
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# 将项目根目录加入 sys.path（脚本位于 scripts/ 子目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import _path_setup  # noqa: F401,E402
from config import load_config
from dataset.data_loader import MMDataLoader
from models import UMRMv3
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


def acc2_at(preds, labels, tau):
    return float(np.mean((labels >= 0) == (np.asarray(preds) >= tau)))


@torch.no_grad()
def collect_preds(model, loader, args, device):
    model.eval()
    preds = {name: [] for name, _ in COMBOS}
    labels = []
    for batch in loader:
        text = batch['text'].to(device)
        audio = batch['audio'].to(device)
        vision = batch['vision'].to(device)
        label = batch['label'].to(device)
        B = text.size(0)
        labels.append(label.cpu().numpy())
        use_bert = getattr(model, 'use_bert', False)
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--bert-model', default=None)
    ap.add_argument('--bert-pretokenized', action='store_true')
    ap.add_argument('--feature-path', default=None)
    ap.add_argument('--num-layers', type=int, default=None)
    ap.add_argument('--bi', action='store_true')
    ap.add_argument('--uaf', action='store_true')
    ap.add_argument('--uni', action='store_true')
    ap.add_argument('--uni-weight', type=float, default=0.1)
    ap.add_argument('--gap-weight', type=float, default=0.05)
    ap.add_argument('--rcd', action='store_true')
    ap.add_argument('--rcd-weight', type=float, default=0.1)
    ap.add_argument('--cd-kd', action='store_true')
    cli = ap.parse_args()

    args = load_config(cli.config, cli.dataset)
    args.seed = cli.seed
    if cli.bert_model: args.bert_pretrained = cli.bert_model
    if cli.bert_pretokenized: args.bert_pretokenized = True
    if cli.feature_path: args.featurePath = cli.feature_path
    if cli.num_layers is not None: args.num_layers = cli.num_layers
    if cli.bi: args.use_bi_direction = True
    if cli.uaf: args.use_uaf = True
    if cli.uni:
        args.use_uni = True
        args.uni_weight = cli.uni_weight
        args.gap_weight = cli.gap_weight
    if cli.rcd:
        args.use_rcd = True
        args.rcd_weight = cli.rcd_weight
    if cli.cd_kd: args.use_cd_kd = True
    args.use_text_recon = True
    args.use_text_injection = True
    args.use_kd = True
    args.kd_weight = 0.3

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UMRMv3(args).to(device)
    state = torch.load(cli.checkpoint, map_location=device)
    model.load_state_dict(state['model'] if 'model' in state else state)
    model.eval()
    print(f'已加载 {cli.checkpoint}')

    valid_loader = MMDataLoader(args, mode='valid')
    test_loader = MMDataLoader(args, mode='test')

    vp, vl = collect_preds(model, valid_loader, args, device)
    tp, tl = collect_preds(model, test_loader, args, device)

    missing_names = [n for n, _ in COMBOS if n != 'TAV']
    taus = np.arange(-0.40, 0.41, 0.01)
    best_tau, best_val = 0.0, -1.0
    for tau in taus:
        accs = [acc2_at(vp[n], vl, tau) for n in missing_names]
        m = float(np.mean(accs))
        if m > best_val:
            best_val, best_tau = m, float(tau)
    print(f'验证集扫描: 最优 τ* = {best_tau:.2f} (6 缺失组合均值 Acc2 = {best_val:.4f})')

    report = {'best_tau': best_tau, 'combos': {}}
    print(f'{"组合":6s} {"τ=0 Acc2":>10s} {"τ* Acc2":>10s} {"Acc7":>7s} {"MAE":>7s}')
    for name, _ in COMBOS:
        p, y = tp[name], tl
        from utils.metrics import eval_regression
        m0 = eval_regression(p, y)
        a0 = acc2_at(p, y, 0.0)
        a1 = acc2_at(p, y, best_tau)
        report['combos'][name] = {'Acc2_t0': round(a0, 4), 'Acc2_tau': round(a1, 4),
                                  'Acc7': m0['Acc7'], 'MAE': m0['MAE']}
        print(f'{name:6s} {a0:10.4f} {a1:10.4f} {m0["Acc7"]:7.4f} {m0["MAE"]:7.4f}')

    avg0 = float(np.mean([report['combos'][n]['Acc2_t0'] for n in missing_names]))
    avg1 = float(np.mean([report['combos'][n]['Acc2_tau'] for n in missing_names]))
    report['missing_avg_Acc2_t0'] = round(avg0, 4)
    report['missing_avg_Acc2_tau'] = round(avg1, 4)
    report['TAV_Acc2_t0'] = report['combos']['TAV']['Acc2_t0']
    report['TAV_Acc2_tau'] = report['combos']['TAV']['Acc2_tau']
    print(f'缺失均值 Acc2: τ=0 {avg0:.4f} → τ* {avg1:.4f} (Δ {avg1-avg0:+.4f})')

    out_path = cli.checkpoint.replace('.pth', '_threshold.json').replace('checkpoints/', 'results/')
    os.makedirs('results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'已保存 {out_path}')


if __name__ == '__main__':
    main()
