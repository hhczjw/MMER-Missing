#!/usr/bin/env python3
"""
Token 缺失集成评测：显式选择 feature_valid 或 input_valid_aware。
feature_valid 在 ISM 后模拟缺失，编码阶段看到完整输入；不等于输入级缺失。

同一缺失率、同一 batch 在所有 checkpoint 上复用完全相同的随机掩码；
否则平均的将是不同损坏输入上的预测，不能视为严格模型集成。
用法: python scripts/reeval_ensemble_token.py --config ... --checkpoints ckpt1.pth ckpt2.pth
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
from utils.token_protocol import validity, native_corruption, aligned_availability, feature_corruption_masks

RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def masked_batch(t, a, v, rate, use_bert, unk_id, mask_seed, rate_idx, batch_idx):
    """Create a reproducible corruption shared by every ensemble member."""
    # A separate deterministic stream per rate/batch makes results independent
    # of checkpoint order and of how many ensemble members are evaluated.
    seed = int(mask_seed + rate_idx * 1_000_003 + batch_idx * 10_007)
    gen = torch.Generator(device=t.device).manual_seed(seed)
    tt = t.clone()
    if use_bert and tt.dim() == 3:
        ids = tt[:, 0].clone()
        missing = torch.rand(ids.shape, device=t.device, generator=gen) < rate
        missing[:, 0] = False
        missing[:, -1] = False
        ids[missing] = unk_id
        tt[:, 0] = ids
    else:
        missing = torch.rand(tt[:, :, :1].shape, device=t.device, generator=gen) < rate
        missing[:, 0, :] = False
        missing[:, -1, :] = False
        tt = tt * (~missing).to(tt.dtype)
    keep_a = torch.rand(a[:, :, :1].shape, device=a.device, generator=gen) >= rate
    keep_v = torch.rand(v[:, :, :1].shape, device=v.device, generator=gen) >= rate
    keep_a[:, 0, :] = True; keep_a[:, -1, :] = True
    keep_v[:, 0, :] = True; keep_v[:, -1, :] = True
    return tt, a * keep_a.to(a.dtype), v * keep_v.to(v.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--dataset', default='MOSI')
    ap.add_argument('--split', choices=['valid', 'test'], default='test',
                    help='验证集用于选择成员，测试集仅用于最终一次报告')
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
    ap.add_argument('--mask-seed', type=int, default=20261005,
                    help='固定 Token 缺失掩码；所有 checkpoint 共用')
    ap.add_argument('--protocol', choices=['feature_valid','input_valid_aware','legacy_input'], default='feature_valid',
                    help='默认与训练位置一致的特征层缺失；输入级需显式选择 input_valid_aware')
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
    device = torch.device(cli.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; native Mamba cannot silently fall back to CPU')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    tl = MMDataLoader(args, mode=cli.split)

    batches = []
    labels = []
    ids = []
    for batch in tl:
        batches.append({k:v.to(device) for k,v in batch.items() if isinstance(v,torch.Tensor)})
        labels.append(batch['label'].cpu().numpy())
        ids.extend([str(i) for i in batch['id']])
    labels = np.concatenate(labels).flatten()

    use_bert = getattr(args, 'use_bert', True)
    unk_id = 100
    tok = None

    curve_sum = {r: None for r in RATES}
    member_results = []
    n = len(cli.checkpoints)
    archives = {'labels':labels,'sample_ids':np.asarray(ids)}
    for cp in cli.checkpoints:
        if not os.path.isfile(cp) or os.path.getsize(cp) == 0:
            raise FileNotFoundError(f'checkpoint 不存在或为空: {cp}')
        m = UMRMv3(args).to(device)
        st = torch.load(cp, map_location=device)
        m.load_state_dict(st['model'] if 'model' in st else st)
        m.eval()
        if tok is None:
            tok = getattr(m.bert_encoder, 'tokenizer', None)
            if tok is not None and tok.unk_token_id is not None:
                unk_id = tok.unk_token_id
        print(f'已加载 {cp} (unk={unk_id})')
        member_curve = {}
        for rate_idx, r in enumerate(RATES):
            preds = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(batches):
                    t,a,v = [batch[k] for k in ('text','audio','vision')]
                    tm = None
                    if cli.protocol == 'legacy_input':
                        tt,aa,vv = masked_batch(t,a,v,r,use_bert,unk_id,cli.mask_seed,rate_idx,batch_idx)
                    else:
                        valid = validity(t,a,v,batch['audio_valid_lengths'],batch['vision_valid_lengths'])
                        generator = torch.Generator(device=device).manual_seed(cli.mask_seed+rate_idx*1_000_003+batch_idx*10_007)
                        L = t.shape[-1] if use_bert else t.shape[1]
                        if cli.protocol == 'feature_valid':
                            tt,aa,vv = t,a,v
                            tm,_ = feature_corruption_masks(valid,L,r,generator)
                        else:
                            (tt,aa,vv),keeps,_ = native_corruption(t,a,v,valid,r,generator,unk_id)
                            tm = aligned_availability(keeps,L,valid)
                    o = m(tt, aa, vv, modality_mask=None, token_mask=tm)
                    preds.append(o['logits'].squeeze(-1).cpu().numpy())
            arr = np.concatenate(preds).flatten()
            archives[f'member{len(member_results)}_r{r}_preds'] = arr
            curve_sum[r] = arr if curve_sum[r] is None else curve_sum[r] + arr
            member_curve[str(r)] = {k: round(float(v), 4)
                                    for k, v in eval_regression(arr, labels).items()}
        member_avg = {metric: round(float(np.mean([
            member_curve[str(r)][metric] for r in RATES])), 4)
            for metric in ['MAE', 'Corr', 'Acc2_nonzero', 'Acc2_has0',
                           'F1_nonzero', 'F1_has0', 'Acc7', 'Acc7_bins']}
        member_results.append({'checkpoint': cp, 'per_rate': member_curve,
                               'avg_over_rates': member_avg})
        del m
        torch.cuda.empty_cache()

    curve = {}
    for r in RATES:
        archives[f'ensemble_r{r}_preds'] = curve_sum[r] / n
        met = eval_regression(curve_sum[r] / n, labels)
        curve[str(r)] = {k: round(float(v), 4) for k, v in met.items()}
        print(f'r={r}: ', {k: round(v, 4) for k, v in met.items()})
    avg = {m: round(float(np.mean([curve[str(r)][m] for r in RATES])), 4)
           for m in ['MAE', 'Corr', 'Acc2_nonzero', 'Acc2_has0',
                     'F1_nonzero', 'F1_has0', 'Acc7', 'Acc7_bins']}
    print('十率平均:', avg)
    out_path = cli.output or ('results/ensemble_token_' + os.path.basename(cli.checkpoints[0]).split('_seed')[0]
                              + f'_n{n}_{cli.protocol}_acc7_clip_round.json')
    if os.path.exists(out_path):
        raise FileExistsError(f'Refusing to overwrite existing result: {out_path}')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)),exist_ok=True)
    archive_path = os.path.splitext(out_path)[0] + '.predictions.npz'
    np.savez_compressed(archive_path,**archives)
    json.dump({'protocol': {
                   'acc7': 'clip[-3,3]+numpy.round',
                   'rates': RATES,
                   'split': cli.split,
                   'name': cli.protocol,
                   'prediction_archive': archive_path,
                   'feature_valid_note': 'Clean encoding before post-ISM dropout; not raw-input missing' if cli.protocol=='feature_valid' else '',
                   'mask_seed': cli.mask_seed,
                   'same_mask_across_checkpoints': True,
                   'mask_stream': 'mask_seed + rate_idx*1000003 + batch_idx*10007',
               },
               'per_rate': curve, 'avg_over_rates': avg,
               'members': member_results, 'checkpoints': cli.checkpoints},
              open(out_path, 'w'), indent=2)
    print('已保存', out_path)


if __name__ == '__main__':
    main()
