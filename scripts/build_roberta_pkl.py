#!/usr/bin/env python3
"""
离线生成 RoBERTa 三通道 text_bert pkl
========================================

将 BERT 三通道 [ids, mask, segment] 转换为 RoBERTa 三通道 [ids, mask, ones]，
保持每个样本的序列长度 L 与原 pkl 一致（下游以 L 对齐音/视觉帧数）。

用法:
    python scripts/build_roberta_pkl.py \
        --input /media/.../CMU-MOSI/Processed/unaligned_50.pkl \
        --output /media/.../CMU-MOSI/Processed/unaligned_50_roberta.pkl \
        --model FacebookAI/roberta-base
"""

import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, default='FacebookAI/roberta-base')
    parser.add_argument('--bert-tokenizer', type=str, default='bert-base-uncased')
    args = parser.parse_args()

    from transformers import AutoTokenizer, BertTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    bert_tok = BertTokenizer.from_pretrained(args.bert_tokenizer)

    print(f'读取 {args.input} ...')
    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    # 识别 split 结构
    splits = []
    if 'train' in data and isinstance(data['train'], dict):
        splits = [k for k in ('train', 'valid', 'test') if k in data]
        print(f'嵌套结构，splits: {splits}')
    else:
        splits = [None]
        print('扁平结构（单 split）')

    for split_name in splits:
        split = data[split_name] if split_name else data
        text_bert = split['text_bert']
        arr = np.asarray(text_bert, dtype=np.int64)
        N, C, L = arr.shape
        assert C == 3, f'text_bert 通道数异常: {C}'
        print(f'[{split_name or "all"}] N={N}, L={L}')

        new_arr = np.zeros((N, 3, L), dtype=np.int64)
        for i in tqdm(range(N), desc=f'编码 {split_name or "all"}'):
            ids = arr[i, 0].tolist()
            raw = bert_tok.decode(ids, skip_special_tokens=True)
            enc = tok(raw, padding='max_length', truncation=True,
                      max_length=L, return_tensors='np')
            new_arr[i, 0] = enc['input_ids'][0]
            new_arr[i, 1] = enc['attention_mask'][0]
            new_arr[i, 2] = 1  # RoBERTa 无 token_type_ids，恒为 1

        split['text_bert'] = new_arr

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    print(f'已保存 {args.output}')

    # 冒烟验证
    with open(args.output, 'rb') as f:
        chk = pickle.load(f)
    v = chk['valid'] if 'valid' in chk else chk
    print('验证: text_bert shape =', np.asarray(v['text_bert']).shape,
          'dtype =', np.asarray(v['text_bert']).dtype)


if __name__ == '__main__':
    main()
