#!/usr/bin/env python3
"""
MOSI 音频 wav2vec2-large 特征提取
=================================
读 aligned_50.pkl 的 id 顺序，对每条 wav 提取 wav2vec2-large-960h 特征，
adaptive avg pool 到 50 帧（词级对齐），替换 audio 字段为 (N,50,1024)。

用法:
    python scripts/build_wav2vec_pkl.py \
        --input /media/.../CMU-MOSI/Processed/aligned_50.pkl \
        --wav-dir /media/.../CMU-MOSI/Raw/Audio \
        --output /media/.../CMU-MOSI/Processed/aligned_50_w2v.pkl
"""

import argparse
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--wav-dir', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--model', default='facebook/wav2vec2-large-960h')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--pool', type=int, default=50)
    cli = ap.parse_args()

    import soundfile as sf
    from transformers import Wav2Vec2Model

    model = Wav2Vec2Model.from_pretrained(cli.model).to(cli.device).eval()

    with open(cli.input, 'rb') as f:
        data = pickle.load(f)

    for split_name in ('train', 'valid', 'test'):
        split = data[split_name]
        ids = split['id']
        N = len(ids)
        feats = np.zeros((N, cli.pool, model.config.hidden_size), dtype=np.float32)
        for i, vid in enumerate(tqdm(ids, desc=split_name)):
            # id 格式: video$_$segment（多片段）或 video
            if '$_$' in vid:
                video, seg = vid.split('$_$')
                wav_path = os.path.join(cli.wav_dir, video, f'{seg}.wav')
            else:
                wav_path = os.path.join(cli.wav_dir, vid, '1.wav')
                if not os.path.exists(wav_path):
                    wav_path = os.path.join(cli.wav_dir, f'{vid}.wav')
            try:
                x, sr = sf.read(wav_path)
            except Exception as e:
                print(f'跳过 {vid}: {e}')
                continue
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != 16000:
                # 重采样用简单线性插值（MOSI 为 16k，一般不需要）
                from scipy.signal import resample_poly
                x = resample_poly(x, 16000, sr).astype(np.float32)
            with torch.no_grad():
                t = torch.FloatTensor(x).unsqueeze(0).to(cli.device)
                h = model(t).last_hidden_state  # (1, T, 1024)
            h = h.transpose(1, 2)  # (1, 1024, T)
            hp = torch.nn.functional.adaptive_avg_pool1d(h, cli.pool).squeeze(0).transpose(0, 1)  # (50, 1024)
            feats[i] = hp.cpu().numpy()
        split['audio'] = feats
        split['w2v_audio'] = feats  # 保留别名字段
        print(f'[{split_name}] audio -> (N, {cli.pool}, {model.config.hidden_size})')

    with open(cli.output, 'wb') as f:
        pickle.dump(data, f)
    print(f'已保存 {cli.output}')

    # 冒烟
    with open(cli.output, 'rb') as f:
        chk = pickle.load(f)
    print('验证 audio shape:', np.asarray(chk['valid']['audio']).shape)


if __name__ == '__main__':
    main()
