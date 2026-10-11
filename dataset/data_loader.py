"""
数据加载器：支持标准 MSADataset pkl 格式（嵌套 train/valid/test 结构）
======================================================================

支持两种 pkl 结构：
1. 嵌套结构（标准 MSA 数据集）:
    {
        'train': {'text': (N,L,D), 'audio': ..., 'vision': ...,
                  'regression_labels': (N,), 'classification_labels': (N,), 'id': ...},
        'valid': {...}, 'test': {...}
    }
2. 扁平结构（旧式）:
    {'train_text': ..., 'train_audio': ..., 'train_labels': ..., ...}

标签选择：回归任务取 regression_labels，分类任务取 classification_labels。
自动过滤 NaN 标签样本。
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class MMDataset(Dataset):
    """多模态情感数据集（MOSI / MOSEI 等标准 MSADataset 格式）"""

    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode

        dataset_root = getattr(args, 'dataset_root_dir', '')
        feature_path = getattr(args, 'featurePath', '')
        data_file = os.path.join(dataset_root, feature_path)

        if not os.path.exists(data_file):
            raise FileNotFoundError(f'数据文件不存在: {data_file}')

        with open(data_file, 'rb') as f:
            data = pickle.load(f)

        # ---- 结构探测 ----
        if mode in data and isinstance(data[mode], dict):
            split = data[mode]
            prefix = ''
        else:
            split = data
            prefix = f'{mode}_'

        def _get(key):
            if key in split:
                return split[key]
            return split.get(f'{prefix}{key}', None)

        self.text = _get('text')
        self.audio = _get('audio')
        self.vision = _get('vision')
        self.ids = _get('id')
        self.audio_valid_lengths = _get('audio_lengths')
        self.vision_valid_lengths = _get('vision_lengths')

        # ---- BERT 三通道输入（use_bert 时优先 text_bert） ----
        self.use_bert = bool(getattr(args, 'use_bert', False))
        if self.use_bert:
            bert_feat = _get('text_bert')
            if bert_feat is not None:
                self.text_bert = bert_feat
            else:
                self.use_bert = False  # 降级为普通特征

        # ---- 标签选择 ----
        task_type = getattr(args, 'task_type', 'regression')
        if task_type == 'regression':
            labels = _get('regression_labels')
            if labels is None:
                labels = _get('labels')  # 兜底
        else:
            labels = _get('classification_labels')
            if labels is None:
                labels = _get('labels')

        if labels is None:
            raise KeyError(f'未找到标签字段（task_type={task_type}），可用字段: {list(split.keys())[:12]}')

        labels = np.asarray(labels)
        if len(labels) and isinstance(labels[0], dict):
            labels = np.asarray([lab['M'] if isinstance(lab, dict) else lab for lab in labels])

        self.labels = labels

        # ---- 过滤 NaN 标签样本（MOSEI 常见）----
        if np.issubdtype(labels.dtype, np.floating):
            valid = ~np.isnan(labels)
        else:
            valid = np.ones(len(labels), dtype=bool)
        if not valid.all():
            self.text = self.text[valid] if self.text is not None else None
            self.audio = self.audio[valid] if self.audio is not None else None
            self.vision = self.vision[valid] if self.vision is not None else None
            self.text_bert = self.text_bert[valid] if self.use_bert and self.text_bert is not None else None
            self.ids = np.asarray(self.ids)[valid] if self.ids is not None else None
            if self.audio_valid_lengths is not None:
                self.audio_valid_lengths = np.asarray(self.audio_valid_lengths)[valid]
            if self.vision_valid_lengths is not None:
                self.vision_valid_lengths = np.asarray(self.vision_valid_lengths)[valid]
            self.labels = labels[valid]

        # 兜底：缺失模态用零特征
        self.fd_text, self.fd_audio, self.fd_vision = self._infer_feature_dims()
        self.default_seq_len = int(getattr(args, 'seq_lens', [50, 50, 50])[0])

    def _infer_feature_dims(self):
        dims = [768, 74, 20]
        if self.text is not None and len(self.text):
            dims[0] = self.text[0].shape[-1]
        if self.audio is not None and len(self.audio):
            dims[1] = self.audio[0].shape[-1]
        if self.vision is not None and len(self.vision):
            dims[2] = self.vision[0].shape[-1]
        return dims

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        def _feat(x, fd, seq_len):
            if x is None:
                return torch.zeros(seq_len, fd, dtype=torch.float32)
            return torch.FloatTensor(np.asarray(x[idx], dtype=np.float32))

        if self.use_bert and self.text_bert is not None:
            text = torch.LongTensor(np.asarray(self.text_bert[idx], dtype=np.int64))  # (3, L)
        else:
            text = _feat(self.text, self.fd_text, self.default_seq_len)
        audio = _feat(self.audio, self.fd_audio, self.default_seq_len)
        vision = _feat(self.vision, self.fd_vision, self.default_seq_len)

        label = self.labels[idx]
        task_type = getattr(self.args, 'task_type', 'regression')
        if task_type == 'regression':
            label_t = torch.FloatTensor([float(label)])
        else:
            label_t = torch.LongTensor([int(label)])

        sample_id = self.ids[idx] if self.ids is not None else str(idx)
        return {
            'text': text, 'audio': audio, 'vision': vision,
            'label': label_t, 'id': sample_id,
            'audio_valid_length': min(int(self.audio_valid_lengths[idx]), audio.size(0)) if self.audio_valid_lengths is not None else audio.size(0),
            'vision_valid_length': min(int(self.vision_valid_lengths[idx]), vision.size(0)) if self.vision_valid_lengths is not None else vision.size(0),
        }


def mm_collate(batch):
    """batch 内 padding（各模态序列长度可能不一致）"""
    texts = [b['text'] for b in batch]
    audios = [b['audio'] for b in batch]
    visions = [b['vision'] for b in batch]
    labels = torch.stack([b['label'] for b in batch])
    ids = [b['id'] for b in batch]

    # 序列长度（padding 前）
    audio_lengths = torch.LongTensor([a.size(0) for a in audios])
    vision_lengths = torch.LongTensor([v.size(0) for v in visions])
    audio_valid_lengths = torch.LongTensor([b.get('audio_valid_length', b['audio'].size(0)) for b in batch])
    vision_valid_lengths = torch.LongTensor([b.get('vision_valid_length', b['vision'].size(0)) for b in batch])

    # 文本：三通道 BERT 格式 (3, L) 沿 L padding；普通特征 (L, D) 沿 L padding
    if texts[0].ndim == 2 and texts[0].size(0) == 3:
        Lmax = max(t.size(1) for t in texts)
        text_p = torch.zeros(len(texts), 3, Lmax, dtype=texts[0].dtype)
        for i, t in enumerate(texts):
            text_p[i, :, :t.size(1)] = t
    else:
        text_p = pad_sequence(texts, batch_first=True)
    audio_p = pad_sequence(audios, batch_first=True)
    vision_p = pad_sequence(visions, batch_first=True)

    return {
        'text': text_p, 'audio': audio_p, 'vision': vision_p,
        'label': labels, 'id': ids,
        'audio_lengths': audio_lengths, 'vision_lengths': vision_lengths,
        'audio_valid_lengths': audio_valid_lengths, 'vision_valid_lengths': vision_valid_lengths,
    }


def MMDataLoader(args, mode='train'):
    """创建 DataLoader"""
    dataset = MMDataset(args, mode=mode)
    batch_size = getattr(args, 'batch_size', 32)
    num_workers = getattr(args, 'num_workers', 0)
    shuffle = (mode == 'train')

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=mm_collate,
        pin_memory=True,
        drop_last=(mode == 'train'),
    )
