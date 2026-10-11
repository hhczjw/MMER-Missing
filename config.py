"""
配置加载器：读取 configs/*.json 并合并为扁平 SimpleNamespace
=============================================================

合并优先级：datasets[dataset] > training > model > 顶层
（dataset 级配置覆盖 model 级与全局级）
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Dict


def _flatten(d: Dict, prefix: str = '') -> Dict:
    """将嵌套 dict 展平为 {'model.d_model': 128, ...}"""
    out = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def load_config(config_path: str, dataset: str = 'MOSI') -> SimpleNamespace:
    """
    加载配置并合并：
    - 顶层键（dataset_root_dir 等）
    - model.*（模型超参）
    - training.*（训练超参）
    - datasets[dataset].*（数据集级，最高优先级）

    Args:
        config_path: json 路径
        dataset: 数据集名（MOSI / MOSEI / IEMOCAP）
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'配置文件不存在: {config_path}')

    with open(config_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    merged: Dict = {}

    # 1. 顶层（排除 section）
    for k, v in raw.items():
        if not isinstance(v, dict):
            merged[k] = v

    # 2. model section
    if 'model' in raw:
        merged.update(raw['model'])

    # 3. training section
    if 'training' in raw:
        merged.update(raw['training'])

    # 4. dataset 级（最高优先级）
    ds_cfg = raw.get('datasets', {}).get(dataset, {})
    if not ds_cfg:
        available = list(raw.get('datasets', {}).keys())
        raise KeyError(f'未知数据集 {dataset}，可用: {available}')
    merged.update(ds_cfg)
    merged['dataset'] = dataset

    return SimpleNamespace(**merged)


def load_config_from_flags(config_path: str, dataset: str = 'MOSI',
                           overrides: Dict = None) -> SimpleNamespace:
    """加载配置并应用命令行覆盖"""
    args = load_config(config_path, dataset)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                setattr(args, k, v)
    return args
