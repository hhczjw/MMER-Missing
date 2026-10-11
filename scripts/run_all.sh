#!/bin/bash
# 三个架构完整训练：v0 (missing) / v1 (umrm) / v2 (umrmv2)
# 用法: bash scripts/run_all.sh [数据集]
set -e
cd "$(dirname "$0")/.."

DATASET=${1:-MOSI}
ENV_SETUP="source /home/zjw/anaconda3/etc/profile.d/conda.sh && conda activate coupled_mamba_3"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "############################# v0: MissingModalityMER #############################"
bash -c "$ENV_SETUP && python scripts/train.py --model missing --config configs/missing_modality_config.json --dataset $DATASET --epochs 120"

echo "############################# v1: UMRM #############################"
bash -c "$ENV_SETUP && python scripts/train.py --model umrm --config configs/umrm_config.json --dataset $DATASET --epochs 120"

echo "############################# v2: UMRMv2 #############################"
bash -c "$ENV_SETUP && python scripts/train.py --model umrmv2 --config configs/umrm_v2_config.json --dataset $DATASET --epochs 120"

echo "############################# 全部完成 #############################"
ls -la results/ logs/
