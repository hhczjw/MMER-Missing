#!/usr/bin/env bash
# MMER-Missing：统一双口径指标后的 GPU 0 正式训练。
set -euo pipefail

source /home/zjw/anaconda3/etc/profile.d/conda.sh
set +u
conda activate coupled_mamba_3
set -u

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/mamba${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0

python scripts/train.py \
  --model umrmv3 \
  --config configs/umrm_v3_config.json \
  --dataset MOSI \
  --epochs 60 \
  --batch-size 8 \
  --seed 51 \
  --key-eval Acc2_nonzero \
  --kd --kd-weight 0.3 \
  --text-recon --text-inject \
  --bi --num-layers 2 \
  --bert-model FacebookAI/roberta-large \
  --bert-pretokenized \
  --feature-path CMU-MOSI/Processed/unaligned_50_roberta.pkl \
  --lora-bert \
  --uni --uni-weight 0.1 --gap-weight 0.05 \
  --rcd --rcd-weight 0.1 \
  --cd-kd --balanced-missing --combo-heads
