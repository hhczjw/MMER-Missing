#!/usr/bin/env bash
# ============================================================================
# MMER BSSM 门控 A/B 实验队列 (2026-09-18)
#   Job1: BSSM 门控开启 (--bssm-gate)
#   Job2: 同代码基线复核 (无门控, 同种子同参数)
# 对照: 历史 seed42 基线 (2026-09-18 03:13, 已备份 _archive_baseline_seed42_20260918/)
# GPU0 串行 (GPU1 正在跑 H-DCD P0 队列)
# 用法: bash scripts/run_bssm_ab.sh
# ============================================================================
set -e
source /home/zjw/anaconda3/etc/profile.d/conda.sh
set +u; conda activate coupled_mamba_3; set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/mamba${PYTHONPATH:+:$PYTHONPATH}"   # set -u 安全展开
export CUDA_VISIBLE_DEVICES=0

COMMON=(--model umrmv3 --config configs/umrm_v3_config.json --dataset MOSI
  --epochs 60 --batch-size 8 --seed 42
  --kd --kd-weight 0.3 --text-recon --text-inject --bi --num-layers 2
  --bert-model FacebookAI/roberta-large --bert-pretokenized
  --feature-path CMU-MOSI/Processed/unaligned_50_roberta.pkl
  --lora-bert --uni --uni-weight 0.1 --gap-weight 0.05
  --rcd --rcd-weight 0.1 --cd-kd --balanced-missing --combo-heads
  --skip-extra-eval)

echo "=============================================="
echo "[$(date '+%F %T')] Job1 START: BSSM gate ON (seed42)"
echo "=============================================="
python scripts/train.py "${COMMON[@]}" --bssm-gate

echo "=============================================="
echo "[$(date '+%F %T')] Job2 START: baseline no-gate (seed42)"
echo "=============================================="
python scripts/train.py "${COMMON[@]}"

echo "[$(date '+%F %T')] ALL DONE"
