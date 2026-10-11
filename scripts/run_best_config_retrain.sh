#!/usr/bin/env bash
# 按 references/BEST_CONFIG.md「当时最好效果」的训练参数原样重训（5-seed: 42-46）。
# 与 2026-10-05 统一口径 run（seed51）的差别仅两处：
#   1) 不传 --key-eval → 用配置默认 KeyEval=MAE 选 ckpt / 早停（旧最佳协议）
#   2) seed 用 42-46（历史最佳 5-seed 集成的成员 seed）
# 其余参数与 BEST_CONFIG.md 训练命令逐字一致。
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/mamba${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
# 用环境解释器绝对路径（项目约定，避免 conda activate 在脚本内不生效）
PY=/home/zjw/anaconda3/envs/coupled_mamba_3/bin/python

for SEED in 42 43 44 45 46; do
  echo "===== [$(date '+%F %T')] 训练 seed=${SEED} ====="
  "$PY" scripts/train.py --model umrmv3 --config configs/umrm_v3_config.json \
    --dataset MOSI --epochs 60 --batch-size 8 --seed "${SEED}" \
    --kd --kd-weight 0.3 --text-recon --text-inject --bi --num-layers 2 \
    --bert-model FacebookAI/roberta-large --bert-pretokenized \
    --feature-path CMU-MOSI/Processed/unaligned_50_roberta.pkl \
    --lora-bert --uni --uni-weight 0.1 --gap-weight 0.05 \
    --rcd --rcd-weight 0.1 --cd-kd --balanced-missing --combo-heads
done
echo "===== [$(date '+%F %T')] 5-seed 全部完成 ====="
