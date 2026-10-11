#!/usr/bin/env bash
# GPU0: 等 seed44(PID 592746) 结束 -> 归档44 -> 训 43 -> 45 -> 42
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/mamba${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/zjw/anaconda3/envs/coupled_mamba_3/bin/python
ARCH=/media/zjw/951FB31A9E1EB7E0/MMER-Missing-checkpoints/retrain_20261005_acc7clip
mkdir -p "$ARCH"
while kill -0 592746 2>/dev/null; do sleep 30; done
echo "===== [$(date '+%F %T')] seed44 结束，归档 ====="
mv checkpoints/*_seed44_*.pth "$ARCH"/ 2>/dev/null || true
for SEED in 43 45 42; do
  echo "===== [$(date '+%F %T')] 训练 seed=${SEED} @GPU0 ====="
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train.py --model umrmv3 --config configs/umrm_v3_config.json \
    --dataset MOSI --epochs 60 --batch-size 8 --seed "${SEED}" \
    --kd --kd-weight 0.3 --text-recon --text-inject --bi --num-layers 2 \
    --bert-model FacebookAI/roberta-large --bert-pretokenized \
    --feature-path CMU-MOSI/Processed/unaligned_50_roberta.pkl \
    --lora-bert --uni --uni-weight 0.1 --gap-weight 0.05 \
    --rcd --rcd-weight 0.1 --cd-kd --balanced-missing --combo-heads
  echo "[$(date '+%F %T')] 归档 seed=${SEED}"
  mv checkpoints/*_seed"${SEED}"_*.pth "$ARCH"/
done
echo "===== [$(date '+%F %T')] GPU0 完成 43/45/42 ====="
