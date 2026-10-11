#!/usr/bin/env bash
# Acc7 口径改为 clip+round 后重训 seed42-48，重新按新口径选 best_Acc7.pth。
# 训练参数与 references/BEST_CONFIG.md 旧最佳完全一致（KeyEval=MAE 默认），
# 唯一变量：TRACK_METRICS['Acc7'] 现在是 clip+round 口径（utils/metrics.py）。
# 每个 seed 训完即把 ckpt 归档到外置盘（本地磁盘仅剩 ~65G，装不下 7 份 × 10 文件）。
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/mamba${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/zjw/anaconda3/envs/coupled_mamba_3/bin/python
ARCH=/media/zjw/951FB31A9E1EB7E0/MMER-Missing-checkpoints/retrain_20261005_acc7clip
mkdir -p "$ARCH"

GPU_LIST="${1:-0}"
IFS=',' read -ra GPUS <<< "$GPU_LIST"

# ---- 0) 完成昨日（bins口径）ckpt 向外置盘的搬移 ----
OLD=/media/zjw/951FB31A9E1EB7E0/MMER-Missing-checkpoints/retrain_20261005_binsAcc7_seed42-48
mkdir -p "$OLD"
if ls checkpoints/*_seed4[2-8]_*.pth >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] 继续搬移昨日 ckpt -> $OLD"
  mv checkpoints/*_seed4[2-8]_*.pth "$OLD"/
fi
echo "[$(date '+%F %T')] 搬移完成，本地剩余: $(ls checkpoints/*_seed4[2-8]_*.pth 2>/dev/null | wc -l)"

train_one() {
  local SEED=$1 GPU=$2
  echo "===== [$(date '+%F %T')] 训练 seed=${SEED} @ GPU${GPU} ====="
  CUDA_VISIBLE_DEVICES=$GPU "$PY" scripts/train.py --model umrmv3 --config configs/umrm_v3_config.json \
    --dataset MOSI --epochs 60 --batch-size 8 --seed "${SEED}" \
    --kd --kd-weight 0.3 --text-recon --text-inject --bi --num-layers 2 \
    --bert-model FacebookAI/roberta-large --bert-pretokenized \
    --feature-path CMU-MOSI/Processed/unaligned_50_roberta.pkl \
    --lora-bert --uni --uni-weight 0.1 --gap-weight 0.05 \
    --rcd --rcd-weight 0.1 --cd-kd --balanced-missing --combo-heads
  echo "[$(date '+%F %T')] 归档 seed=${SEED} ckpt -> $ARCH"
  mv checkpoints/*_seed"${SEED}"_*.pth "$ARCH"/
}

# ---- 1) 双 GPU 并行：GPU0=42-45, GPU1=46-48 ----
if [ "${#GPUS[@]}" -ge 2 ]; then
  ( for S in 42 43 44 45; do train_one "$S" "${GPUS[0]}"; done ) &
  PID0=$!
  ( for S in 46 47 48; do train_one "$S" "${GPUS[1]}"; done ) &
  PID1=$!
  wait $PID0 $PID1
else
  for S in 42 43 44 45 46 47 48; do train_one "$S" "${GPUS[0]}"; done
fi
echo "===== [$(date '+%F %T')] 7-seed Acc7clip 重训全部完成 ====="
