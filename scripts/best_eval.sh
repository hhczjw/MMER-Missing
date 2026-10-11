#!/usr/bin/env bash
# 一键复现"当时最好效果"：最终架构（RoBERTa-large LoRA + 弱权 + bal + 组合头）
# 用法: bash scripts/best_eval.sh
set -e
source /home/zjw/anaconda3/etc/profile.d/conda.sh
conda activate coupled_mamba_3
cd "$(dirname "$0")/.."

P="checkpoints/umrmv3_kd0.3_recon_inj_rbl_rbp_lora_bi_uni0.1_g0.05_rcd0.1_cdkd_bal_ch_MOSI_seed"
N5_MAE=$(for s in 42 43 44 45 46; do echo "$P${s}_best_MAE.pth"; done)
N5_CORR=$(for s in 42 43 44 45 46; do echo "$P${s}_best_Corr.pth"; done)
N7_ACC2=$(for s in 42 43 44 45 46 47 48; do echo "$P${s}_best_Acc2.pth"; done)

COMMON=(--config configs/umrm_v3_config.json --dataset MOSI
  --bert-model FacebookAI/roberta-large --bert-pretokenized
  --feature-path CMU-MOSI/Processed/unaligned_50_roberta.pkl
  --lora-bert --uni --gap-weight 0.05 --rcd --cd-kd --balanced-missing --combo-heads --bi)

echo "========== ① 完整模态最优（5-seed MAE-ckpt 集成）=========="
CUDA_VISIBLE_DEVICES=0 python scripts/reeval_ensemble.py "${COMMON[@]}" --checkpoints $N5_MAE

echo "========== ② 缺失组合最优（5-seed Corr-ckpt 集成）=========="
CUDA_VISIBLE_DEVICES=0 python scripts/reeval_ensemble.py "${COMMON[@]}" --checkpoints $N5_CORR

echo "========== ③ 完整 Acc2 最优（7-seed Acc2-ckpt 集成）=========="
CUDA_VISIBLE_DEVICES=0 python scripts/reeval_ensemble.py "${COMMON[@]}" --checkpoints $N7_ACC2

echo "========== ④ Token 缺失最优（基线 seed42，历史结果）=========="
python - << 'EOF'
import json
d = json.load(open('results/umrmv3_kd0.3_recon_inj_rbl_rbp_lora_bi_MOSI_seed42_tokenmissing.json'))
print('Token 全率平均:', d.get('全率平均') or d.get('summary') or list(d.keys()))
EOF
