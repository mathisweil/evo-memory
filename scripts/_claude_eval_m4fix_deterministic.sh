#!/bin/bash
set -u
IDX="${1:?usage: $0 <0|1|2>}"
PROJ=/cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo
ENV=/cs/student/project_msc/2025/csml/rhautier/envs/th2
cd "$PROJ"
source /cs/student/project_msc/2025/csml/rhautier/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV"
export CUDA_VISIBLE_DEVICES=0

NAMM="eval_results/namm_cs1024_maskfix/ckpt.pt"
LORA="checkpoints_backup/lora_m4_cs1024_maskfix/best_ckpt_step260_val52.06.pt"

case "$IDX" in
  0)
    # M4 maskfix + NAMM, batch_size=1 for determinism
    python scripts/eval_namm_splits.py \
        --namm_checkpoint "$NAMM" \
        --lora_checkpoint "$LORA" \
        --filter_by_length 8192 --cache_size 1024 --batch_size 1 \
        --splits test extended_test --run_label ext_maskfix_bs1 \
        --output_dir eval_results/lora_m4_cs1024_maskfix_5t 2>&1
    ;;
  1)
    # A4 maskfix ablation, batch_size=1
    python scripts/eval_namm_splits.py \
        --lora_checkpoint "$LORA" \
        --filter_by_length 8192 --cache_size 8192 --batch_size 1 \
        --splits test extended_test --run_label ext_maskfix_ablation_bs1 \
        --output_dir eval_results/lora_m4_cs1024_maskfix_ablation_5t 2>&1
    ;;
  2)
    # Val split check
    python scripts/eval_namm_splits.py \
        --namm_checkpoint "$NAMM" \
        --lora_checkpoint "$LORA" \
        --filter_by_length 8192 --cache_size 1024 --batch_size 1 \
        --splits val --run_label val_bs1 \
        --output_dir eval_results/lora_m4_cs1024_maskfix_5t 2>&1
    ;;
esac
