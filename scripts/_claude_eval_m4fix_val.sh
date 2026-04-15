#!/bin/bash
set -u
PROJ=/cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo
ENV=/cs/student/project_msc/2025/csml/rhautier/envs/th2
cd "$PROJ"
source /cs/student/project_msc/2025/csml/rhautier/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV"
export CUDA_VISIBLE_DEVICES=0
python scripts/eval_namm_splits.py \
    --namm_checkpoint eval_results/namm_cs1024_maskfix/ckpt.pt \
    --lora_checkpoint checkpoints_backup/lora_m4_cs1024_maskfix/best_ckpt_step260_val52.06.pt \
    --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
    --splits val \
    --run_label val_check \
    --output_dir eval_results/lora_m4_cs1024_maskfix_5t 2>&1
