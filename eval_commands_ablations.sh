#!/bin/bash
# Ablation evals to run AFTER the 8 main evals finish.
# These re-test the M4 LoRA checkpoints with their NAMM eviction DISABLED
# (cache_size=8192 = filter_by_length, no eviction). They show whether the
# NAMM-trained LoRA still works without the NAMM that shaped its training,
# i.e. whether NAMM-finetuning puts the LoRA out of distribution for plain
# inference.
#
# Each command writes its own per-run subfolder under --output_dir.
# Adjust CUDA_VISIBLE_DEVICES per your machine (these can run on any 2 free
# GPUs once the main batch finishes).

set -euo pipefail
cd /cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo

# ===== 9. ABLATION: M4 cs1024 LoRA WITHOUT its NAMM =====
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --lora_checkpoint results/rh_m4_frozen_5t/42/gcs_upload/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
    --splits test extended_test --run_label ext_no_namm \
    --output_dir eval_results/lora_m4_cs1024_5t_ablation \
    2>&1 | tee /tmp/eval_lora_m4_cs1024_ablation.log

# ===== 10. ABLATION: M4 cs2048 LoRA WITHOUT its NAMM =====
env CUDA_VISIBLE_DEVICES=1 python scripts/eval_namm_splits.py \
    --lora_checkpoint eval_results/lora_m4_cs2048/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
    --splits test extended_test --run_label ext_no_namm \
    --output_dir eval_results/lora_m4_cs2048_5t_ablation \
    2>&1 | tee /tmp/eval_lora_m4_cs2048_ablation.log
