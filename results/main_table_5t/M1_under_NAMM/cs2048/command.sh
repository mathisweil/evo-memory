#!/bin/bash
set -e
cd /cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo

env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --namm_checkpoint eval_results/namm_cs2048_friend/ckpt.pt \
    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/lora_m1_namm_cs2048_5t
