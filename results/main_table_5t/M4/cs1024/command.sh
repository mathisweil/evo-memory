#!/bin/bash
set -e
cd /cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo

env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --namm_checkpoint experiments/namm_only_runs/memory_evolution_hf/namm-training/rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs1024/1337/ckpt.pt \
    --lora_checkpoint results/rh_m4_frozen_5t/42/gcs_upload/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/lora_m4_cs1024_5t
