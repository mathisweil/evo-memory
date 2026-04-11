#!/bin/bash
set -e
cd /cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo

env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/recency_cs1024_5t
