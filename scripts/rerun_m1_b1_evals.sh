#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python}"

"$PY" scripts/eval_namm_splits.py \
    --run_config namm_bam_i1_llama32_1b_5t \
    --lora_checkpoint artifacts/m1_r8/best_ckpt.pt \
    --cache_size 8192 --filter_by_length 8192 \
    --splits test \
    --output_dir experiments/experiment_N/m1_full_cache/test \
    --batch_size 2

"$PY" scripts/eval_namm_splits.py \
    --run_config namm_bam_i1_llama32_1b_5t \
    --lora_checkpoint artifacts/m1_r8/best_ckpt.pt \
    --use_classic_recency \
    --cache_size 1024 \
    --splits test \
    --output_dir experiments/experiment_N/m1_recency/cs1024 \
    --batch_size 2

"$PY" scripts/eval_namm_splits.py \
    --run_config recency_baseline_llama32_1b \
    --cache_size 1024 --filter_by_length 6500 \
    --splits test \
    --output_dir experiments/experiment_N/b1_recency/cs1024 \
    --batch_size 2

"$PY" scripts/eval_namm_splits.py \
    --run_config namm_bam_i1_llama32_1b_5t \
    --lora_checkpoint artifacts/m1_r8/best_ckpt.pt \
    --cache_size 8192 --filter_by_length 8192 \
    --splits test \
    --output_dir experiments/experiment_N/m1_full_cache/test \
    --batch_size 2

"$PY" scripts/eval_namm_splits.py \
    --run_config namm_bam_i1_llama32_1b_5t \
    --lora_checkpoint artifacts/m1_r8/best_ckpt.pt \
    --use_classic_recency \
    --cache_size 1024 \
    --splits test \
    --output_dir experiments/experiment_N/m1_recency/cs1024 \
    --batch_size 2

"$PY" scripts/eval_namm_splits.py \
    --run_config recency_baseline_llama32_1b \
    --cache_size 1024 \
    --splits test \
    --output_dir experiments/experiment_N/b1_recency/cs1024 \
    --batch_size 2
