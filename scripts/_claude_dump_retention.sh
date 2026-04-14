#!/bin/bash
# Dump retained positions for NAMM conditions on extended_test.
# Usage: bash scripts/_claude_dump_retention.sh <idx>
#   idx 0: M2 NAMM cs1024
#   idx 1: M2 NAMM cs2048
#   idx 2: M4 LoRA+NAMM cs1024
#   idx 3: M4 LoRA+NAMM cs2048
set -u
IDX="${1:?usage: $0 <idx>}"

PROJ=/cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo
ENV=/cs/student/project_msc/2025/csml/rhautier/envs/th2
cd "$PROJ"
source /cs/student/project_msc/2025/csml/rhautier/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV"
export CUDA_VISIBLE_DEVICES=0

NAMM_1024="experiments/namm_only_runs/memory_evolution_hf/namm-training/rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs1024/1337/ckpt.pt"
NAMM_2048="eval_results/namm_cs2048_friend/ckpt.pt"
LORA_M4_1024="results/rh_m4_frozen_5t/42/gcs_upload/best_ckpt.pt"
LORA_M4_2048="eval_results/lora_m4_cs2048/best_ckpt.pt"
OUTDIR="analysis_out/retention_dumps"

case "$IDX" in
  0)
    echo "=== M2 NAMM cs1024 ==="
    python scripts/dump_retained_positions.py \
        --namm_checkpoint "$NAMM_1024" \
        --cache_size 1024 --filter_by_length 8192 \
        --splits extended_test \
        --output_dir "$OUTDIR"
    ;;
  1)
    echo "=== M2 NAMM cs2048 ==="
    python scripts/dump_retained_positions.py \
        --namm_checkpoint "$NAMM_2048" \
        --cache_size 2048 --filter_by_length 8192 \
        --splits extended_test \
        --output_dir "$OUTDIR"
    ;;
  2)
    echo "=== M4 LoRA+NAMM cs1024 ==="
    # dump_retained_positions doesn't support --lora_checkpoint natively.
    # Use NAMM-only to get the eviction policy's decisions (same policy,
    # LoRA doesn't change eviction — only changes generation quality).
    python scripts/dump_retained_positions.py \
        --namm_checkpoint "$NAMM_1024" \
        --cache_size 1024 --filter_by_length 8192 \
        --splits extended_test \
        --output_dir "$OUTDIR"
    ;;
  3)
    echo "=== M4 LoRA+NAMM cs2048 ==="
    python scripts/dump_retained_positions.py \
        --namm_checkpoint "$NAMM_2048" \
        --cache_size 2048 --filter_by_length 8192 \
        --splits extended_test \
        --output_dir "$OUTDIR"
    ;;
esac
