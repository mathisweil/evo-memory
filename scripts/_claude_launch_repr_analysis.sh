#!/bin/bash
# Launch eviction representation analysis.
# Usage: bash scripts/_claude_launch_repr_analysis.sh <idx>
#   0: Plain (no LoRA) + NAMM cs1024
#   1: M1 LoRA + NAMM cs1024
#   2: M4 LoRA + NAMM cs1024
set -u
IDX="${1:?usage: $0 <idx>}"

PROJ=/cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo
ENV=/cs/student/project_msc/2025/csml/rhautier/envs/th2
cd "$PROJ"
source /cs/student/project_msc/2025/csml/rhautier/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV"
export CUDA_VISIBLE_DEVICES=0

NAMM_1024="experiments/namm_only_runs/memory_evolution_hf/namm-training/rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs1024/1337/ckpt.pt"
LORA_M1="results/rh_m1_lora_instruct_5t/42/best_ckpt.pt"
LORA_M4="results/rh_m4_frozen_5t/42/gcs_upload/best_ckpt.pt"
OUTDIR="analysis_out/eviction_repr"

case "$IDX" in
  0)
    echo "=== Plain (no LoRA) + NAMM cs1024 ==="
    python scripts/eviction_representation_analysis.py \
        --namm_checkpoint "$NAMM_1024" \
        --cache_size 1024 \
        --variant plain \
        --splits test extended_test \
        --output_dir "$OUTDIR"
    ;;
  1)
    echo "=== M1 LoRA + NAMM cs1024 ==="
    python scripts/eviction_representation_analysis.py \
        --namm_checkpoint "$NAMM_1024" \
        --cache_size 1024 \
        --variant m1 \
        --lora_checkpoint "$LORA_M1" \
        --splits test extended_test \
        --output_dir "$OUTDIR"
    ;;
  2)
    echo "=== M4 LoRA + NAMM cs1024 ==="
    python scripts/eviction_representation_analysis.py \
        --namm_checkpoint "$NAMM_1024" \
        --cache_size 1024 \
        --variant m4 \
        --lora_checkpoint "$LORA_M4" \
        --splits test extended_test \
        --output_dir "$OUTDIR"
    ;;
esac
