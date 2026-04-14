#!/bin/bash
# Launch hidden state shift analysis.
# Usage: bash scripts/_claude_launch_hidden_shift.sh <idx>
#   0: No LoRA, NAMM cs1024
#   1: M1 LoRA (trained without NAMM), NAMM cs1024
#   2: M4 LoRA (trained with NAMM), NAMM cs1024
#   3: No LoRA, NAMM cs2048
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
LORA_M1="results/rh_m1_lora_instruct_5t/42/best_ckpt.pt"
LORA_M4="results/rh_m4_frozen_5t/42/gcs_upload/best_ckpt.pt"
OUTDIR="analysis_out/hidden_state_shift"

case "$IDX" in
  0)
    echo "=== No LoRA, NAMM cs1024 ==="
    python scripts/hidden_state_shift_analysis.py \
        --namm_checkpoint "$NAMM_1024" \
        --cache_size 1024 \
        --splits test extended_test \
        --output_dir "$OUTDIR"
    ;;
  1)
    echo "=== M1 LoRA (no NAMM training), NAMM cs1024 ==="
    python scripts/hidden_state_shift_analysis.py \
        --namm_checkpoint "$NAMM_1024" \
        --lora_checkpoint "$LORA_M1" \
        --cache_size 1024 \
        --splits test extended_test \
        --output_dir "$OUTDIR"
    ;;
  2)
    echo "=== M4 LoRA (NAMM-trained), NAMM cs1024 ==="
    python scripts/hidden_state_shift_analysis.py \
        --namm_checkpoint "$NAMM_1024" \
        --lora_checkpoint "$LORA_M4" \
        --cache_size 1024 \
        --splits test extended_test \
        --output_dir "$OUTDIR"
    ;;
  3)
    echo "=== No LoRA, NAMM cs2048 ==="
    python scripts/hidden_state_shift_analysis.py \
        --namm_checkpoint "$NAMM_2048" \
        --cache_size 2048 \
        --splits test extended_test \
        --output_dir "$OUTDIR"
    ;;
esac
