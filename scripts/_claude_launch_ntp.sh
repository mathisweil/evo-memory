#!/bin/bash
# Launch NTP perplexity evaluation.
# Usage: bash scripts/_claude_launch_ntp.sh <idx>
#   0: Base model, full cache
#   1: Base model, maskfix NAMM cs1024
#   2: Base model, truncation 1024
#   3: M1 LoRA, full cache
#   4: M1 LoRA, maskfix NAMM cs1024
#   5: M4 maskfix LoRA, maskfix NAMM cs1024
#   6: Base model, truncation 2048
#   7: M1 LoRA, truncation 1024
set -u
IDX="${1:?usage: $0 <idx>}"

PROJ=/cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo
ENV=/cs/student/project_msc/2025/csml/rhautier/envs/th2
cd "$PROJ"
source /cs/student/project_msc/2025/csml/rhautier/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV"
export CUDA_VISIBLE_DEVICES=0

NAMM_MASKFIX="eval_results/namm_cs1024_maskfix/ckpt.pt"
LORA_M1="results/rh_m1_lora_instruct_5t/42/best_ckpt.pt"
LORA_M4FIX="checkpoints_backup/lora_m4_cs1024_maskfix/best_ckpt_step222_val47.17.pt"
OUTDIR="analysis_out/ntp_perplexity"

case "$IDX" in
  0)
    echo "=== Base model, full cache ==="
    python scripts/eval_ntp_perplexity.py \
        --splits test extended_test \
        --label base_full \
        --output_dir "$OUTDIR"
    ;;
  1)
    echo "=== Base model, maskfix NAMM cs1024 ==="
    python scripts/eval_ntp_perplexity.py \
        --namm_checkpoint "$NAMM_MASKFIX" \
        --cache_size 1024 \
        --splits test extended_test \
        --label base_namm1024 \
        --output_dir "$OUTDIR"
    ;;
  2)
    echo "=== Base model, truncation 1024 ==="
    python scripts/eval_ntp_perplexity.py \
        --truncate_to 1024 \
        --splits test extended_test \
        --label base_trunc1024 \
        --output_dir "$OUTDIR"
    ;;
  3)
    echo "=== M1 LoRA, full cache ==="
    python scripts/eval_ntp_perplexity.py \
        --lora_checkpoint "$LORA_M1" \
        --splits test extended_test \
        --label m1_full \
        --output_dir "$OUTDIR"
    ;;
  4)
    echo "=== M1 LoRA, maskfix NAMM cs1024 ==="
    python scripts/eval_ntp_perplexity.py \
        --namm_checkpoint "$NAMM_MASKFIX" \
        --lora_checkpoint "$LORA_M1" \
        --cache_size 1024 \
        --splits test extended_test \
        --label m1_namm1024 \
        --output_dir "$OUTDIR"
    ;;
  5)
    echo "=== M4 maskfix LoRA, maskfix NAMM cs1024 ==="
    python scripts/eval_ntp_perplexity.py \
        --namm_checkpoint "$NAMM_MASKFIX" \
        --lora_checkpoint "$LORA_M4FIX" \
        --cache_size 1024 \
        --splits test extended_test \
        --label m4fix_namm1024 \
        --output_dir "$OUTDIR"
    ;;
  6)
    echo "=== Base model, truncation 2048 ==="
    python scripts/eval_ntp_perplexity.py \
        --truncate_to 2048 \
        --splits test extended_test \
        --label base_trunc2048 \
        --output_dir "$OUTDIR"
    ;;
  7)
    echo "=== M1 LoRA, truncation 1024 ==="
    python scripts/eval_ntp_perplexity.py \
        --truncate_to 1024 \
        --lora_checkpoint "$LORA_M1" \
        --splits test extended_test \
        --label m1_trunc1024 \
        --output_dir "$OUTDIR"
    ;;
esac
