#!/bin/bash
# Launch maskfix analysis experiments.
# Usage: bash scripts/_claude_launch_maskfix_analyses.sh <idx>
#   0: Attention comparison — original NAMM (broken attn) on test prompt
#   1: Attention comparison — maskfix NAMM (correct attn) on same prompt
#   2: Ghost info re-measurement with maskfix NAMM
#   3: Retention dump with maskfix NAMM (what tokens does it keep?)
#   4: Retention dump with original NAMM (for comparison)
set -u
IDX="${1:?usage: $0 <idx>}"

PROJ=/cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo
ENV=/cs/student/project_msc/2025/csml/rhautier/envs/th2
cd "$PROJ"
source /cs/student/project_msc/2025/csml/rhautier/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV"
export CUDA_VISIBLE_DEVICES=0

NAMM_ORIG="experiments/namm_only_runs/memory_evolution_hf/namm-training/rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs1024/1337/ckpt.pt"
NAMM_MASKFIX="eval_results/namm_cs1024_maskfix/ckpt.pt"

case "$IDX" in
  0)
    echo "=== Attention: original NAMM (broken attn) ==="
    python scripts/eviction_representation_analysis.py \
        --namm_checkpoint "$NAMM_ORIG" \
        --cache_size 1024 \
        --variant plain \
        --splits test extended_test \
        --output_dir analysis_out/attn_comparison_orig
    ;;
  1)
    echo "=== Attention: maskfix NAMM (correct attn) ==="
    python scripts/eviction_representation_analysis.py \
        --namm_checkpoint "$NAMM_MASKFIX" \
        --cache_size 1024 \
        --variant plain \
        --splits test extended_test \
        --output_dir analysis_out/attn_comparison_maskfix
    ;;
  2)
    echo "=== Ghost info: maskfix NAMM ==="
    python scripts/ghost_information_analysis.py \
        --namm_checkpoint "$NAMM_MASKFIX" \
        --cache_size 1024 \
        --splits test extended_test \
        --output_dir analysis_out/ghost_info_maskfix
    ;;
  3)
    echo "=== Retention dump: maskfix NAMM ==="
    python scripts/dump_retained_positions.py \
        --namm_checkpoint "$NAMM_MASKFIX" \
        --cache_size 1024 --filter_by_length 8192 \
        --splits extended_test \
        --output_dir analysis_out/retention_dumps_maskfix
    ;;
  4)
    echo "=== Retention dump: original NAMM (for comparison) ==="
    python scripts/dump_retained_positions.py \
        --namm_checkpoint "$NAMM_ORIG" \
        --cache_size 1024 --filter_by_length 8192 \
        --splits extended_test \
        --output_dir analysis_out/retention_dumps
    ;;
esac
