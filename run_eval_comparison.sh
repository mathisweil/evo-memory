#!/usr/bin/env bash
# =============================================================================
# Evaluation comparison: NAMM vs Recency vs Full-cache — LLaMA 3.2-1B
#
# Runs all three methods at multiple cache sizes and logs every run to the
# same wandb group (Llama-3.2-1B/eval-comparison) for easy side-by-side
# comparison.
#
# Usage (from the repo directory):
#   tmux new-session -s eval
#   bash run_eval_comparison.sh
#
# Or run individual sections by commenting out the ones you don't need.
# =============================================================================

set -euo pipefail

CONDA_SH="/cs/student/project_msc/2025/csml/gmaralla/miniconda3/etc/profile.d/conda.sh"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${CONDA_SH}"
conda activate th2
export HF_HOME="/cs/student/project_msc/2025/csml/gmaralla/.hf_cache"
cd "${REPO_DIR}"

# ---------------------------------------------------------------------------
# EDIT THIS: path to your best NAMM checkpoint from stage-1 training
# ---------------------------------------------------------------------------
NAMM_CKPT="/cs/student/project_msc/2025/csml/gmaralla/NAMM_implementation/exp_local/memory_evolution_hf/Llama-3.2-1B/NAMM/attn-spec-norm/bam/binary-1024cs/qasper-cma-es-p8-rMeanTrue-shared-8pop-16qs-256fixDel-llama32-1b-stage1/1337/ckpt.pt"

# Cache sizes to sweep for NAMM and recency (must be ≤ training cache_size=1024)
CACHE_SIZES=(256 512 1024)

# ---------------------------------------------------------------------------
# 1.  Full-cache baseline  (no eviction, upper bound)  -- DONE
# ---------------------------------------------------------------------------
# torchrun --standalone --nproc_per_node=1 main.py \
#     run@_global_=full_cache_baseline_llama32_1b.yaml

# ---------------------------------------------------------------------------
# 2.  Recency baseline at each cache size  -- DONE
# ---------------------------------------------------------------------------
# for CS in "${CACHE_SIZES[@]}"; do
#     torchrun --standalone --nproc_per_node=1 main.py \
#         run@_global_=recency_baseline_llama32_1b.yaml \
#         cache_size="${CS}"
# done

# ---------------------------------------------------------------------------
# 3.  NAMM eval at each cache size
# ---------------------------------------------------------------------------
for CS in "${CACHE_SIZES[@]}"; do
    echo '============================================================'
    echo " NAMM eval  (cache_size=${CS})"
    echo " wandb run: namm-cs${CS}-llama32-1b"
    echo '============================================================'

    torchrun --standalone --nproc_per_node=1 main.py \
        run@_global_=namm_bam_eval_llama32_1b.yaml \
        init_from="${NAMM_CKPT}" \
        cache_size="${CS}"

    echo ''
done

echo '============================================================'
echo ' All eval runs complete!'
echo ' Check wandb: project=memory_evolution_hf'
echo '              group=Llama-3.2-1B/eval-comparison'
echo '============================================================'
