#!/usr/bin/env bash
# =============================================================================
# NAMM vs Recency baseline — LLaMA 3.2-1B on single GPU
#
# Usage (from the repo directory):
#   tmux new-session -s namm
#   bash run_namm_experiment.sh
#
# Then detach with Ctrl+b d. Reattach with: tmux attach -t namm
# Results land in: ./experiments/memory_evolution_hf/
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate.sh"

echo '============================================================'
echo ' NAMM Training — LLaMA 3.2-1B, Stage 1 (qasper)'
echo ' pop_size=8, samples_batch_size=16, cache_size=1024, max_iters=200'
echo '============================================================'

torchrun --standalone --nproc_per_node=1 run_namm_training.py \
    run@_global_=namm_bam_i1_llama32_1b.yaml

echo ''
echo '============================================================'
echo ' Training complete!'
echo '============================================================'
echo ''
echo 'Checkpoints are in: ./experiments/memory_evolution_hf/'
echo 'Find the best checkpoint (ckpt.pt) and run evaluation:'
echo ''
echo '  # NAMM eval (pass your checkpoint path):'
echo '  torchrun --standalone --nproc_per_node=1 run_namm_training.py \'
echo '      run@_global_=namm_bam_eval_llama32_1b.yaml \'
echo "      init_from='./experiments/.../ckpt.pt'"
echo ''
echo '  # Recency baseline eval (no checkpoint needed):'
echo '  torchrun --standalone --nproc_per_node=1 run_namm_training.py \'
echo '      run@_global_=recency_baseline_llama32_1b.yaml'
echo ''
