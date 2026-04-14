#!/bin/bash
# Deterministic eval at batch_size=1 (no OOM retries).
# Usage: bash scripts/_claude_eval_bs1.sh <idx>
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
LORA_M1="results/rh_m1_lora_instruct_5t/42/best_ckpt.pt"
LORA_M4_ORIG="checkpoints_backup/lora_m4_cs1024_original/best_ckpt.pt"
LORA_M4_MASKFIX="checkpoints_backup/lora_m4_cs1024_maskfix/best_ckpt_step260_val52.06.pt"
BS="--batch_size 1"
SPLITS="--splits test extended_test"
FILTER="--filter_by_length 8192"

case "$IDX" in
  0) # B0 plain
    python scripts/eval_plain_llama.py $FILTER $BS $SPLITS --run_label bs1 \
        --output_dir eval_results/plain_baseline_5t_bs1 2>&1 ;;
  1) # M2 maskfix NAMM cs1024
    python scripts/eval_namm_splits.py --namm_checkpoint "$NAMM_MASKFIX" \
        --cache_size 1024 $FILTER $BS $SPLITS --run_label bs1 \
        --output_dir eval_results/namm_cs1024_maskfix_5t_bs1 2>&1 ;;
  2) # Trunc plain 1024
    python scripts/eval_namm_splits.py --truncate_input_to 1024 \
        $FILTER $BS $SPLITS --run_label bs1 \
        --output_dir eval_results/trunc_plain_1024_5t_bs1 2>&1 ;;
  3) # M1 LoRA full cache
    python scripts/eval_namm_splits.py --lora_checkpoint "$LORA_M1" \
        --cache_size 8192 $FILTER $BS $SPLITS --run_label bs1 \
        --output_dir eval_results/lora_m1_5t_bs1 2>&1 ;;
  4) # M1 under maskfix NAMM
    python scripts/eval_namm_splits.py --namm_checkpoint "$NAMM_MASKFIX" \
        --lora_checkpoint "$LORA_M1" --cache_size 1024 $FILTER $BS $SPLITS --run_label bs1 \
        --output_dir eval_results/lora_m1_namm_cs1024_maskfix_5t_bs1 2>&1 ;;
  5) # Trunc LoRA 1024
    python scripts/eval_namm_splits.py --truncate_input_to 1024 \
        --lora_checkpoint "$LORA_M1" $FILTER $BS $SPLITS --run_label bs1 \
        --output_dir eval_results/trunc_lora_m1_1024_5t_bs1 2>&1 ;;
  6) # M4 maskfix + maskfix NAMM
    python scripts/eval_namm_splits.py --namm_checkpoint "$NAMM_MASKFIX" \
        --lora_checkpoint "$LORA_M4_MASKFIX" --cache_size 1024 $FILTER $BS $SPLITS --run_label bs1 \
        --output_dir eval_results/lora_m4_cs1024_maskfix_5t_bs1 2>&1 ;;
  7) # A4 maskfix (no NAMM)
    python scripts/eval_namm_splits.py --lora_checkpoint "$LORA_M4_MASKFIX" \
        --cache_size 8192 $FILTER $BS $SPLITS --run_label bs1 \
        --output_dir eval_results/lora_m4_cs1024_maskfix_ablation_5t_bs1 2>&1 ;;
esac
