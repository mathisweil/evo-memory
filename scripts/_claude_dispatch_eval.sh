#!/bin/bash
# Dispatcher for the parallel eval jobs.
# Usage: bash scripts/_claude_dispatch_eval.sh <idx>
#   idx in 0..7  (main 8 evals)
#   idx in 8..9  (ablations: M4 LoRA without NAMM, large cache)
#
# Runs the matching command on the local GPU (CUDA_VISIBLE_DEVICES=0)
# and tees output to logs/eval_<label>_<host>.log

set -u
IDX="${1:?usage: $0 <idx 0..7>}"

PROJ=/cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo
ENV=/cs/student/project_msc/2025/csml/rhautier/envs/th2

cd "$PROJ"

# Activate conda env
source /cs/student/project_msc/2025/csml/rhautier/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV"

export CUDA_VISIBLE_DEVICES=0
HOST=$(hostname -s)
TS=$(date +%Y%m%d_%H%M%S)

# Marker so the launcher can extract pid + log
echo "DISPATCH_START idx=$IDX host=$HOST ts=$TS pid=$$"

case "$IDX" in
  0)
    LABEL=plain_baseline_5t
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_plain_llama.py \
        --filter_by_length 8192 --batch_size 16 \
        --splits test extended_test --run_label ext \
        --output_dir eval_results/plain_baseline_5t \
        2>&1 | tee "$LOG"
    ;;
  1)
    LABEL=namm_cs1024_5t
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --namm_checkpoint experiments/namm_only_runs/memory_evolution_hf/namm-training/rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs1024/1337/ckpt.pt \
        --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
        --splits test extended_test --run_label ext \
        --output_dir eval_results/namm_cs1024_5t \
        2>&1 | tee "$LOG"
    ;;
  2)
    LABEL=namm_cs2048_5t
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --namm_checkpoint eval_results/namm_cs2048_friend/ckpt.pt \
        --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
        --splits test extended_test --run_label ext \
        --output_dir eval_results/namm_cs2048_5t \
        2>&1 | tee "$LOG"
    ;;
  3)
    LABEL=recency_cs1024_5t
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
        --splits test extended_test --run_label ext \
        --output_dir eval_results/recency_cs1024_5t \
        2>&1 | tee "$LOG"
    ;;
  4)
    LABEL=recency_cs2048_5t
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
        --splits test extended_test --run_label ext \
        --output_dir eval_results/recency_cs2048_5t \
        2>&1 | tee "$LOG"
    ;;
  5)
    LABEL=lora_m1_5t
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
        --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
        --splits test extended_test --run_label ext \
        --output_dir eval_results/lora_m1_5t \
        2>&1 | tee "$LOG"
    ;;
  6)
    LABEL=lora_m4_cs1024_5t
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --namm_checkpoint experiments/namm_only_runs/memory_evolution_hf/namm-training/rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs1024/1337/ckpt.pt \
        --lora_checkpoint results/rh_m4_frozen_5t/42/gcs_upload/best_ckpt.pt \
        --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
        --splits test extended_test --run_label ext \
        --output_dir eval_results/lora_m4_cs1024_5t \
        2>&1 | tee "$LOG"
    ;;
  7)
    LABEL=lora_m4_cs2048_5t
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --namm_checkpoint eval_results/namm_cs2048_friend/ckpt.pt \
        --lora_checkpoint eval_results/lora_m4_cs2048/best_ckpt.pt \
        --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
        --splits test extended_test --run_label ext \
        --output_dir eval_results/lora_m4_cs2048_5t \
        2>&1 | tee "$LOG"
    ;;
  8)
    LABEL=lora_m4_cs1024_ablation
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --lora_checkpoint results/rh_m4_frozen_5t/42/gcs_upload/best_ckpt.pt \
        --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
        --splits test extended_test --run_label ext_no_namm \
        --output_dir eval_results/lora_m4_cs1024_5t_ablation \
        2>&1 | tee "$LOG"
    ;;
  9)
    LABEL=lora_m4_cs2048_ablation
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --lora_checkpoint eval_results/lora_m4_cs2048/best_ckpt.pt \
        --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
        --splits test extended_test --run_label ext_no_namm \
        --output_dir eval_results/lora_m4_cs2048_5t_ablation \
        2>&1 | tee "$LOG"
    ;;
  10)
    LABEL=recency_plain_cs1024_rerun
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
        --splits test extended_test --run_label ext_rerun \
        --output_dir eval_results/recency_cs1024_5t \
        2>&1 | tee "$LOG"
    ;;
  11)
    LABEL=recency_plain_cs2048_rerun
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
        --splits test extended_test --run_label ext_rerun \
        --output_dir eval_results/recency_cs2048_5t \
        2>&1 | tee "$LOG"
    ;;
  12)
    LABEL=lora_m1_recency_cs1024
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
        --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
        --splits test extended_test --run_label ext \
        --output_dir eval_results/lora_m1_recency_cs1024_5t \
        2>&1 | tee "$LOG"
    ;;
  13)
    LABEL=lora_m1_recency_cs2048
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
        --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
        --splits test extended_test --run_label ext \
        --output_dir eval_results/lora_m1_recency_cs2048_5t \
        2>&1 | tee "$LOG"
    ;;
  40)
    LABEL=lora_m1_namm_cs1024
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --namm_checkpoint experiments/namm_only_runs/memory_evolution_hf/namm-training/rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs1024/1337/ckpt.pt \
        --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
        --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
        --splits test extended_test --run_label ext \
        --output_dir eval_results/lora_m1_namm_cs1024_5t \
        2>&1 | tee "$LOG"
    ;;
  41)
    LABEL=lora_m1_namm_cs2048
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --namm_checkpoint eval_results/namm_cs2048_friend/ckpt.pt \
        --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
        --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
        --splits test extended_test --run_label ext \
        --output_dir eval_results/lora_m1_namm_cs2048_5t \
        2>&1 | tee "$LOG"
    ;;
  50)
    # M4 cs2048 with NEW best checkpoint (step 396, resumed training)
    LABEL=lora_m4_cs2048_newbest
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --namm_checkpoint eval_results/namm_cs2048_friend/ckpt.pt \
        --lora_checkpoint results/rh_m4_frozen_5t/42/best_ckpt.pt \
        --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
        --splits test extended_test --run_label ext_newbest \
        --output_dir eval_results/lora_m4_cs2048_5t \
        2>&1 | tee "$LOG"
    ;;
  51)
    # A4 cs2048 ablation with NEW best checkpoint
    LABEL=lora_m4_cs2048_newbest_ablation
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --lora_checkpoint results/rh_m4_frozen_5t/42/best_ckpt.pt \
        --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
        --splits test extended_test --run_label ext_newbest \
        --output_dir eval_results/lora_m4_cs2048_5t_ablation \
        2>&1 | tee "$LOG"
    ;;
  30)
    LABEL=trunc_plain_1024
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --truncate_input_to 1024 \
        --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
        --splits test extended_test --run_label trunc \
        --output_dir eval_results/trunc_plain_1024_5t \
        2>&1 | tee "$LOG"
    ;;
  31)
    LABEL=trunc_plain_2048
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --truncate_input_to 2048 \
        --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
        --splits test extended_test --run_label trunc \
        --output_dir eval_results/trunc_plain_2048_5t \
        2>&1 | tee "$LOG"
    ;;
  32)
    LABEL=trunc_lora_m1_1024
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --truncate_input_to 1024 \
        --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
        --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
        --splits test extended_test --run_label trunc \
        --output_dir eval_results/trunc_lora_m1_1024_5t \
        2>&1 | tee "$LOG"
    ;;
  33)
    LABEL=trunc_lora_m1_2048
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --truncate_input_to 2048 \
        --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
        --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
        --splits test extended_test --run_label trunc \
        --output_dir eval_results/trunc_lora_m1_2048_5t \
        2>&1 | tee "$LOG"
    ;;
  20)
    LABEL=classic_recency_plain_cs1024
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --use_classic_recency \
        --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
        --splits test extended_test --run_label classic \
        --output_dir eval_results/classic_recency_plain_cs1024_5t \
        2>&1 | tee "$LOG"
    ;;
  21)
    LABEL=classic_recency_plain_cs2048
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --use_classic_recency \
        --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
        --splits test extended_test --run_label classic \
        --output_dir eval_results/classic_recency_plain_cs2048_5t \
        2>&1 | tee "$LOG"
    ;;
  22)
    LABEL=classic_recency_lora_m1_cs1024
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --use_classic_recency \
        --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
        --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
        --splits test extended_test --run_label classic \
        --output_dir eval_results/classic_recency_lora_m1_cs1024_5t \
        2>&1 | tee "$LOG"
    ;;
  23)
    LABEL=classic_recency_lora_m1_cs2048
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --use_classic_recency \
        --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
        --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
        --splits test extended_test --run_label classic \
        --output_dir eval_results/classic_recency_lora_m1_cs2048_5t \
        2>&1 | tee "$LOG"
    ;;
  60)
    # M2 maskfix: NAMM only (no LoRA), cs1024, maskfix checkpoint
    LABEL=namm_cs1024_maskfix_5t
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --namm_checkpoint eval_results/namm_cs1024_maskfix/ckpt.pt \
        --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
        --splits test extended_test --run_label ext_maskfix \
        --output_dir eval_results/namm_cs1024_maskfix_5t \
        2>&1 | tee "$LOG"
    ;;
  61)
    # M1 LoRA under maskfix NAMM cs1024 (distribution shift test)
    LABEL=lora_m1_namm_cs1024_maskfix
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --namm_checkpoint eval_results/namm_cs1024_maskfix/ckpt.pt \
        --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
        --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
        --splits test extended_test --run_label ext_maskfix \
        --output_dir eval_results/lora_m1_namm_cs1024_maskfix_5t \
        2>&1 | tee "$LOG"
    ;;
  70)
    # M4 maskfix: LoRA + maskfix NAMM cs1024
    LABEL=lora_m4_cs1024_maskfix_5t
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --namm_checkpoint eval_results/namm_cs1024_maskfix/ckpt.pt \
        --lora_checkpoint checkpoints_backup/lora_m4_cs1024_maskfix/best_ckpt_step222_val47.17.pt \
        --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
        --splits test extended_test --run_label ext_maskfix \
        --output_dir eval_results/lora_m4_cs1024_maskfix_5t \
        2>&1 | tee "$LOG"
    ;;
  71)
    # A4 maskfix: LoRA only (no NAMM), full cache
    LABEL=lora_m4_cs1024_maskfix_ablation
    LOG="$PROJ/logs/eval_${LABEL}_${HOST}_${TS}.log"
    echo "DISPATCH_LOG $LOG"
    python scripts/eval_namm_splits.py \
        --lora_checkpoint checkpoints_backup/lora_m4_cs1024_maskfix/best_ckpt_step222_val47.17.pt \
        --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
        --splits test extended_test --run_label ext_maskfix_ablation \
        --output_dir eval_results/lora_m4_cs1024_maskfix_ablation_5t \
        2>&1 | tee "$LOG"
    ;;
  *)
    echo "ERROR: unknown idx" >&2
    exit 2
    ;;
esac

echo "DISPATCH_END idx=$IDX rc=$?"
