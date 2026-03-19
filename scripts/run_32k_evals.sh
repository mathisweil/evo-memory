#!/bin/bash
# Full-context (32K) eval suite.
# Uses llama3 rope scaling (transformers 4.45+) + SDPA — fits on 16GB VRAM.
# Covers 100% passage_ret docs, 99% qasper, 76% narrativeqa.
set -e

CKPT_NAMM="exp_local/memory_evolution_hf/Llama-3.2-1B-Instruct/NAMM/attn-spec-norm/bam/binary-1024cs/lb3subset-eval-cma-es-p8-rMeanTrue-shared-8pop-16qs-256fixDel-llama32-1b-instruct-stage1/1337/ckpt.pt"
CKPT_SFT="results/m1_sft/1337/ckpt.pt"
GROUP="Llama-3.2-1B-Instruct/fullctx-32k-evals"
MXPOS=32768

echo "=== 1/5: Base full-cache @ 32K ==="
python main.py 'run@_global_=full_cache_baseline_llama32_1b_instruct' \
    max_position_id=$MXPOS \
    wandb_group_name="$GROUP" \
    wandb_run_name="base-fullcache-cs${MXPOS}-llama32-1b"

echo "=== 2/5: Recency @ 1024 on 32K input ==="
python main.py 'run@_global_=recency_baseline_llama32_1b_instruct' \
    max_position_id=$MXPOS cache_size=1024 \
    wandb_group_name="$GROUP" \
    wandb_run_name="recency-cs1024-32kinput-llama32-1b"

echo "=== 3/5: Recency @ 4096 on 32K input ==="
python main.py 'run@_global_=recency_baseline_llama32_1b_instruct' \
    max_position_id=$MXPOS cache_size=4096 \
    wandb_group_name="$GROUP" \
    wandb_run_name="recency-cs4096-32kinput-llama32-1b"

echo "=== 4/5: NAMM @ 1024 on 32K input ==="
python main.py 'run@_global_=namm_bam_eval_llama32_1b_instruct' \
    init_from="$CKPT_NAMM" seed=1337 \
    max_position_id=$MXPOS \
    wandb_group_name="$GROUP" \
    wandb_run_name="namm-cs1024-32kinput-llama32-1b"

echo "=== 5/5: m1-SFT @ 32K (OOD — trained on 3500 tokens, indicative only) ==="
python run_eval.py --method m1_sft --ckpt "$CKPT_SFT" --seed 1337 \
    --max-position-id $MXPOS --wandb-group "$GROUP"

echo "=== Done ==="
