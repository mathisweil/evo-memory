#!/bin/bash
# Re-run all 4 m1_sft evals with the lora_rank bug fixed.
# The checkpoint is correct (rank=1 weights); only the eval loading was wrong.
set -e

CKPT="results/m1_sft/1337/ckpt.pt"
REC_CFG="recency_baseline_llama32_1b_instruct"
GROUP="Llama-3.2-1B-Instruct/chat-template-evals"

echo "=== m1_sft evals (lora_rank bug fixed) ==="
python run_eval.py --method m1_sft --ckpt $CKPT --seed 1337 --wandb-group "$GROUP"
python run_eval.py --method m1_sft --ckpt $CKPT --seed 1337 --run-config $REC_CFG --cache-size 4096 --wandb-group "$GROUP"
python run_eval.py --method m1_sft --ckpt $CKPT --seed 1337 --run-config $REC_CFG --cache-size 2048 --wandb-group "$GROUP"
python run_eval.py --method m1_sft --ckpt $CKPT --seed 1337 --run-config $REC_CFG --cache-size 1024 --wandb-group "$GROUP"

echo "=== Done ==="
