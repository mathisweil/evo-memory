# 00 — Evaluation Results (matched hyperparams)

F1 scores for all conditions at cs1024, with M1 and M4 both trained at lr=1e-4, dropout=0.05.

## Conditions
- **B0**: Base Llama 3.2-1B, no LoRA, full cache
- **Base+NAMM (M2)**: No LoRA, NAMM cs1024 eviction
- **M1-matched (full cache)**: LoRA lr=1e-4, full cache eval
- **M1-matched trunc 1024**: Same LoRA, input truncated to last 1024 tokens
- **M1-matched under NAMM**: Same LoRA, post-hoc NAMM cs1024
- **M4 LoRA+NAMM**: LoRA trained WITH NAMM active, eval with NAMM
- **A4 (M4 no NAMM)**: Same M4 weights, eval WITHOUT NAMM (full cache)

## Checkpoints used
- M1 LoRA: `checkpoints_backup/lora_m1_lr1e4_matched/best_ckpt.pt` (step 64)
- M4 LoRA: `checkpoints_backup/lora_m4_cs1024_maskfix/best_ckpt_step260_val52.06.pt`
- NAMM: `eval_results/namm_cs1024_maskfix/ckpt.pt` (iter 135, maskfix, CMA-ES mean)

## Commands
```bash
# M1 full cache
python scripts/eval_namm_splits.py --lora_checkpoint <M1> --cache_size 8192 \
    --splits test extended_test --extended_max_conditioning_length 8192 \
    --output_dir eval_results/lora_m1_lr1e4_matched_full_cache

# M1 under NAMM
python scripts/eval_namm_splits.py --lora_checkpoint <M1> --namm_checkpoint <NAMM> \
    --cache_size 1024 --splits test extended_test --extended_max_conditioning_length 8192 \
    --output_dir eval_results/lora_m1_lr1e4_matched_namm_cs1024

# M1 truncation
python scripts/eval_namm_splits.py --lora_checkpoint <M1> --truncate_input_to 1024 \
    --splits test extended_test --extended_max_conditioning_length 8192 \
    --output_dir eval_results/lora_m1_lr1e4_matched_trunc1024
```

## Outputs
- `matched_mean_f1.png` — bar chart, all conditions, test + extended test
- `matched_per_task_f1.png` — per-task grouped bars
- `matched_delta_decomposition.png` — eviction cost, adaptation gain, NAMM ablation deltas
- `m1_matched_*.json` — raw eval results with per-prompt F1
