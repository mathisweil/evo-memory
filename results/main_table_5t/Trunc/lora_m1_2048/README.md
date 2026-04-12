# M1 LoRA, input truncated to last 2048 tokens

Same as Trunc/lora_m1_1024 but with double the input budget.

**Source run:** `eval_results/trunc_lora_m1_2048_5t/trunc_20260412_141735/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 31.56 | 30.62 |
| lb/2wikimqa | 27.56 | 18.72 |
| lb/qasper_e | 30.04 | 31.68 |
| lb/hotpotqa_e | 33.95 | 41.85 |
| lb/2wikimqa_e | 21.43 | 23.83 |
| **mean F1** | **28.91** | **29.34** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --truncate_input_to 2048 \
    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
    --splits test extended_test --run_label trunc \
    --output_dir eval_results/trunc_lora_m1_2048_5t
```

