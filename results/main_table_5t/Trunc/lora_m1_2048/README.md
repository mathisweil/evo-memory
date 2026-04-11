# M1 LoRA, input truncated to last 2048 tokens

Same as Trunc/lora_m1_1024 but with double the input budget.

**Source run:** `eval_results/trunc_lora_m1_2048_5t/trunc_20260411_204758/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 39.09 | 25.68 |
| lb/2wikimqa | 26.67 | 29.14 |
| lb/qasper_e | 29.73 | 27.32 |
| lb/hotpotqa_e | 16.67 | 31.94 |
| lb/2wikimqa_e | 25.51 | 25.28 |
| **mean F1** | **27.53** | **27.87** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --truncate_input_to 2048 \
    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
    --splits test extended_test --run_label trunc \
    --output_dir eval_results/trunc_lora_m1_2048_5t
```

