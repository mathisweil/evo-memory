# Plain Llama, input truncated to last 2048 tokens

Same as Trunc/plain_1024 but with double the input budget.

**Source run:** `eval_results/trunc_plain_2048_5t/trunc_20260411_204804/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 8.08 | 10.15 |
| lb/2wikimqa | 29.76 | 26.12 |
| lb/qasper_e | 10.13 | 11.71 |
| lb/hotpotqa_e | 8.33 | 27.53 |
| lb/2wikimqa_e | 6.91 | 17.62 |
| **mean F1** | **12.64** | **18.63** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --truncate_input_to 2048 \
    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
    --splits test extended_test --run_label trunc \
    --output_dir eval_results/trunc_plain_2048_5t
```

