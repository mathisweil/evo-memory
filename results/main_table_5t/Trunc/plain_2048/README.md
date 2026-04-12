# Plain Llama, input truncated to last 2048 tokens

Same as Trunc/plain_1024 but with double the input budget.

**Source run:** `eval_results/trunc_plain_2048_5t/trunc_20260412_141754/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 24.81 | 19.97 |
| lb/2wikimqa | 25.00 | 17.23 |
| lb/qasper_e | 12.42 | 12.72 |
| lb/hotpotqa_e | 17.28 | 31.91 |
| lb/2wikimqa_e | 14.29 | 19.54 |
| **mean F1** | **18.76** | **20.27** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --truncate_input_to 2048 \
    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
    --splits test extended_test --run_label trunc \
    --output_dir eval_results/trunc_plain_2048_5t
```

