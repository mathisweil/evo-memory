# B1 — recency eviction, cache_size=2048

Same as B1 cs1024 but with double the cache budget.

**Source run:** `eval_results/recency_cs2048_5t/ext_20260412_141739/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 23.32 | 16.84 |
| lb/2wikimqa | 7.63 | 4.40 |
| lb/qasper_e | 6.14 | 7.66 |
| lb/hotpotqa_e | 25.93 | 21.47 |
| lb/2wikimqa_e | 8.93 | 8.32 |
| **mean F1** | **14.39** | **11.74** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/recency_cs2048_5t
```

