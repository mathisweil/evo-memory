# B1 — recency eviction, cache_size=2048

Same as B1 cs1024 but with double the cache budget.

**Source run:** `eval_results/recency_cs2048_5t/ext_20260411_145757/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 16.09 | 9.86 |
| lb/2wikimqa | 7.80 | 7.64 |
| lb/qasper_e | 14.41 | 11.35 |
| lb/hotpotqa_e | 15.55 | 13.06 |
| lb/2wikimqa_e | 11.66 | 10.87 |
| **mean F1** | **13.10** | **10.55** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/recency_cs2048_5t
```

