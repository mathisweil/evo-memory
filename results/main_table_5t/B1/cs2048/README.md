# B1 — recency eviction, cache_size=2048

Same as B1 cs1024 but with double the cache budget.

**Source run:** `eval_results/recency_cs2048_5t/ext_20260413_135806/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 23.46 | 13.79 |
| lb/2wikimqa | 7.63 | 4.22 |
| lb/qasper_e | 6.14 | 7.36 |
| lb/hotpotqa_e | 9.26 | 18.24 |
| lb/2wikimqa_e | 9.68 | 8.35 |
| **mean F1** | **11.23** | **10.39** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --filter_by_length 8192 --cache_size 2048 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/recency_cs2048_5t
```

