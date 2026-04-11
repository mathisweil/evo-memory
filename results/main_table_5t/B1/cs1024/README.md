# B1 — recency eviction, cache_size=1024

Base model with a fixed recency policy (keep most recent, evict oldest). No learned policy, no training.

**Source run:** `eval_results/recency_cs1024_5t/ext_rerun_20260411_173358/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 15.70 | 13.61 |
| lb/2wikimqa | 7.94 | 7.37 |
| lb/qasper_e | 15.16 | 10.02 |
| lb/hotpotqa_e | 10.15 | 5.05 |
| lb/2wikimqa_e | 9.12 | 8.17 |
| **mean F1** | **11.62** | **8.84** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/recency_cs1024_5t
```

