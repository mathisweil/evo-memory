# B1 — recency eviction, cache_size=1024

Base model with a fixed recency policy (keep most recent, evict oldest). No learned policy, no training.

**Source run:** `eval_results/recency_cs1024_5t/ext_20260413_135806/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 22.77 | 12.23 |
| lb/2wikimqa | 10.42 | 4.42 |
| lb/qasper_e | 7.26 | 4.80 |
| lb/hotpotqa_e | 10.60 | 14.24 |
| lb/2wikimqa_e | 6.55 | 4.27 |
| **mean F1** | **11.52** | **7.99** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/recency_cs1024_5t
```

