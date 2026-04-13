# B0 — plain Llama, full KV cache

Base Llama-3.2-1B-Instruct with no eviction, no fine-tuning. Performance ceiling for the 5-task QA subset.

**Source run:** `eval_results/plain_baseline_5t/ext_20260413_140734/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 25.85 | 18.34 |
| lb/2wikimqa | 26.52 | 17.86 |
| lb/qasper_e | 6.06 | 13.11 |
| lb/hotpotqa_e | 44.56 | 45.88 |
| lb/2wikimqa_e | 17.46 | 23.27 |
| **mean F1** | **24.09** | **23.69** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_plain_llama.py \
    --filter_by_length 8192 --batch_size 16 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/plain_baseline_5t
```

