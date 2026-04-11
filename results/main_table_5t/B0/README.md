# B0 — plain Llama, full KV cache

Base Llama-3.2-1B-Instruct with no eviction, no fine-tuning. Performance ceiling for the 5-task QA subset.

**Source run:** `eval_results/plain_baseline_5t/ext_20260411_150627/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 9.19 | 11.54 |
| lb/2wikimqa | 20.71 | 23.10 |
| lb/qasper_e | 11.61 | 12.60 |
| lb/hotpotqa_e | 28.31 | 40.50 |
| lb/2wikimqa_e | 30.27 | 20.84 |
| **mean F1** | **20.02** | **21.72** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_plain_llama.py \
    --filter_by_length 8192 --batch_size 16 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/plain_baseline_5t
```

