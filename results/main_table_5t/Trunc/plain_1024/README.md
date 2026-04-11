# Plain Llama, input truncated to last 1024 tokens

Naive tail-only baseline: every prompt is decoded from its last 1024 token ids before the model sees it. No KV cache eviction, no policy hooks — the model simply runs on a shorter input. The cleanest 'StreamingLLM rolling-window' comparison: how much can we recover with no learned policy?

**Source run:** `eval_results/trunc_plain_1024_5t/trunc_20260411_204759/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 10.68 | 10.80 |
| lb/2wikimqa | 27.96 | 17.43 |
| lb/qasper_e | 12.38 | 13.00 |
| lb/hotpotqa_e | 8.33 | 23.89 |
| lb/2wikimqa_e | 14.29 | 14.11 |
| **mean F1** | **14.73** | **15.85** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --truncate_input_to 1024 \
    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
    --splits test extended_test --run_label trunc \
    --output_dir eval_results/trunc_plain_1024_5t
```

