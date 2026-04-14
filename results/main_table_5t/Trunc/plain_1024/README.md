# Plain Llama, input truncated to last 1024 tokens

Naive tail-only baseline: every prompt is decoded from its last 1024 token ids before the model sees it. No KV cache eviction, no policy hooks — the model simply runs on a shorter input. The cleanest 'StreamingLLM rolling-window' comparison: how much can we recover with no learned policy?

**Status:** ⏳ pending — no completed `ext_*` run found in `eval_results/trunc_plain_1024_5t_bs1/`.

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --truncate_input_to 1024 \
    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
    --splits test extended_test --run_label trunc \
    --output_dir eval_results/trunc_plain_1024_5t
```
