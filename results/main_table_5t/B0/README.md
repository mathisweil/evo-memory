# B0 — plain Llama, full KV cache

Base Llama-3.2-1B-Instruct with no eviction, no fine-tuning. Performance ceiling for the 5-task QA subset.

**Status:** ⏳ pending — no completed `ext_*` run found in `eval_results/plain_baseline_5t_bs1/`.

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_plain_llama.py \
    --filter_by_length 8192 --batch_size 16 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/plain_baseline_5t
```
