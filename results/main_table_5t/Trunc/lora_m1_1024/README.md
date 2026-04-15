# M1 LoRA, input truncated to last 1024 tokens

M1 LoRA evaluated on its last-1024-token input. Pairs with Trunc/plain_1024 to isolate how much the LoRA recovers under naive truncation, and pairs with M2/M4 to see how learned eviction compares to the simplest possible baseline at the same budget.

**Status:** ⏳ pending — no completed `ext_*` run found in `eval_results/trunc_lora_m1_1024_5t_bs1/`.

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --truncate_input_to 1024 \
    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
    --splits test extended_test --run_label trunc \
    --output_dir eval_results/trunc_lora_m1_1024_5t
```
