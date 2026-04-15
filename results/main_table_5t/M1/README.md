# M1 — LoRA only (no NAMM)

LoRA SFT on the 5-task QA subset, full KV cache during training and eval (cache_size=8192). No eviction.

**Status:** ⏳ pending — no completed `ext_*` run found in `eval_results/lora_m1_5t_bs1/`.

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 8192 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/lora_m1_5t
```
