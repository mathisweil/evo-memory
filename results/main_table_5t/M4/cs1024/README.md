# M4 — LoRA on frozen NAMM, cache_size=1024

LoRA fine-tuned on top of a frozen NAMM (cs=1024). The LoRA and NAMM are evaluated together.

**Source run:** `eval_results/lora_m4_cs1024_5t/ext_20260411_145757/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 19.55 | 13.54 |
| lb/2wikimqa | 27.56 | 21.29 |
| lb/qasper_e | 34.83 | 24.66 |
| lb/hotpotqa_e | 26.39 | 41.15 |
| lb/2wikimqa_e | 18.76 | 19.93 |
| **mean F1** | **25.42** | **24.12** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --namm_checkpoint experiments/namm_only_runs/memory_evolution_hf/namm-training/rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs1024/1337/ckpt.pt \
    --lora_checkpoint results/rh_m4_frozen_5t/42/gcs_upload/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/lora_m4_cs1024_5t
```

