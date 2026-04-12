# M2 — standalone NAMM, cache_size=1024

Trained NAMM eviction policy on top of the frozen base model. No LoRA, no fine-tuning of the LM weights.

**Source run:** `eval_results/namm_cs1024_5t/ext_20260412_141755/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 28.30 | 19.49 |
| lb/2wikimqa | 27.56 | 23.74 |
| lb/qasper_e | 8.09 | 12.13 |
| lb/hotpotqa_e | 17.50 | 26.36 |
| lb/2wikimqa_e | 24.16 | 22.05 |
| **mean F1** | **21.12** | **20.75** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --namm_checkpoint experiments/namm_only_runs/memory_evolution_hf/namm-training/rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs1024/1337/ckpt.pt \
    --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/namm_cs1024_5t
```

