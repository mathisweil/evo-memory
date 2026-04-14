# M1 LoRA (no NAMM training) + NAMM eviction cs1024

M1 LoRA evaluated under NAMM eviction it was NOT trained with. Measures the distribution shift penalty: the LoRA adapted to full-context attention patterns but now faces a post-eviction cache. Compare with M4/cs1024 (LoRA trained WITH NAMM) to quantify the value of training under eviction.

**Source run:** `eval_results/lora_m1_namm_cs1024_5t/ext_20260412_205519/`

## Results

| Task | test | extended_test |
|---|---|---|
| lb/qasper | 19.42 | 17.52 |
| lb/2wikimqa | 27.56 | 19.24 |
| lb/qasper_e | 27.48 | 22.28 |
| lb/hotpotqa_e | 35.33 | 33.57 |
| lb/2wikimqa_e | 26.19 | 20.09 |
| **mean F1** | **27.20** | **22.54** |

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 python scripts/eval_namm_splits.py \
    --namm_checkpoint experiments/namm_only_runs/memory_evolution_hf/namm-training/rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs1024/1337/ckpt.pt \
    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/lora_m1_namm_cs1024_5t
```

