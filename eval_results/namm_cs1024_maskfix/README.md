# NAMM cs1024 — maskfix checkpoint

CMA-ES trained NAMM policy at cache_size=1024 with the attention mask fix
(commit e3655c3). The causal mask misalignment that caused attention collapse
during split processing is corrected in this training run.

## Checkpoint info
- **ckpt.pt**: best checkpoint by val_tasks_aggregate
- **latest.pt**: latest checkpoint (for warmstart/resume)
- **rng_*.pt**: RNG state for exact resumption

## Training command (to resume)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_namm.py \
    'run@_global_=namm_bam_i1_llama32_1b_5t' \
    cache_size=1024 max_memory_length=1024 \
    run_name_suffix=llama32-1b-5t-cs1024-maskfix \
    wandb_log=true wandb_project=memory_evolution_hf \
    seed=1337 \
    init_from=eval_results/namm_cs1024_maskfix/latest.pt
```

## LoRA training command (M4 with this NAMM)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_lora.py \
    --config scripts/configs/lora_rh_m4_instruct_5t.yaml \
    --run_name rh_m4_5t_cs1024_maskfix \
    --wandb_project memory_evolution_hf \
    --wandb_run_name rh_m4_5t_cs1024_maskfix \
    --no-gcs --eval_interval 2 --cache_size 1024 \
    --namm_checkpoint eval_results/namm_cs1024_maskfix/ckpt.pt
```
