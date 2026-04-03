# NAMM Cache Size Sweep — LLaMA 3.2-1B, 5-Task QA

## Overview

Train NAMM eviction policies at different KV cache budgets (1024, 2048, 4096) on the same 5-task LongBench QA dataset. This produces checkpoints that can be compared to understand how cache budget affects NAMM's ability to handle long contexts.

## Prerequisites

### 1. Environment

```bash
# Python 3.10 with PyTorch + CUDA
# Tested on NVIDIA RTX 3090 Ti (24 GB VRAM)
conda activate th2  # or your environment

# Required packages: torch, transformers, hydra-core, omegaconf, wandb,
# accelerate, lm-eval, peft, numpy, jieba, fuzzywuzzy
```

### 2. Repository

```bash
git clone https://github.com/mathisweil/evo-memory.git
cd evo-memory
git checkout 76dae10  # branch: dev/train_test_fixes
```

### 3. Model Access

The runs use `meta-llama/Llama-3.2-1B-Instruct`. Either:
- Log in to HuggingFace: `huggingface-cli login`
- Or set a local path: `export LLM_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct`

### 4. WandB

```bash
wandb login
```

## Config File

All runs use the same base config: `config/run/namm_bam_i1_llama32_1b_5t.yaml`

Key parameters (set in the config, not overridden via CLI unless noted):

| Parameter | Value | Description |
|---|---|---|
| `pop_size` | 8 | CMA-ES population size |
| `samples_batch_size` | 8 | Prompts per task per training step |
| `batch_size` | 4 | Sequences per GPU forward pass |
| `max_iters` | 200 | Total CMA-ES iterations |
| `eval_interval` | 5 | Val eval every N steps |
| `memory_policy_fixed_delay` | 256 | Policy fires every 256 tokens |
| `max_new_tokens` | 256 | Chunk size for split processing |
| `max_answer_tokens` | 64 | Answer length filter |
| `min_conditioning_length` | 4096 | Min prompt tokens |
| `max_conditioning_length` | 6500 | Max prompt tokens |
| `train_frac` | 0.7 | Training split ratio |
| `val_frac` | 0.15 | Validation split ratio |

## Launch Commands

### Cache size 1024

```bash
python scripts/run_namm.py \
    run@_global_=namm_bam_i1_llama32_1b_5t \
    filter_by_length=8192 \
    cache_size=1024 \
    max_memory_length=1024 \
    run_name_suffix=llama32-1b-5t-cs1024 \
    wandb_project=memory_evolution_hf \
    wandb_group_name=namm-training
```

- **Estimated time**: ~6 hours on RTX 3090 Ti
- **VRAM usage**: ~10 GB peak

### Cache size 2048

```bash
python scripts/run_namm.py \
    run@_global_=namm_bam_i1_llama32_1b_5t \
    filter_by_length=8192 \
    cache_size=2048 \
    max_memory_length=2048 \
    run_name_suffix=llama32-1b-5t-cs2048 \
    wandb_project=memory_evolution_hf \
    wandb_group_name=namm-training
```

- **Estimated time**: ~8 hours on RTX 3090 Ti
- **VRAM usage**: ~14 GB peak

### Cache size 4096

```bash
python scripts/run_namm.py \
    run@_global_=namm_bam_i1_llama32_1b_5t \
    filter_by_length=8192 \
    cache_size=4096 \
    max_memory_length=4096 \
    batch_size=2 \
    eval_max_batch_size=2 \
    run_name_suffix=llama32-1b-5t-cs4096 \
    wandb_project=memory_evolution_hf \
    wandb_group_name=namm-training
```

- **Note**: `batch_size=2` required — batch_size=4 OOMs with cache_size=4096 on 24 GB GPUs
- **Estimated time**: ~14 hours on RTX 3090 Ti
- **VRAM usage**: ~20 GB peak

## What Each CLI Override Does

| Override | Purpose |
|---|---|
| `run@_global_=namm_bam_i1_llama32_1b_5t` | Load the 5-task NAMM config |
| `filter_by_length=8192` | Max RoPE position embeddings (GPU memory cap) |
| `cache_size=N` | KV cache budget — how many tokens the policy retains |
| `max_memory_length=N` | Must match cache_size — physical KV buffer limit |
| `batch_size=2` | (cs4096 only) Reduce batch to avoid OOM |
| `eval_max_batch_size=2` | (cs4096 only) Match batch_size for eval |
| `run_name_suffix=...` | Appended to wandb run name and output directory |
| `wandb_project=memory_evolution_hf` | WandB project name |
| `wandb_group_name=namm-training` | WandB group for organizing runs |

## Resuming Interrupted Runs

By default, `scratch=true` starts fresh. To resume from a checkpoint:

```bash
python scripts/run_namm.py \
    run@_global_=namm_bam_i1_llama32_1b_5t \
    ... \
    scratch=false
```

This loads `latest.pt` from the output directory and continues from the last saved iteration. The best checkpoint (`ckpt.pt`) is only overwritten if `val_tasks_aggregate` improves.

## Output Structure

```
experiments/namm_only_runs/memory_evolution_hf/namm-training/
  rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-csN/
    1337/
      ckpt.pt              # Best checkpoint (highest val_tasks_aggregate)
      latest.pt            # Last checkpoint
      rng_ckpt.pt          # RNG state for best checkpoint
      rng_latest.pt        # RNG state for last checkpoint
      iter_25.pt           # Periodic checkpoints (every 25 iters)
      iter_50.pt
      ...
      eval_0.json          # Eval metrics at step 0
      eval_10.json         # Eval metrics at step 10
      ...
      eval_200.json        # Final eval metrics
      .hydra/config.yaml   # Full resolved config
```

## Monitoring

- **WandB**: Watch `val_tasks_aggregate` — higher is better
- **Key metric**: `val_tasks_aggregate` is the aggregated F1 across all 5 val tasks
- **Best checkpoint**: Auto-saved when `val_tasks_aggregate` hits a new high

## Training Dataset

5-task LongBench QA subset (prompts in [4096, 6500] tokens):

| Task | Train | Val | Test |
|---|---|---|---|
| `lb/qasper` | 60 | 13 | 14 |
| `lb/2wikimqa` | 56 | 12 | 12 |
| `lb/qasper_e` | 77 | 16 | 17 |
| `lb/hotpotqa_e` | 51 | 10 | 12 |
| `lb/2wikimqa_e` | 62 | 13 | 14 |
| **Total** | **306** | **64** | **69** |

## Using Checkpoints Downstream

### For LoRA fine-tuning (M4-frozen)

```bash
python scripts/run_lora.py \
    --config scripts/lora_rh_m4_instruct_5t.yaml \
    --init_from experiments/.../ckpt.pt \
    --run_name my_lora_run \
    --wandb_project memory_evolution_hf
```

### For standalone evaluation

```bash
python scripts/run_namm.py \
    run@_global_=namm_bam_i1_llama32_1b_5t \
    filter_by_length=8192 \
    cache_size=N \
    max_memory_length=N \
    init_from=experiments/.../ckpt.pt \
    max_iters=0
```
