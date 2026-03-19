# CLAUDE.md — evo-memory-giacommo

## What this repo is

A fork of Mathis Weil's `evo-memory` codebase (branch `dev/joint-namm-lora-es`)
adapted by Romain Hautier for his MSc CSML thesis at UCL. The project studies
**NAMM (Neural Associative Memory Model)** — a learned KV-cache eviction policy
for LLMs, trained via CMA-ES (derivative-free evolutionary optimisation).

The repo supports four experiment modes on **Llama-3.2-1B-Instruct**:
1. **Full-context** — no eviction, baseline evaluation
2. **Recency** — simple evict-oldest policy (baseline)
3. **NAMM (CMA-ES)** — train the eviction scoring network with evolution
4. **LoRA fine-tuning** (m1 = LoRA only, m4 = LoRA + frozen NAMM)

## Environment

- **Cluster**: UCL CS, CentOS/RHEL, CUDA GPUs (3090/4070 Ti SUPER)
- **Shell**: `csh` — use `setenv VAR value`, not `export VAR=value`
- **Python**: 3.10
- **Critical dependency**: `transformers==4.41.2` (NOT 4.45+). The code deeply
  patches HuggingFace's `LlamaModel` internals; 4.45+ breaks DynamicCache API,
  cache_position handling, and causal mask construction.
- **PyTorch**: 2.7.0+cu128 (installed separately from PyTorch index, not from pip)
- See `SETUP_NOTES.md` for full environment setup instructions.

## Directory layout

> **After running `bash migrate.sh --apply`** the repo adopts this structure.
> Until then the legacy layout is still in place (see git history for mapping).

```
config/                         # Hydra configs (was cfgs/)
  run/
    rh_instruct_namm_train.yaml # NAMM CMA-ES training (200 iters, pop=8)
    rh_instruct_namm_eval.yaml  # NAMM eval on held-out test set
    rh_instruct_recency.yaml    # Recency baseline eval
    rh_m1_lora_instruct.yaml    # LoRA-only fine-tuning (m1)
    rh_m4_lora_instruct.yaml    # LoRA + frozen NAMM (m4)
  model/wrapped_llm/
    llama32-1b-instruct.yaml    # Model config (max_position_id=8192)
  task/rh_multi_qa.yaml         # 4-task multi-QA setup
  trainer/
    default.yaml                # NAMM trainer config
    lora_grad.yaml              # LoRA trainer config

data/                           # Dataset config JSONs (was LongBench/ + ChouBun/)
  longbench/config/             # dataset2prompt.json, dataset2maxlen.json, …
  choubun/config/

models/                         # LLM wrappers (was memory_llms/)
  base.py                       # MemoryModelWrapper abstract base
  llama.py                      # Patched LlamaForCausalLM + NAMM + SDPA

policy/                         # Eviction policy (was memory_policy/)
  base.py                       # Abstract MemoryPolicy + ParamMemoryPolicy
  base_dynamic.py               # Recency params + score normalisation
  deep.py                       # DeepMP — main NAMM eviction policy
  components.py                 # TokenEmbedding, ScoringNetwork, … (was base_deep_components.py)
  selection.py                  # BinarySelection, TopK, Dynamic (was deep_selection.py)
  auxiliary_losses.py           # Sparsity / L2 regularisation
  embedding/
    base.py                     # RecencyExponents (was deep_embedding.py)
    shared.py                   # PositionalEmbedding + clamp safety (was deep_embedding_shared.py)
    spectogram.py               # STFT AttentionSpectrogram (was deep_embedding_spectogram.py)
    wrappers.py                 # RecencyEmbeddingWrapper (was deep_embedding_wrappers.py)
  scoring/
    base.py                     # MLPScoring, GeneralizedScoring (was deep_scoring.py)
    bam.py                      # BAM scoring variant (was deep_scoring_bam.py)

evolution/                      # CMA-ES optimizer (was memory_evolution/)
  base.py                       # Abstract MemoryEvolution (ask/tell)
  cma_es.py                     # CMA-ES implementation

ops/                            # Stateless parallel ops (was stateless_parallel_modules/)
  base.py  attention.py  mlp.py

training/                       # All trainers + datasets
  namm_trainer.py               # NAMM CMA-ES training loop (was memory_trainer.py)
  lora_trainer.py               # Gradient LoRA fine-tuning (was lora_grad_trainer.py)
  datasets/
    sft.py                      # SFT dataset, answer-only masking (was lora_sft_dataset.py)
    ntp.py                      # NTP dataset (was lora_ntp_dataset.py)

evaluation/                     # Inference + task management
  evaluator.py                  # MemoryHFEvaluator (was memory_evaluator.py)
  task_sampler.py               # LongBench task loading + splits (was task_sampler.py)

metrics/                        # Metric implementations
  longbench.py                  # F1, rouge, retrieval… (was longbench_metrics.py)
  choubun.py                    # Japanese QA metrics (was choubun_metrics.py)

utils/                          # Shared utilities
  core.py                       # Tensor ops, masking, GPU utils (was utils.py)
  hydra.py                      # LlamaCompatModel, Hydra helpers (was utils_hydra.py)
  log.py                        # COLOR + logging helpers (was utils_log.py)
  longbench.py                  # Score dispatch + chat builder (was utils_longbench.py)

scripts/                        # Shell scripts (was repo root)
  run_namm_eval.sh
  run_m1sft_evals.sh
  run_all_evals.sh

main.py                         # Hydra entry point (stays at root)
run_eval.py                     # Standalone eval runner
migrate.sh                      # One-shot migration script (run once, then remove)
```

## Running experiments

All commands assume you're in the repo root with the conda env active.
Set `HYDRA_FULL_ERROR=1` for full tracebacks.

### NAMM training (CMA-ES)
```csh
setenv CUDA_VISIBLE_DEVICES 0
python main.py +run=rh_instruct_namm_train seed=42
```

### NAMM eval (after training)
Update `init_from` in the yaml first, then:
```csh
python main.py +run=rh_instruct_namm_eval seed=1337
```

### Recency baseline eval
```csh
python main.py +run=rh_instruct_recency cache_size=1024 seed=1337
```

### LoRA m1 (no NAMM)
```csh
python main.py +run=rh_m1_lora_instruct seed=1337
```

### LoRA m4 (frozen NAMM)
Update `init_from` in the yaml to the trained NAMM checkpoint, then:
```csh
python main.py +run=rh_m4_lora_instruct seed=1337
```

## Data split convention

All experiments use an **80/10/10** (train/val/test) deterministic split on
LongBench samples filtered to ≤6500 tokens. The split is computed by
`task_sampler.apply_train_val_test_split()` and is shared across NAMM training,
LoRA training, and evaluation to ensure fair comparison.

Tasks: `qasper`, `multifieldqa_en`, `hotpotqa`, `2wikimqa`.

## Architecture notes

- **LlamaCompatModel** (`utils/hydra.py`): Custom model loader that patches
  `rope_scaling` when `max_position_embeddings <= 8192`. This avoids allocating
  a huge RoPE cache (saves ~400 MB) and is a key speed optimization.

- **max_position_id** (currently 8192): Serves dual purpose:
  1. Size of NAMM's sinusoidal PositionalEmbedding table
  2. LLM's `max_position_embeddings` for RoPE
  A safety clamp in `policy/embedding/shared.py` prevents OOB access when
  recency values exceed this during long sequences with many eviction steps.

- **BinarySelection**: Keeps tokens with score ≥ 0.0. Early in CMA-ES training,
  most tokens score negative → low retention (~380/6460). Expected to improve as
  CMA-ES converges.

- **Split processing**: Long sequences are chunked by `memory_policy_fixed_delay`
  (256 tokens) for NAMM eviction. Attention mask shape must match post-eviction
  KV cache size — mismatch causes RuntimeError.

- **Eviction diagnostics**: During NAMM training, coloured terminal output shows
  retained (green) vs evicted (red strikethrough) tokens. Also logged to wandb
  as HTML via `wandb.Html`.

## Known issues

- **Attention mask off-by-one**: After NAMM eviction, the causal mask can be 1
  token shorter than the actual KV cache, causing `RuntimeError: The size of
  tensor a (N) must match the size of tensor b (N-1)`. This is a race between
  the pre-computed mask and post-eviction KV length. Under investigation.

- **CMA-ES slow convergence**: With `pop_size=8` and `samples_batch_size=2`,
  fitness signal is very noisy. `sample_D_mean` barely changes in the first 20
  steps. This matches the reference config — 200 iterations may be needed.

## Wandb

Project: `memory_evolution_hf`
Groups: `Llama-3.2-1B-Instruct/namm-replication`, `Llama-3.2-1B-Instruct/lora-replication`, `Llama-3.2-1B-Instruct/baselines`
