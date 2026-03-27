# Combined LoRA Fine-Tuning + Frozen NAMM

Fine-tuning LLaMA 3.2-1B-Instruct via LoRA adapters while NAMM's pre-trained eviction policy manages the KV cache. This is the **m4-frozen condition**: can the base model learn to produce better answers when its KV cache is actively compressed by a frozen scoring network?

This parallels the ES+NAMM combined system (`es-ft-namm-guide.md`), replacing the gradient-free ES optimiser with standard backpropagation through LoRA adapters. The key challenge is that NAMM's eviction decisions are non-differentiable, requiring a two-phase forward pass to route gradients only through the answer tokens.

---

## How Both Systems Interact

| System   | Optimises               | Optimiser | Combined run                |
| -------- | ----------------------- | --------- | --------------------------- |
| NAMM     | Scoring net (~100s)     | CMA-ES    | **Frozen** -- weights fixed |
| LoRA     | A/B in q/v_proj (~6M)   | AdamW     | **Active** -- backprop      |
| Base LLM | All 1.24B params        | --        | **Frozen** -- LoRA only     |

The NAMM scoring network is trained first (Stage 1), then frozen. LoRA fine-tuning (Stage 2) modifies only the low-rank adapter matrices while the frozen NAMM policy continues to run inside every attention layer, evicting tokens from the KV cache.

### Two-level view

```
Outer loop (LoRA):  optimise adapter weights via AdamW to minimise NTP/SFT loss
                    loss is computed ONLY on answer tokens (after context processed by NAMM)

Inner loop (fixed): for each forward pass, NAMM scoring network phi* (frozen)
                    scores and evicts tokens from the KV cache
                    phi* was pre-trained on the base model's attention patterns
```

### What each component controls

| Component        | Parameters            | Grad?         | Role                     |
| ---------------- | --------------------- | ------------- | ------------------------ |
| LoRA A           | rank x dim, per layer | Yes (float32) | Low-rank input proj      |
| LoRA B           | dim x rank, per layer | Yes (float32) | Low-rank output proj     |
| Base LLM         | ~1.24B (all layers)   | No (bf16)     | Fixed backbone           |
| lm_head          | 128256 x 2048         | No (bf16)     | Chunked by trainer       |
| NAMM scoring net | ~hundreds of params   | No (frozen)   | Scores/evicts KV tokens  |
| NAMM EMA buffers | Norm statistics       | No (frozen)   | Embedding running stats  |

---

## Two-Phase Forward Pass

Gradient checkpointing is **incompatible** with NAMM because NAMM requires `use_cache=True` to build and evict from the KV cache. Without gradient checkpointing, storing activations for the full context (2000-4000 tokens) would exceed GPU memory. The solution is a two-phase forward:

```
Input: [context tokens ... | answer tokens]
       |<-- Phase 1 (no_grad) -->|<-- Phase 2 (grad) -->|

Phase 1 -- Context (torch.no_grad):
  - Process context tokens through the model with NAMM eviction active
  - Builds the KV cache; NAMM evicts tokens at every 256-token boundary
  - skip_lm_head=True (hidden states not needed, only KV cache)
  - No activations stored, no gradient graph
  - Output: past_key_values (the evicted KV cache)

Phase 2 -- Answer (with gradients):
  - Process answer tokens with the evicted KV cache from Phase 1
  - NAMM continues to run (apply_memory_policy=True)
  - skip_lm_head=True (trainer applies lm_head in chunks)
  - output_hidden_states=True (needed for chunked cross-entropy)
  - Gradients flow: loss -> chunked_lm_head -> hidden_states -> LoRA A/B
  - Output: hidden_states for loss computation
```

### Phase boundary alignment

The answer start position is aligned down to `chunk_align` (= `max_new_tokens`, typically 64) to ensure clean chunk boundaries:

```python
answer_mask = (labels[0] != -100)
answer_start = answer_mask.nonzero()[0].item()
context_end = (answer_start // chunk_align) * chunk_align
# Phase 1 processes input_ids[:, :context_end]
# Phase 2 processes input_ids[:, context_end:]
```

### Why gradients cannot flow through NAMM eviction

NAMM's top-k token selection during eviction is a discrete, non-differentiable operation. Tokens are either kept or evicted based on their scores -- there is no smooth gradient through this decision boundary. The two-phase split is the practical solution: gradients only flow through the answer portion where no eviction decisions are made on the current tokens.

---

## Parameters (Combined LoRA + NAMM)

### LoRA hyperparameters

| Parameter              | Default            | Meaning                       |
| ---------------------- | ------------------ | ----------------------------- |
| `--run_name`           | (required)         | Name for this run             |
| `--config`             | `None`             | YAML config file              |
| `--experiment`         | auto               | Experiment ID or creates new  |
| `--method`             | `lora_grad`        | Training method identifier    |
| `--lora_rank`          | `8`                | Rank of LoRA decomposition    |
| `--lora_target_modules`| `[q_proj, v_proj]` | Target linear layers          |
| `--lora_alpha`         | `None` (= rank)    | LoRA scaling factor           |
| `--lora_dropout`       | `0.0`              | Dropout on LoRA outputs       |
| `--learning_rate`      | `2e-4`             | AdamW learning rate           |
| `--weight_decay`       | `0.01`             | AdamW weight decay            |
| `--max_grad_norm`      | `1.0`              | Gradient clipping threshold   |
| `--warmup_ratio`       | `0.03`             | Cosine warmup fraction        |
| `--num_epochs`         | `3`                | Training epochs               |
| `--batch_size`         | `1`                | Micro-batch size              |
| `--grad_accum_steps`   | `16`               | Steps before optimizer update |
| `--max_seq_len`        | `3500`             | Max token sequence length     |
| `--sft_mode`           | `False`            | Answer-only loss masking      |

### NAMM-specific

| Parameter           | Default          | Meaning                        |
| ------------------- | ---------------- | ------------------------------ |
| `--namm_active`     | `False`          | **Must be `True`** for m4      |
| `--namm_checkpoint` | `None`           | Path to NAMM ckpt or `latest`  |
| `--run_config`      | `namm_bam_i1_*`  | Hydra config for eviction etc. |
| `--cache_size`      | `None`           | Override NAMM cache size       |

### Data filtering and splits

| Parameter                | Default | Meaning                        |
| ------------------------ | ------- | ------------------------------ |
| `--filter_by_tokens`     | `None`  | Drop samples > N tokens       |
| `--filter_answers_by_tokens` | `64`| Drop samples with long answers |
| `--train_split`          | `0.8`   | Fraction for training          |
| `--val_split`            | `0.1`   | Fraction for validation        |
| `--split_seed`           | `42`    | Deterministic split seed       |

### What the NAMM checkpoint contains

The checkpoint (`ckpt.pt`) stores:
- `evo_state['best_member']` -- flat parameter vector for the scoring network
- EMA normalisation buffers -- running statistics from the embedding module
- CMA-ES state (mean, sigma, covariance) -- not used during LoRA fine-tuning

When loaded, the scoring network parameters are set via `memory_policy.set_params_batch_idxs(np.zeros([1]))`, fixing the policy to a single parameter set for all evaluations.

---

## Quick Start

**LoRA + frozen NAMM (m4-frozen):**
```bash
python scripts/run_lora.py \
    --config scripts/lora_default.yaml \
    --run_name m4_frozen_test \
    --namm_active \
    --namm_checkpoint path/to/namm_ckpt.pt
```

**With GCS auto-download of latest NAMM checkpoint:**
```bash
python scripts/run_lora.py \
    --config scripts/lora_default.yaml \
    --run_name m4_frozen_test \
    --namm_active \
    --namm_checkpoint latest
```

**With custom cache size:**
```bash
python scripts/run_lora.py \
    --config scripts/lora_default.yaml \
    --run_name m4_cache512 \
    --namm_active \
    --namm_checkpoint latest \
    --cache_size 512
```

**SFT mode with NAMM:**
```bash
python scripts/run_lora.py \
    --config scripts/lora_default.yaml \
    --run_name m4_sft \
    --namm_active \
    --namm_checkpoint latest \
    --sft_mode
```

---

## Key Differences from LoRA-Only (m1)

| Aspect             | LoRA-only (m1)          | LoRA+NAMM (m4)               |
| ------------------ | ----------------------- | ---------------------------- |
| **NAMM eviction**  | Inactive (full cache)   | Active (256-token boundary)  |
| **Grad ckpt**      | Enabled (saves VRAM)    | **Disabled** (use_cache)     |
| **Forward pass**   | Single pass, full seq   | Two-phase: ctx + answer      |
| **Memory**         | Lower (grad ckpt)       | Higher (no ckpt, Phase 1)    |
| **KV cache reset** | Not needed              | init_buffers() per doc       |
| **skip_lm_head**   | Yes (chunked in trainer)| Yes (same)                   |
| **Loss**           | All tokens / answer     | Answer-phase only            |
| **Retention log**  | N/A                     | Per-layer ratio (ANLYS-01)   |

### Critical design constraints for NAMM mode

1. **No gradient checkpointing** -- NAMM requires `use_cache=True` to maintain the KV cache across chunks. Gradient checkpointing recomputes forward passes without cache, breaking NAMM's eviction state.
2. **No AMP / GradScaler** -- silently downcasts float32 LoRA weights to float16, causing underflow.
3. **KV cache reset between documents** -- `memory_policy.initialize_buffers()` must be called before each new document to clear stale eviction state.
4. **Two-phase forward is mandatory** -- without it, storing activations for the full context would exceed GPU memory.

---

## Key Differences from ES+NAMM

| Aspect              | ES + frozen NAMM           | LoRA + frozen NAMM          |
| ------------------- | -------------------------- | --------------------------- |
| **Optimised**       | All 1.24B base weights     | ~6M LoRA A/B (~0.5%)       |
| **Optimiser**       | NES (gradient-free)        | AdamW (gradient)            |
| **Grad thru NAMM**  | Not needed (black box)     | Cannot (two-phase fwd)      |
| **Fwd passes**      | ~6,400 (50x8x16)          | ~300-500 (3 epochs)         |
| **Bwd passes**      | 0                          | Same as fwd (answer only)   |
| **Reward signal**   | F1 on generated text       | CE on answer tokens         |
| **Thru eviction**   | Yes (reward captures)      | No (loss bypasses)          |
| **Validation**      | Baseline + final only      | Every eval_interval         |
| **Precision**       | bfloat16 throughout        | float32 LoRA + bf16 base    |
| **Convergence**     | Slow (pop=8, 1.24B-dim)   | Fast (exact, 6M-dim)       |
| **Weight drift**    | Weights change, NAMM stale | Adapters only, base kept    |

### The cooperation hypothesis (LoRA perspective)

ES modifies all base weights, which could cause the frozen NAMM policy to become stale as attention patterns drift. LoRA only modifies a low-rank subspace of the attention projections (q_proj, v_proj), preserving the base model's attention patterns more closely. This may keep the frozen NAMM policy effective for longer, potentially avoiding the need for alternating optimisation (Phase 4).

However, LoRA's loss signal does not flow through the eviction decisions. The model optimises for good answers given the evicted cache, but does not directly learn to produce attention patterns that are "compressible" under NAMM. ES, by contrast, sees the full end-to-end reward including eviction quality.

---

## Experiment Hierarchy

Results are organised under:
```
experiments/experiment_N/lora_grad/run_name/
    config.json          # full configuration snapshot (not used; see results/)
    results/lora_grad/42/
        config.yaml      # training hyperparameters
        metrics.csv      # per-step loss, grad_norm, lr
        ckpt.pt          # latest checkpoint (LoRA weights + optimizer + scheduler)
        best_ckpt.pt     # best checkpoint by val F1
        wandb_run_id.txt # wandb run ID (if enabled)
```

---

## Key Files

- `scripts/run_lora.py` -- entry point and argument parser
- `scripts/lora_default.yaml` -- default configuration
- `grad_lora_finetuning/trainer.py` -- LoRAGradTrainer (two-phase forward, chunked loss, eval, checkpointing)
- `grad_lora_finetuning/datasets.py` -- NTPDataset, SFTDataset, pad_collate_fn
- `namm/llms/llama.py` -- `apply_lora_adapters()`, `forward()` with `skip_lm_head` and `apply_memory_policy`
- `namm/tasks.py` -- TaskSampler with train/val/test split
- `es_finetuning/device.py` -- device abstraction (TPU > CUDA > CPU)
- `es_finetuning/gcs.py` -- GCS experiment management
