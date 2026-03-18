# Merge Plan: Gradient LoRA Fine-Tuning into the `tpu` Branch

**Goal:** Bring the gradient-based LoRA training code from commit `eda3f19` into the
current `tpu` branch so that both ES and LoRA-gradient can be run as parallel
fine-tuning strategies sharing the same data pipeline, model construction, and
evaluation protocol. This enables a fair head-to-head comparison.

**Source:** commit `eda3f19698cc3db26cf057c03ce7014a4c7152ed` (44 commits diverged)
**Target:** `tpu` branch HEAD (92 commits diverged)
**Common ancestor:** `21b2880`

---

## Why not `git merge`?

A raw merge would produce 50+ conflicts because the `tpu` branch reorganized the
entire package layout:

| Old path (at `eda3f19`)      | New path (at `tpu` HEAD)     |
|------------------------------|------------------------------|
| `memory_llms/`               | `namm/llms/`                 |
| `memory_policy/`             | `namm/policy/`               |
| `memory_evaluator.py`        | `namm/evaluator.py`          |
| `memory_trainer.py`          | `namm/trainer.py`            |
| `task_sampler.py`            | `namm/tasks.py`              |
| `main.py`                    | `scripts/run_namm.py`        |
| `utils.py`                   | `utils/helpers.py`           |
| `utils_hydra.py`             | `utils/hydra_helpers.py`     |

Git sees these as delete+create (not renames) because the content also changed
significantly. Every file the LoRA branch touched will conflict against a "deleted"
file. A manual cherry-pick-and-adapt is far cleaner.

---

## Architecture: Where LoRA code lives

```
evo-memory/
  grad_lora_finetuning/       # NEW package (parallel to es_finetuning/)
    __init__.py
    trainer.py                 # LoRAGradTrainer + LoRATrainerConfig
    datasets.py                # LongBenchNTPDataset, LongBenchSFTDataset, collate fns
  namm/
    llms/
      base.py                  # MODIFIED: add LoRA flat-vector API
      llama.py                 # MODIFIED: add apply_lora_adapters(), skip_lm_head
    tasks.py                   # MODIFIED: upgrade to 3-way train/val/test split
  scripts/
    run_lora.py                # NEW entrypoint (parallel to run_es.py)
    lora_default.yaml          # NEW config defaults (parallel to es_default.yaml)
```

**Why a separate `grad_lora_finetuning/` package rather than adding to `es_finetuning/`:**

ES and LoRA-gradient are fundamentally different optimization paradigms. ES is
black-box (perturb weights, evaluate via generation, update); LoRA-gradient is
white-box (forward pass with autograd, backprop through LoRA A/B matrices). They
share no training loop logic. Putting them in separate packages keeps each clean
and self-contained — exactly how `es_finetuning/` is structured today.

---

## Step-by-step plan

### Phase 1: Model-level LoRA support (`namm/llms/`)

#### 1a. Add LoRA flat-vector API to `namm/llms/base.py`

Port three methods from `eda3f19:memory_llms/base.py` into the current
`MemoryModelWrapper`:

- `_get_lora_params()` — returns `[p for p in self.model.parameters() if p.requires_grad]`
- `get_lora_params_flat()` — concatenate + detach + CPU + float32
- `set_lora_params(flat_vec)` — restore from flat vector with dtype/device cast

Also add `import torch` (currently absent from the `tpu` branch's `base.py`).

**Why these live on `MemoryModelWrapper`:** Both the ES trainer and the LoRA
trainer need to get/set adapter weights via the same model object. The flat-vector
API is the contract between the model and any optimizer (ES or gradient).

#### 1b. Add `apply_lora_adapters()` to `namm/llms/llama.py`

Port from `eda3f19:memory_llms/llama.py`:

- `apply_lora_adapters(rank, target_modules, lora_alpha, lora_dropout)`:
  - Creates `LoraConfig` (from `peft`) with `bias='none'`
  - Wraps `self.model` with `get_peft_model()`
  - Freezes `self.lm_head` (PEFT only freezes the backbone it wraps)
  - **Forces float32** on all LoRA params (prevents bfloat16 underflow at
    ES sigma=0.001 and ensures gradient numerical stability)
  - Stores `_lora_rank` and `_lora_target_modules` for checkpoint save/load
- `has_lora_adapters()` — returns `hasattr(self, '_lora_rank')`

Also add the `peft` import (`from peft import LoraConfig, get_peft_model`).

**Why force float32 on LoRA params:** Both training paradigms need it. ES uses
sigma=0.001 perturbations that would underflow in bfloat16 (smallest representable
delta is ~0.0078). Gradient training accumulates small updates that also suffer
from bfloat16 rounding. The base model stays in bfloat16 to save memory.

#### 1c. Add `skip_lm_head` support to `WrappedLlamaForCausalLM.forward()`

Port the `skip_lm_head` parameter from `eda3f19`:

- When `True`, return `logits=None, loss=None` and skip the `lm_head` projection
- The caller (LoRA trainer) applies `lm_head` in 512-token chunks to avoid
  materializing the full `[B, seq_len, vocab_size]` tensor (~2.5 GB for
  seq_len=4096, vocab_size=128k)
- Hidden states are accessible via `output_hidden_states=True`

**Why chunked lm_head:** The Llama 3.2 1B vocabulary is 128,256 tokens. A single
forward pass with `[1, 4096, 128256]` float32 logits requires ~2 GB. The chunked
approach applies lm_head to 512 tokens at a time, reducing peak memory by ~8x.
This is only needed for LoRA gradient training (ES never computes logits or loss).

**Caution:** Must verify this doesn't break the existing ES/NAMM forward path.
The parameter should default to `False` so all existing codepaths are unaffected.

#### 1d. Port hidden-state accumulation in split processing

When NAMM processes long sequences in splits (chunks), the LoRA trainer needs the
full sequence's hidden states concatenated. Port the hidden-state accumulation
logic from `eda3f19:memory_llms/llama.py` that concatenates `hidden_states` across
splits when `output_hidden_states=True`.

**Why:** The LoRA NAMM-active (m4-frozen) path does two-phase forward: context
under `no_grad` (builds KV cache with NAMM eviction), then answer with gradients.
The answer phase needs hidden states from the model to compute loss via the
chunked `lm_head`.

---

### Phase 2: Dataset layer (`grad_lora_finetuning/datasets.py`)

#### 2a. Port `LongBenchNTPDataset` and `LongBenchSFTDataset`

Create `grad_lora_finetuning/datasets.py` containing both dataset classes and
their collate functions, ported from `eda3f19:lora_ntp_dataset.py` and
`eda3f19:lora_sft_dataset.py`.

**Import fixups:**
- `dataset2prompt.json` path: change from `LongBench/config/dataset2prompt.json`
  to `data/longbench/dataset2prompt.json` (matching the `tpu` branch's data
  directory layout)

#### Key design decision: Why separate datasets exist alongside `TaskSampler`

The LoRA datasets and `TaskSampler` serve fundamentally different purposes:

| Aspect | `TaskSampler` (ES) | LoRA Datasets |
|--------|-------------------|---------------|
| **Training paradigm** | Black-box (generate, score) | White-box (teacher forcing, backprop) |
| **Output** | Prompt strings for generation | `input_ids` / `labels` tensors with loss masking |
| **Batching** | Evaluator handles batching internally | PyTorch `DataLoader` with custom collation |
| **Data consumed** | `context + question` via prompt template | `context` only (NTP) or `template + answer` (SFT) |
| **Loss computation** | F1/ROUGE on generated text vs gold answers | Cross-entropy on next-token prediction |

Both load from the same underlying `THUDM/LongBench` HuggingFace datasets. Both
use the same prompt templates from `data/longbench/dataset2prompt.json`. The
difference is entirely in how the data is consumed by the training loop.

**What MUST be shared:** The train/test split indices. If LoRA trains on sample
indices `[0, 1, 3, 7, ...]` and ES trains on `[0, 2, 5, 8, ...]`, the comparison
is unfair. Both must use identical splits. See Phase 3 for how this is enforced.

**What CANNOT be shared:** The PyTorch `Dataset`/`DataLoader` pipeline. ES doesn't
use `DataLoader` at all — it calls `task_sampler.resample_requests()` to pick
indices, then `task_sampler.evaluate()` to run generation. LoRA needs a proper
`DataLoader` for batched gradient training with padding, collation, and
teacher-forced labels.

---

### Phase 3: Train/test split alignment (`namm/tasks.py`)

#### 3a. Upgrade `_build_split()` from 2-way to 3-way

The current `tpu` branch has a 2-way split (train/test). The LoRA branch at
`eda3f19` has a 3-way split (train/val/test) via `apply_train_val_test_split()`.

LoRA gradient training needs a validation set for:
- Early stopping (tracking best checkpoint by val F1)
- Hyperparameter selection without contaminating the test set

ES also benefits from this (currently it only has train/test, so "full eval" on
the test set is used for final reporting but there's no held-out val for
checkpoint selection).

**Changes to `namm/tasks.py`:**

1. Add `val_split` parameter to `TaskSampler.__init__()` (default `None` for
   backward compat — existing ES configs won't break)
2. Modify `_build_split()` to produce 3 index sets when `val_split` is provided:
   `_train_idxs_per_task`, `_val_idxs_per_task`, `_test_idxs_per_task`
3. Add `get_split_indices(split)` method (from `eda3f19`'s `task_sampler.py`)
   that returns the appropriate index dict for `'train'`/`'val'`/`'test'`
4. Update `resample_requests_lb()` and `evaluate_lb_tasks_for_pop()` to
   respect the val split (when `train=False`, use test; add a `split` parameter
   or keep using test as the default non-train split for backward compat)

**Why the split lives in `TaskSampler`, not in the LoRA dataset:** The split must
be computed once and shared across both training paradigms. `TaskSampler` is the
single source of truth for "which samples exist and how they're partitioned."
The LoRA dataset classes receive the split indices from `TaskSampler` and filter
their samples accordingly. This guarantees that LoRA sample index 42 and ES
sample index 42 refer to the same LongBench document.

**Backward compatibility:** When `val_split=None` (the default), behavior is
identical to today's 2-way split. Existing ES configs and scripts are unaffected.

---

### Phase 4: LoRA trainer (`grad_lora_finetuning/trainer.py`)

#### 4a. Port `LoRATrainerConfig` dataclass

Port from `eda3f19:lora_grad_trainer.py` with these adjustments:

- Remove `cache_dir` field (use the model's existing cache — no need for a
  separate LoRA-specific cache directory)
- Keep all hyperparameter fields: `max_seq_len`, `num_epochs`, `batch_size`,
  `gradient_accumulation_steps`, `learning_rate`, `weight_decay`,
  `max_grad_norm`, `warmup_ratio`, `namm_active`, `eval_interval`,
  `log_interval`, `sft_mode`, `train_frac`, `val_frac`,
  `max_conditioning_length`

#### 4b. Port `LoRAGradTrainer` class

Port from `eda3f19:lora_grad_trainer.py` (~1072 lines) with these changes:

**Import updates:**
- `from memory_trainer import WandbConfig` → `from namm.trainer import WandbConfig`
- `from lora_ntp_dataset import ...` → `from grad_lora_finetuning.datasets import ...`
- `from lora_sft_dataset import ...` → `from grad_lora_finetuning.datasets import ...`

**Functional changes:**
- Use `TaskSampler.get_split_indices()` (Phase 3) instead of the old
  `apply_train_val_test_split()` / `get_split_indices()` API. The trainer should
  receive a pre-split `TaskSampler` rather than splitting it itself.
- Checkpoint save/load: keep the existing contract (`lora_state_dict`,
  `lora_config`, optimizer/scheduler state, `step_num`). This is independent
  of ES checkpoints (which use delta format).

**What stays the same:**
- Two-phase forward for NAMM-active (m4-frozen): context under `no_grad`,
  answer with gradients
- Chunked cross-entropy via `skip_lm_head=True`
- AdamW optimizer with cosine warmup scheduler
- Gradient clipping via `clip_grad_norm_`
- Per-layer token retention logging (ANLYS-01)
- Validation F1 tracking and best-checkpoint selection
- Wandb logging

**The trainer does NOT run under `torch.no_grad()`** — this is the critical
difference from the ES path. The entrypoint script (Phase 5) must ensure no
global `no_grad` context wraps the LoRA training loop.

---

### Phase 5: Entrypoint script (`scripts/run_lora.py`)

#### 5a. Create `scripts/run_lora.py`

Modeled on `scripts/run_es.py` — same structure, same experiment management
pattern, but dispatching to `LoRAGradTrainer` instead of `ESTrainer`.

**Flow:**
1. Parse CLI args (from `lora_default.yaml` + overrides)
2. Load Hydra config via `load_hydra_config()` (reused from `run_es.py` or
   extracted to a shared utility)
3. Construct model via `make_eval_model()` (from `run_namm.py`)
4. Load NAMM checkpoint if `namm_active=True` (same logic as `run_es.py`)
5. Create `TaskSampler` via `make_task_sampler()` with 3-way split params
6. Apply token-based filtering (same as `run_es.py`)
7. Cast model to bfloat16, move to device
8. Call `model.apply_lora_adapters(rank, targets, alpha, dropout)`
9. Enable gradient checkpointing if `namm_active=False`
10. Construct `LoRAGradTrainer` — **outside** `torch.no_grad()`
11. Call `trainer.train()`

**Why a separate entrypoint instead of adding a flag to `run_es.py`:**

`run_es.py` runs its entire training loop under `torch.no_grad()` (line-level
wrapping in `main()`). The LoRA trainer must NOT be under `no_grad` — the
autograd graph must survive from the loss through the LoRA A/B matrices for
backprop. Mixing these in one script would require fragile context manager
toggling. Two scripts keeps it clean.

Additionally, the CLI args are substantially different: ES has `sigma`, `alpha`,
`population_size`, `noise_mode`; LoRA has `learning_rate`, `num_epochs`,
`batch_size`, `gradient_accumulation_steps`, `sft_mode`, etc. A shared parser
would be unwieldy.

**Shared utilities to extract (optional):** `load_hydra_config()`,
`get_or_create_experiment()`, `claim_run_gcs()`, and
`get_base_llm_param_names()` are currently in `run_es.py`. If `run_lora.py`
needs experiment management and GCS integration (likely, for fair tracking),
these should be extracted to a shared module (e.g., `scripts/experiment_utils.py`
or into `es_finetuning/` with a more generic name). This is a minor refactor and
can be deferred — for the initial port, `run_lora.py` can duplicate the necessary
functions or import directly from `run_es.py`.

---

### Phase 6: Configuration (`scripts/lora_default.yaml`)

#### 6a. Create `scripts/lora_default.yaml`

Parallel to `scripts/es_default.yaml`. Contains all LoRA-specific defaults:

```yaml
# ── LoRA hyperparameters ──
lora_rank: 8
lora_target_modules: [q_proj, v_proj]
lora_alpha: null         # null = defaults to rank
lora_dropout: 0.0
learning_rate: 2e-4
weight_decay: 0.01
max_grad_norm: 1.0
warmup_ratio: 0.03
num_epochs: 3
batch_size: 1
gradient_accumulation_steps: 16
max_seq_len: 3500
sft_mode: false

# ── NAMM ──
namm_active: false
namm_checkpoint: null
run_config: namm_bam_i1_llama32_1b
cache_size: null

# ── Data (shared with ES for fair comparison) ──
filter_by_tokens: null
filter_answers_by_tokens: 64
train_split: 0.8
val_split: 0.1           # LoRA needs val for early stopping
split_seed: 42

# ── Evaluation ──
eval_interval: 40
log_interval: 10
batch_size_eval: null     # inference batch size for F1 eval

# ── Checkpointing & GCS ──
gcs: true
checkpoint_every: 0       # LoRA saves at eval_interval, not rolling
resume_checkpoint: null
```

**Why shared data params match `es_default.yaml`:** `filter_answers_by_tokens: 64`,
`split_seed: 42`, and the same `run_config` ensure both methods train and evaluate
on identical data subsets. This is the core requirement for a fair comparison.

---

### Phase 7: Tests

#### 7a. Port test files

Port from `eda3f19`:

- `tests/test_lora_seam.py` → tests for Phase 1 (LoRA API on `MemoryModelWrapper`)
  - Update imports: `memory_llms` → `namm.llms`
  - Update `utils_hydra.LlamaCompatModel` → `utils.hydra_helpers.LlamaCompatModel`
    (verify this class exists on `tpu` branch; if not, port it)

- `tests/test_lora_grad_trainer.py` → tests for Phase 4 (LoRA trainer)
  - Update imports: `lora_grad_trainer` → `grad_lora_finetuning.trainer`
  - Update `memory_trainer.WandbConfig` → `namm.trainer.WandbConfig`
  - Update fixture to use new package paths

**Why port tests:** These are correctness gates — they verify gradient flow
through LoRA params, base weight stability under `set_lora_params`, round-trip
checkpoint save/load, and loss decrease over 10 steps. Without them, we can't
be confident the port didn't break anything.

---

### Phase 8: Documentation (`docs/`)

Write two user-facing guides that mirror the existing ES documentation style and
structure (`docs/es-ft-guide.md` and `docs/es-ft-namm-guide.md`).

#### 8a. `docs/lora-grad-ft.md` — LoRA Gradient Fine-Tuning Guide

Parallel to `docs/es-ft-guide.md`. Covers LoRA-only training (m1 condition:
full KV cache, no NAMM eviction).

**Structure (matching `es-ft-guide.md`):**

1. **Header & context** — one-line summary, link to PEFT/LoRA paper, comparison
   table of LoRA hyperparameters vs the `eda3f19` branch's original settings.
   Explain this is the gradient-based counterpart to ES fine-tuning: same model,
   same data, different optimizer.

2. **What is LoRA gradient fine-tuning?** — Explain low-rank adapters (A/B
   matrices injected into attention projections), parameter efficiency (~0.5%
   of base model params trainable), and how it differs from ES (white-box
   autograd vs black-box reward signal). Explain NTP mode (all tokens supervised)
   vs SFT mode (answer-only loss masking with chat templates).

3. **Parameters** — two tables:
   - **LoRA hyperparameters (argparse):** `--lora_rank`, `--lora_target_modules`,
     `--lora_alpha`, `--lora_dropout`, `--learning_rate`, `--weight_decay`,
     `--max_grad_norm`, `--warmup_ratio`, `--num_epochs`, `--batch_size`,
     `--gradient_accumulation_steps`, `--max_seq_len`, `--sft_mode`,
     `--eval_interval`, `--log_interval`, `--train_split`, `--val_split`,
     `--resume_checkpoint`, etc.
   - **Model/task config (from Hydra):** same as ES guide (`cache_size`,
     `max_new_tokens`, `max_position_id`, etc.)

4. **Experiment hierarchy** — same directory structure as ES experiments,
   but with `lora_grad/` method directory. Describe checkpoint contents
   (`lora_state_dict`, `lora_config`, optimizer/scheduler state, `step_num`).

5. **LoRA training workflow** — pseudocode showing:
   - Model construction + LoRA injection (`apply_lora_adapters`)
   - DataLoader creation (NTP or SFT dataset)
   - Training loop: forward with `skip_lm_head` → chunked CE loss → backward →
     gradient clipping → optimizer step
   - Periodic val F1 evaluation via `TaskSampler.evaluate()` (generation-based,
     same as ES)
   - Best checkpoint selection by val F1

6. **Step-by-step breakdown** — explain the chunked lm_head approach (why it's
   needed for 128k vocab), the float32 LoRA invariant, gradient checkpointing.

7. **Quick start** — copy-paste commands:
   ```
   python scripts/run_lora.py --config scripts/lora_default.yaml --run_name m1_test
   ```
   And a variant with custom hyperparams.

8. **Data pipeline** — explain how `TaskSampler` provides the train/val/test
   split, how `LongBenchNTPDataset`/`LongBenchSFTDataset` consume the same
   underlying data, and how filtering (`filter_answers_by_tokens`) is shared
   with ES.

9. **Comparing with ES** — side-by-side table of what's shared (model, data,
   split, eval) and what differs (optimizer, loss computation, training loop,
   memory requirements). This is the key section for understanding the
   experimental comparison.

10. **Troubleshooting** — common issues: OOM (reduce `max_seq_len`, enable
    gradient checkpointing), loss NaN (check float32 on LoRA params), slow
    convergence (increase `learning_rate`, check `sft_mode`).

#### 8b. `docs/lora-grad-ft-namm.md` — Combined LoRA + NAMM Guide

Parallel to `docs/es-ft-namm-guide.md`. Covers LoRA training with frozen NAMM
eviction active (m4-frozen condition).

**Structure (matching `es-ft-namm-guide.md`):**

1. **Header & context** — one-line summary emphasizing this is the gradient
   counterpart to the ES+NAMM experiment. Link to both LoRA and NAMM papers.

2. **How both systems interact** — table showing:
   | System | What it optimises | Optimiser | During combined run |
   | NAMM | Scoring network | CMA-ES | Frozen |
   | LoRA | Adapter weights (~0.5% of LLM) | AdamW | Active (backprop) |

   Contrast with ES+NAMM where ES optimises ALL base LLM weights (~100%).

3. **Two-level optimization view** — pseudocode showing the outer loop (LoRA
   gradient updates on adapter weights) and inner loop (frozen NAMM scoring +
   eviction during forward pass). Explain the two-phase forward: context under
   `no_grad` to build KV cache with NAMM eviction, answer phase with gradients
   for backprop through LoRA.

4. **The cooperation hypothesis** — same framing as `es-ft-namm-guide.md`:
   can the LoRA adapters learn to produce attention patterns that cooperate
   with the frozen eviction policy? Key difference: LoRA only modifies a tiny
   subspace (low-rank projections in q/v), so the adaptation signal is more
   constrained than ES's full-weight modification. This is the core
   scientific question for the comparison.

5. **Parameters (combined system)** — merge of LoRA hyperparameters and
   NAMM-specific parameters:
   - `--namm_active true`
   - `--namm_checkpoint` (required)
   - `--cache_size` (override)
   - All LoRA params from 8a, noting `max_seq_len` may need to be shorter
     (2048 vs 3500) for NAMM compatibility

6. **What the NAMM checkpoint contains** — same content as `es-ft-namm-guide.md`
   (scoring network weights, EMA buffers, CMA-ES state).

7. **Combined workflow** — pseudocode showing:
   - Stage 1: Train NAMM (reference to `namm-guide.md`)
   - Stage 2: LoRA fine-tuning with frozen NAMM
   - The two-phase forward in detail (context under no_grad → NAMM eviction →
     answer with gradients → chunked CE loss → backward)
   - Per-layer token retention logging (ANLYS-01)

8. **What happens during each forward pass** — diagram showing token flow
   through NAMM eviction and LoRA-modified attention, matching the style of
   the ES+NAMM guide's forward pass diagram.

9. **Quick start** — copy-paste commands:
   ```
   python scripts/run_lora.py --config scripts/lora_default.yaml \
       --namm_active true \
       --namm_checkpoint path/to/ckpt.pt \
       --run_name m4_frozen_test
   ```

10. **Key differences from LoRA-only (m1)** — table:
    - Gradient checkpointing disabled (incompatible with NAMM's `use_cache=True`)
    - Shorter `max_seq_len` (NAMM adds overhead)
    - Two-phase forward vs single forward
    - Lower learning rate recommended (1e-4 vs 2e-4)

11. **Key differences from ES+NAMM** — table:
    - LoRA modifies ~0.5% of params vs ES modifying 100%
    - LoRA uses backprop vs ES using reward-noise correlation
    - LoRA sees loss gradients directly vs ES seeing only scalar reward
    - LoRA needs train/val split; ES uses test set for final eval only
    - LoRA training is much faster per iteration but may need more iterations

**Why two separate guides rather than one:** This mirrors the ES documentation
pattern (`es-ft-guide.md` + `es-ft-namm-guide.md`). The NAMM interaction adds
substantial complexity (two-phase forward, frozen checkpoint loading, cache size
constraints, gradient checkpointing incompatibility) that would clutter a guide
focused on LoRA basics. Users running m1 (LoRA-only) shouldn't need to wade
through NAMM details.

---

## Execution order and dependencies

```
Phase 1a (base.py LoRA API)
Phase 1b (llama.py apply_lora_adapters)  ← depends on 1a
Phase 1c (llama.py skip_lm_head)
Phase 1d (llama.py hidden-state accum)
    |
    v
Phase 2 (datasets.py)                    ← independent of Phase 1
    |
    v
Phase 3 (tasks.py 3-way split)           ← independent of Phase 1-2
    |
    v
Phase 4 (trainer.py)                     ← depends on 1, 2, 3
    |
    v
Phase 5 (run_lora.py)                    ← depends on 4
Phase 6 (lora_default.yaml)              ← depends on 5
Phase 7 (tests)                          ← depends on 1, 4
Phase 8 (documentation)                  ← depends on 5, 6

Phases 1, 2, 3 can be done in parallel.
Phases 5, 6, 7, 8 can be done in parallel after Phase 4.
Phase 8 should be written last (after code is finalized) since the
guides reference exact CLI args, config keys, and code paths.
```

---

## Risk assessment

### Low risk
- **Phase 2 (datasets):** Self-contained new files with no impact on existing code.
- **Phase 6 (yaml config):** New file, no impact.
- **Phase 7 (tests):** New files, no impact on production code.
- **Phase 8 (documentation):** New files, no impact on code.

### Medium risk
- **Phase 1a-1b (LoRA API + apply_lora_adapters):** Modifies `namm/llms/base.py`
  and `namm/llms/llama.py`. These are critical files used by every training path.
  However, the LoRA methods are purely additive (new methods, no changes to
  existing methods) and are only called when LoRA is explicitly requested.
  **Mitigation:** Run existing ES training end-to-end after these changes to
  verify no regression.

- **Phase 3 (3-way split):** Modifies `namm/tasks.py`. The change adds an
  optional `val_split` parameter with `None` default, so existing 2-way behavior
  is preserved. The `_build_split()` method gains a branch for the 3-way case.
  **Mitigation:** Verify that `run_es.py` with default configs produces identical
  train/test indices before and after the change.

### Higher risk
- **Phase 1c-1d (skip_lm_head, hidden-state accumulation):** Modifies the
  `forward()` path in `llama.py`. `skip_lm_head` defaults to `False` so existing
  paths are unaffected, but the hidden-state accumulation in split processing
  touches the NAMM forward loop.
  **Mitigation:** Test NAMM eval (recency baseline and NAMM BAM) before and after
  to verify identical scores. Use `torch.testing.assert_close()` on outputs if
  possible.

- **Phase 4 (trainer):** Large file (~1000 lines) with many integration points.
  The main risk is subtle API mismatches between the ported trainer and the
  restructured codebase (e.g., `TaskSampler` API changes, `WandbConfig` field
  changes).
  **Mitigation:** Port the tests (Phase 7) first or in parallel, and run them
  after the trainer is ported.

---

## `peft` dependency

The LoRA code requires `peft` (Parameter-Efficient Fine-Tuning by Hugging Face).
This must be added to `requirements.txt`. The version at `eda3f19` used
`peft>=0.6.0`. Current latest is ~0.15.x.

**Action:** Add `peft>=0.6.0` to `requirements.txt`. No pinned version needed
since the API (`LoraConfig`, `get_peft_model`) has been stable.

---

## Files modified (existing)

| File | Change type | Risk |
|------|------------|------|
| `namm/llms/base.py` | Add 3 methods + `import torch` | Low |
| `namm/llms/llama.py` | Add `apply_lora_adapters()`, `has_lora_adapters()`, `skip_lm_head`, hidden-state accum | Medium |
| `namm/tasks.py` | Add optional 3-way split, `get_split_indices()` | Medium |
| `requirements.txt` | Add `peft>=0.6.0` | Low |

## Files created (new)

| File | Description |
|------|------------|
| `grad_lora_finetuning/__init__.py` | Package init with exports |
| `grad_lora_finetuning/trainer.py` | LoRAGradTrainer + LoRATrainerConfig |
| `grad_lora_finetuning/datasets.py` | NTP + SFT datasets and collate functions |
| `scripts/run_lora.py` | CLI entrypoint for LoRA training |
| `scripts/lora_default.yaml` | Default configuration |
| `tests/test_lora_seam.py` | LoRA API correctness tests |
| `tests/test_lora_grad_trainer.py` | Trainer correctness tests |
| `docs/lora-grad-ft.md` | LoRA gradient fine-tuning guide (mirrors `es-ft-guide.md`) |
| `docs/lora-grad-ft-namm.md` | Combined LoRA + NAMM guide (mirrors `es-ft-namm-guide.md`) |

---

## Validation checklist

After all phases are complete:

- [ ] `python scripts/run_es.py --config scripts/es_default.yaml --run_name test`
      runs without error (ES regression check)
- [ ] `python scripts/run_lora.py --config scripts/lora_default.yaml --run_name test`
      runs and loss decreases
- [ ] Both use identical train/test splits (print and compare indices)
- [ ] Both use identical data filtering (same sample counts after filtering)
- [ ] `pytest tests/test_lora_seam.py` passes (LoRA API contract)
- [ ] `pytest tests/test_lora_grad_trainer.py` passes (gradient flow, checkpointing)
- [ ] NAMM eval scores unchanged before/after `llama.py` modifications
- [ ] `docs/lora-grad-ft.md` covers all CLI args from `scripts/lora_default.yaml`
- [ ] `docs/lora-grad-ft-namm.md` covers two-phase forward, NAMM checkpoint loading,
      and differences from both LoRA-only and ES+NAMM
- [ ] Both guides include working quick-start commands that match `run_lora.py` CLI
