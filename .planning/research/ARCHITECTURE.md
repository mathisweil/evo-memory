# Architecture: Gradient-Based LoRA Integration with NAMM

**Domain:** Gradient LoRA training atop existing NAMM CMA-ES system
**Researched:** 2026-03-03
**Confidence:** HIGH (based on direct codebase analysis of all relevant files)

---

## Context: What Changed in v2.0

The v1.0 architecture doc (same file, now superseded) described ES-based LoRA training
(OpenES / EggRoll perturbation of the LoRA flat vector). That approach was superseded by a
group decision: standard gradient-based LoRA finetuning answers the research questions more
directly and is far simpler to implement correctly.

The v1.0 LoRA seam infrastructure (PEFT injection, flat-vector API, checkpoint I/O) is
**preserved and reused** — the seam is still needed. Only the training algorithm changes:
perturbation-and-tell is replaced by forward-backward-optimizer.step().

---

## Existing Architecture (What Stays)

```
main.py  (Hydra entrypoint)
  └── MemoryTrainer  (memory_trainer.py)
        ├── MemoryEvolution / CMA_ES  (memory_evolution/cma_es.py)
        │     └── ask() / tell() — evolves NAMM params, gradient-free
        ├── WrappedLlamaForCausalLM  (memory_llms/llama.py)
        │     ├── LlamaMemoryModel
        │     │     └── LlamaMemoryAttention (x16 layers)
        │     │           └── q_proj, v_proj (nn.Linear)
        │     │                 └── [LoRA A/B wrappers, injected by PEFT]
        │     ├── apply_lora_adapters()  — injects PEFT, already in v1.0
        │     └── forward() — has CrossEntropyLoss path when labels= given
        ├── MemoryHFEvaluator  (memory_evaluator.py)
        │     └── _encode_and_generate() — drives generation + scoring
        └── TaskSampler  (task_sampler.py)
              └── evaluate() — LongBench task sampling + scoring
```

Key property of the existing system: **everything runs under `@torch.no_grad()`**.
`MemoryTrainer.__init__`, `_train_step`, `_evaluate`, and `train` are all decorated.
Gradient-based LoRA training **cannot share this training loop** — it requires gradients
to flow through the forward pass and into LoRA A/B parameters.

---

## Architecture Decision: New Class vs. Extend MemoryTrainer

**Decision: New `LoRAGradTrainer` class, not an extension of `MemoryTrainer`.**

Rationale:
- `MemoryTrainer._train_step()` is decorated `@torch.no_grad()` and its entire
  logic is structured around the CMA-ES ask/tell loop. Unwinding the no_grad
  decorator and replacing the inner loop structure is a refactor that risks
  breaking the working NAMM-only path.
- `MemoryTrainer.train()` is also `@torch.no_grad()`. Adding gradient training
  inside this loop requires careful context manager wrapping around every
  LoRA gradient step, which is fragile.
- The gradient trainer needs an `optimizer.step()` / `optimizer.zero_grad()` lifecycle,
  a `DataLoader` (not `TaskSampler.evaluate()`), and `loss.backward()`. These are
  fundamentally incompatible with the existing population-evaluation loop.
- A new class keeps the NAMM CMA-ES path unchanged and testable in isolation.
  It can delegate to the existing evaluator for NAMM-compatible eval scoring.

**`LoRAGradTrainer` reuses:**
- `WrappedLlamaForCausalLM` (same model object, already has LoRA seam)
- `MemoryHFEvaluator` (for LongBench eval — delegate, don't duplicate)
- `_save_ckpt` / `_load_ckpt` logic (copy or inherit from MemoryTrainer)
- Hydra config system, wandb integration patterns

**`LoRAGradTrainer` adds:**
- `torch.utils.data.DataLoader` over LongBench training documents
- `AdamW` optimizer over LoRA parameters only (`requires_grad=True`)
- Training loop with `loss.backward()`, gradient clipping, `optimizer.step()`
- Optional NAMM activation during forward pass (controlled by config flag)
- Analysis hooks for attention entropy and layer-wise token retention

---

## System Overview (v2.0)

```
+-------------------------------------------------------------+
|                        main.py                              |
|   (Hydra; builds components; routes to trainer by mode)     |
+---------------------+---------------------------------------+
                       |
         +-------------+---------------+
         |                             |
+--------+----------+     +------------+-----------+
| LoRAGradTrainer   |     |   MemoryTrainer         |
| (new, gradient)   |     |   (existing, CMA-ES)    |
|                   |     |                         |
| AdamW optimizer   |     | CMA_ES ask()/tell()     |
| DataLoader        |     | @torch.no_grad()        |
| loss.backward()   |     | population loop         |
+---------+---------+     +------------+-----------+
          |                            |
          +-------------+--------------+
                         |
           +-------------+-------------+
           |                           |
+----------+----------+   +------------+----------+
| WrappedLlama        |   | MemoryHFEvaluator     |
| ForCausalLM         |   | (shared by both       |
|                     |   |  trainers for eval)   |
| LlamaMemoryModel    |   +----------+------------+
| + PEFT LoRA A/B     |              |
| + NAMM eviction     |   +----------+------------+
|   (optional)        |   | TaskSampler           |
+---------------------+   | (LongBench tasks)     |
                          +-----------------------+
```

**m1 / m2 / m3 / m4 pipeline orchestration lives in `main.py`**, not inside either
trainer. Each method is a sequence of trainer invocations:
- m1: `LoRAGradTrainer(namm_active=False)` only
- m2: `MemoryTrainer` only (existing, already done)
- m3: `LoRAGradTrainer(namm_active=False)` → then `MemoryTrainer`
- m4: `MemoryTrainer` → then `LoRAGradTrainer(namm_active=True)`

---

## Component Responsibilities

| Component | Responsibility | Status |
|-----------|---------------|--------|
| `LoRAGradTrainer` | Gradient LoRA training loop; Adam; DataLoader; NTP loss; analysis logging | **New file** |
| `LongBenchNTPDataset` | Tokenizes LongBench docs into (input_ids, labels) pairs; left-truncates to max_length | **New file** |
| `WrappedLlamaForCausalLM` | Forward pass with NAMM eviction; returns `loss` when `labels=` given; LoRA seam | Existing (unchanged) |
| `MemoryTrainer` | CMA-ES NAMM evolution; population eval; all runs under no_grad | Existing (unchanged) |
| `MemoryHFEvaluator` | LongBench scoring (generation + F1 / exact match) | Existing (delegated to) |
| `TaskSampler` | LongBench dataset management, sample indexing | Existing (delegated to) |
| `memory_llms/llama.py` | `apply_lora_adapters()`, `has_lora_adapters()`, forward with NTP loss | Existing (no change needed) |
| `memory_llms/base.py` | `get_lora_params_flat()`, `set_lora_params()` (still useful for eval) | Existing (no change needed) |
| `cfgs/trainer/lora_grad.yaml` | Hydra config for `LoRAGradTrainer` | **New config** |
| `cfgs/run/m1_*.yaml` | Run config for m1 (LoRA only, no NAMM) | **New config** |
| `cfgs/run/m3_*.yaml` | Run config for m3 pipeline (LoRA then NAMM) | **New config** |
| `cfgs/run/m4_*.yaml` | Run config for m4 pipeline (NAMM then LoRA+NAMM) | **New config** |

---

## New Component: `LoRAGradTrainer`

### Location

`memory_trainer.py` or a separate `lora_grad_trainer.py`. The latter is preferred:
adding 400+ lines of gradient training code into the existing 1300-line `memory_trainer.py`
would make it unwieldy. New file keeps the CMA-ES trainer readable.

### Constructor Signature

```python
class LoRAGradTrainer:
    def __init__(
        self,
        model: WrappedLlamaForCausalLM,
        tokenizer,
        trainer_config: LoRATrainerConfig,   # new dataclass
        wandb_config: WandbConfig,            # reuse existing
        namm_active: bool = False,            # m4 vs m1/m3
        device: str = 'cuda',
    ):
```

`namm_active=False` freezes the NAMM memory policy (full cache, no eviction) during LoRA
training. `namm_active=True` applies eviction every forward pass — this is the m4 condition
that tests whether NAMM's presence during finetuning changes the outcome.

### `LoRATrainerConfig` dataclass

```python
@dataclass
class LoRATrainerConfig:
    out_dir: str
    max_steps: int                  # gradient steps (not iters)
    learning_rate: float            # AdamW lr; start with 1e-4
    weight_decay: float             # 0.01
    max_grad_norm: float            # gradient clipping; 1.0
    batch_size: int                 # sequences per gradient step; 1-4 on 4070Ti
    gradient_accumulation_steps: int  # accumulate over N batches before step
    max_seq_len: int                # truncate docs to this; 1024 matches NAMM cs
    eval_interval: int              # eval every N steps
    log_interval: int
    always_save_checkpoint: bool
    init_from: Optional[str]        # path to NAMM checkpoint to warm-start from
    dtype: str                      # 'bfloat16' for forward; LoRA params stay float32
    log_analysis_metrics: bool      # whether to log attn entropy + retention
    analysis_interval: int          # how often to log analysis metrics
```

---

## New Component: `LongBenchNTPDataset`

### Why a New DataLoader is Needed

The existing `TaskSampler` feeds documents into `MemoryHFEvaluator._encode_and_generate()`,
which runs **generation** (autoregressive token-by-token generation for evaluation). For NTP
training we need **teacher-forced forward passes**: feed the full document as `input_ids`
and `labels=input_ids` shifted by one, compute `CrossEntropyLoss` over all positions.

The existing evaluator path cannot be reused for this because:
1. It calls `model.generate()`, which runs under `@torch.no_grad()` and is non-differentiable.
2. It scores on the task metric (F1, exact match), not NTP loss.
3. The batch structure is per-generation, not per-document.

### Dataset Design

```python
class LongBenchNTPDataset(torch.utils.data.Dataset):
    """Wraps a LongBench task's context documents as NTP training sequences.

    Each item is a (input_ids, attention_mask) pair.
    Labels are derived from input_ids shifted by 1 (standard NTP).
    Documents are left-truncated to max_seq_len tokens.
    Only the context (question text) is used — not the answer — to avoid
    reward-hacking on the specific answer format.
    """
    def __init__(self, task_name: str, tokenizer, max_seq_len: int = 1024,
                 split: str = 'train', seed: int = 1337):
        ...

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],      # [seq_len]
            'attention_mask': self.attention_masks[idx],  # [seq_len]
            'labels': self.input_ids[idx].clone(),  # same as input_ids for NTP
        }
```

**DataLoader instantiation:**
```python
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=trainer_config.batch_size,
    shuffle=True,
    collate_fn=pad_collate_fn,   # left-pad to batch max length
    drop_last=True,              # avoid variable-size last batch
)
```

**Left-truncation to `max_seq_len`:** LongBench documents run 4k–32k tokens. At
`max_seq_len=1024`, each forward pass fits in one NAMM cache window. This matches
the NAMM training conditions and avoids OOM on 4070Ti.

---

## NTP Loss Flow Through NAMM Eviction

This is the critical integration question for m4.

### Forward Pass in `WrappedLlamaForCausalLM.forward()`

The existing forward pass already computes NTP loss when `labels` is passed (lines 465-476
of `memory_llms/llama.py`). The NAMM eviction happens **after** the Transformer forward
pass and **after** loss computation (lines 478-517). So the loss gradient does not flow
through the NAMM eviction decision.

```
input_ids + labels
    ↓
LlamaMemoryModel.forward()          ← standard transformer forward
    ↓                               ← q_proj/v_proj have LoRA wrappers; grad flows here
logits
    ↓
CrossEntropyLoss(logits, labels)    ← loss computed here
    ↓
loss.backward()                     ← grad flows through LoRA A/B only
    ↓ (separately, in forward())
memory_policy.update_cache()        ← NAMM eviction; NOT on backward path
```

**Gradient isolation is correct behavior:** LoRA params receive gradient signal from NTP
prediction quality. NAMM eviction affects which KV tokens are available for future
generation steps (in the KV cache), but does not affect the current-document NTP loss
because the entire document is fed in one pass (teacher forcing, all tokens present).

**NAMM's role in m4 NTP training:** When `namm_active=True`, NAMM evicts KV entries
**between chunks** if the document is processed in multiple chunks. If the whole document
is fed in a single forward pass (max_seq_len ≤ 1024), NAMM eviction applies at
`memory_policy_fixed_delay` boundaries, evicting earlier tokens from the cache before
later chunks arrive. This creates a different training signal than full-cache (m1/m3):
the model must predict tokens with a smaller effective context window.

**Practical implication:** For m4 to meaningfully differ from m3, `max_seq_len` should be
long enough that NAMM eviction actually fires (i.e., `max_seq_len > cache_size`).
Recommend `max_seq_len=2048` for m4 with `cache_size=128` so NAMM is actively evicting
during training. For m1/m3, `namm_active=False` disables eviction.

### Chunked Processing (When max_seq_len > 1024)

For documents longer than one NAMM cache window, the LoRA gradient trainer must process
documents in chunks and accumulate loss across chunks before calling `optimizer.step()`.
This mirrors NAMM's evaluation mode where documents are chunked to fit the KV cache.

```python
# Pseudo-code for chunked NTP with NAMM (m4 condition)
total_loss = 0
past_key_values = None
model.memory_policy.reset()              # clear KV cache state

for chunk_start in range(0, seq_len, chunk_size):
    chunk_ids = input_ids[:, chunk_start:chunk_start+chunk_size]
    chunk_labels = labels[:, chunk_start:chunk_start+chunk_size]

    with torch.enable_grad():            # override outer no_grad if present
        out = model(
            input_ids=chunk_ids,
            labels=chunk_labels,
            past_key_values=past_key_values,
            apply_memory_policy=(trainer_config.namm_active),
        )
    total_loss += out.loss / num_chunks
    past_key_values = out.past_key_values
    # NAMM eviction already happened inside model.forward() above

total_loss.backward()
optimizer.step()
```

**Important:** `past_key_values` must NOT be detached between chunks if gradients need
to flow through the KV cache. However, since NAMM eviction does not participate in
backprop and LoRA parameters are only in the projection matrices (not the cache),
**detaching past_key_values is safe and recommended** — it prevents exploding
computation graphs over long documents:

```python
past_key_values = detach_kv_cache(past_key_values)  # stops grad at chunk boundary
```

This is consistent with how language model pretraining handles long sequences
(truncated BPTT). The NTP loss signal within each chunk is sufficient.

---

## Multi-Stage Pipeline Orchestration (m3, m4)

The pipeline coordination for m3 and m4 lives in `main.py`, not inside a trainer.
Each stage is a complete trainer invocation with checkpoint handoff.

### m3: LoRA first, then NAMM

```
Stage 1: LoRAGradTrainer
  - init_from: null (fresh LLaMA base weights)
  - namm_active: false (full cache, no NAMM during LoRA training)
  - saves checkpoint: exp_local/.../m3_stage1/ckpt.pt
    → contains: lora_state_dict, lora_config

Stage 2: MemoryTrainer (existing)
  - init_from: exp_local/.../m3_stage1/ckpt.pt
  - loads LoRA weights into model before NAMM training starts
  - NAMM CMA-ES trains on top of LoRA-finetuned model
  - saves checkpoint: exp_local/.../m3_stage2/ckpt.pt
    → contains: evolution_state, lora_state_dict, lora_config
```

The existing `MemoryTrainer._load_ckpt` already handles LoRA state loading (Phase 2
work). No changes needed to MemoryTrainer for m3.

### m4: NAMM first, then LoRA with NAMM active

```
Stage 1: MemoryTrainer (existing)
  - init_from: null or existing best NAMM checkpoint
  - no LoRA (standard NAMM CMA-ES)
  - saves checkpoint: exp_local/.../m4_stage1/ckpt.pt
    → contains: evolution_state (best NAMM params)

Stage 2: LoRAGradTrainer
  - init_from: exp_local/.../m4_stage1/ckpt.pt
  - namm_active: true (NAMM evicts KV cache during LoRA training)
  - loads NAMM params from checkpoint; freezes NAMM; trains only LoRA
  - saves checkpoint: exp_local/.../m4_stage2/ckpt.pt
    → contains: evolution_state (frozen NAMM), lora_state_dict, lora_config
```

In Stage 2 of m4, `LoRAGradTrainer` must:
1. Load the NAMM checkpoint via `_load_ckpt` logic
2. Set the NAMM params to the best saved params (`evolution_algorithm.best_params`)
3. Freeze NAMM: `model.memory_policy.requires_grad_(False)` (already gradient-free)
4. Set `model.training_mode()` so LoRA params receive gradient

### `main.py` Pipeline Routing

```python
# Conceptual routing in main.py (under Hydra cfg.method key)
if cfg.method == 'm1':
    lora_trainer = LoRAGradTrainer(model, namm_active=False, ...)
    lora_trainer.train()

elif cfg.method == 'm3':
    lora_trainer = LoRAGradTrainer(model, namm_active=False, ...)
    lora_trainer.train()  # saves stage-1 ckpt
    namm_trainer = MemoryTrainer(init_from=lora_trainer.ckpt_path, ...)
    namm_trainer.train()

elif cfg.method == 'm4':
    namm_trainer = MemoryTrainer(init_from=None, ...)
    namm_trainer.train()  # saves stage-1 ckpt
    lora_trainer = LoRAGradTrainer(
        init_from=namm_trainer.ckpt_path, namm_active=True, ...)
    lora_trainer.train()
```

---

## Analysis Metrics: Hooks into Forward Pass

The study requires attention entropy and layer-wise token retention during LoRA training.
These are logged during eval (not during every training step, to avoid overhead).

### Attention Entropy

`WrappedLlamaForCausalLM.forward()` already collects attention weights when
`output_attentions=True` is passed (line 282 of llama.py). The attention weights tensor
shape is `[batch, num_heads, seq_len, seq_len]` per layer.

**Hook point:** After the evaluator runs a full document eval forward pass:
```python
# In LoRAGradTrainer._compute_analysis_metrics()
with torch.no_grad():
    out = model(input_ids, output_attentions=True, apply_memory_policy=False)
    # out.attentions: tuple of [batch, heads, seq, seq] per layer
    for layer_idx, attn in enumerate(out.attentions):
        # average attention entropy across batch and heads
        probs = attn.float().softmax(dim=-1)
        entropy = -(probs * probs.log().clamp(min=-1e9)).sum(dim=-1).mean()
        wandb.log({f'analysis/attn_entropy_layer{layer_idx}': entropy.item()})
```

### Layer-Wise Token Retention

Token retention measures what fraction of KV tokens NAMM retains per layer after
eviction. This is already tracked by `ParamMemoryPolicy.record_eval_stats` when set to
True (see `MemoryTrainer.__init__` line 244). `LoRAGradTrainer` can use the same flag:

```python
if trainer_config.log_analysis_metrics:
    model.memory_policy.record_eval_stats = True
    # run eval with NAMM active
    eval_stats = model.get_param_stats()  # returns per-layer retention rates
    wandb.log({f'analysis/{k}': v for k, v in eval_stats.items()})
```

**Analysis logging should run only at `eval_interval` steps, not every gradient step.**
Attention weight collection doubles peak VRAM usage on long sequences.

---

## Integration Points: New vs. Modified vs. Untouched

| Component | Action | Notes |
|-----------|--------|-------|
| `lora_grad_trainer.py` | **New file** | `LoRAGradTrainer` class + `LoRATrainerConfig` |
| `lora_ntp_dataset.py` | **New file** | `LongBenchNTPDataset` + `pad_collate_fn` |
| `main.py` | **Modify** | Add pipeline routing for m1/m3/m4; conditional trainer instantiation |
| `cfgs/trainer/lora_grad.yaml` | **New config** | `LoRATrainerConfig` Hydra defaults |
| `cfgs/run/m1_*.yaml` | **New config** | m1 method: LoRA only, no NAMM |
| `cfgs/run/m3_*.yaml` | **New config** | m3 method: LoRA stage + NAMM stage |
| `cfgs/run/m4_*.yaml` | **New config** | m4 method: NAMM stage + LoRA+NAMM stage |
| `memory_trainer.py` | **Unchanged** | NAMM CMA-ES path untouched |
| `memory_llms/llama.py` | **Unchanged** | LoRA seam, forward with loss already in place |
| `memory_llms/base.py` | **Unchanged** | `get_lora_params_flat`, `set_lora_params` still valid |
| `memory_evaluator.py` | **Unchanged** | `LoRAGradTrainer` delegates to it for LongBench eval |
| `task_sampler.py` | **Unchanged** | Used by evaluator; gradient trainer uses its own DataLoader |
| `memory_evolution/` | **Unchanged** | CMA-ES only; no LoRA ES needed in v2 |
| `memory_policy/` | **Unchanged** | NAMM eviction logic untouched |

---

## Suggested Build Order (Dependency-First)

```
1. LongBenchNTPDataset
   └── Required by: LoRAGradTrainer
   └── No dependencies on other new components
   └── Test: DataLoader yields (input_ids, labels) of correct shape; truncation works

2. LoRAGradTrainer (namm_active=False, m1 mode)
   └── Required by: m1 run, m3 stage 1
   └── Depends on: LongBenchNTPDataset, existing LoRA seam (Phase 2 done)
   └── Test: 10-step training run, loss decreases, LoRA params non-zero after

3. LoRAGradTrainer._save_ckpt / _load_ckpt
   └── Required by: m3/m4 checkpoint handoff
   └── Depends on: existing memory_trainer.py _save_ckpt structure (reuse or copy)
   └── Test: save then load produces bit-identical LoRA weights

4. m1 run config + main.py routing (single-stage)
   └── Test: full m1 run on QASPER, 200 steps, wandb logs, checkpoint saved

5. LoRAGradTrainer (namm_active=True, m4 mode)
   └── Depends on: working LoRAGradTrainer (step 2), NAMM checkpoint load (step 3)
   └── Test: forward pass with NAMM active, loss still decreases, no KV cache corruption

6. m4 pipeline (main.py two-stage coordination)
   └── Depends on: MemoryTrainer (existing), LoRAGradTrainer (step 5)
   └── Test: stage-1 NAMM ckpt loads into stage-2 LoRA trainer; stage-2 trains successfully

7. m3 pipeline (reverse order from m4)
   └── Depends on: steps 3, 4
   └── Test: LoRA ckpt from stage 1 loads into MemoryTrainer for NAMM training

8. Analysis metrics (attention entropy + token retention)
   └── Depends on: working LoRAGradTrainer
   └── Can be added to any completed trainer; test: wandb shows non-constant values

9. Secondary experiment configs (E1-E5)
   └── Depends on: all primary trainers working
   └── E1 (iterative interleaving): requires loop over LoRAGradTrainer + MemoryTrainer
```

---

## Data Flow: m4 Gradient Step

```
LongBenchNTPDataset
    ↓ (input_ids, labels, attention_mask)  [batch, seq_len]
    ↓
optimizer.zero_grad()
    ↓
with torch.enable_grad():
    WrappedLlamaForCausalLM.forward(
        input_ids=input_ids,
        labels=labels,
        apply_memory_policy=True   ← NAMM active (m4)
    )
    ↓
    LlamaMemoryModel.forward()
        for each layer:
            q = q_proj(x)          ← LoRA wraps q_proj: q = W_base·x + B·A·x
            k = k_proj(x)          ← no LoRA on k_proj (unless config says so)
            v = v_proj(x)          ← LoRA wraps v_proj
            attn_out = SDPA(q, k, v, past_kv)
        ↓ (all 16 layers)
    logits = lm_head(hidden_states)
    loss = CrossEntropyLoss(logits[:-1], labels[1:])    ← NTP loss
    ↓
    memory_policy.update_cache(past_kv, ...)   ← NAMM evicts; NOT on grad path
    ↓
    return CausalMemoryLMOutputWithPast(loss=loss, ...)
    ↓
loss.backward()
    ↓ (gradient flows through lm_head, transformer layers, LoRA A/B matrices only)
grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, max_norm)
optimizer.step()   ← updates only LoRA A and B matrices
```

**Grad isolation confirmed:** `model.memory_policy` parameters have `requires_grad=False`
(NAMM is gradient-free by design). `model.model.base_model` parameters also have
`requires_grad=False` (PEFT sets this). Only `requires_grad=True` parameters (LoRA A/B)
receive gradient updates.

---

## VRAM and Gradient Overhead

| Component | VRAM Impact | Notes |
|-----------|------------|-------|
| LoRA parameter gradients (r=4, q+v) | +3.4 MB | 2× param size for grad buffer |
| AdamW optimizer state (m+v for LoRA) | +6.8 MB | 2× param size again |
| Activation memory (seq_len=1024, batch=1) | +200-400 MB | Scales with seq_len |
| Activation memory (seq_len=2048, batch=1) | +400-800 MB | For m4 with longer seqs |
| Base model (bfloat16, frozen) | 2.5 GB | Unchanged |
| NAMM policy state | ~50 MB | Unchanged |

**Total additional VRAM for gradient LoRA training:** ~400-800 MB activation overhead
plus ~10 MB gradient/optimizer buffers. This is dominated by activations, not LoRA.

For `batch_size=1, seq_len=1024`: fits comfortably in 4070Ti (12 GB VRAM).
For `batch_size=4, seq_len=1024`: expect ~2-3 GB activation memory — likely OOM.
**Recommended: `batch_size=1, gradient_accumulation_steps=4`** to match effective
batch_size=4 without OOM. This is the standard workaround for VRAM-constrained gradient
training.

For `seq_len=2048` (m4 with long docs): use `batch_size=1, gradient_accumulation_steps=4`
and consider gradient checkpointing to trade compute for memory if needed.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Extending MemoryTrainer with Gradient Training

**What people do:** Add `if gradient_mode:` branches inside `_train_step`,
add `optimizer.step()` calls inside a `@torch.no_grad()` decorated method,
or wrap the inner loop in `torch.enable_grad()`.

**Why it's wrong:** `@torch.no_grad()` on `_train_step` means gradients are disabled
before the function is entered. `torch.enable_grad()` as a context manager *inside* a
`@torch.no_grad()` function does re-enable gradients in PyTorch, but this creates an
implicit dependency between two decorators that is hard to reason about and will break
if someone changes the outer decorator. The semantic contract of `_train_step` is
"no gradients; population-based ES" — violating it by adding gradient paths leads to
bugs.

**Do this instead:** New class with clean gradient semantics throughout.

### Anti-Pattern 2: Merging LoRA Adapters During Training

**What people do:** Call `model.merge_adapter()` to fuse LoRA into base weights,
then train on the merged model, then reset.

**Why it's wrong:** `merge_adapter()` permanently modifies base weights in-place.
For the m4 condition the NAMM policy scores tokens against query states from the
pre-merge model — merging invalidates the NAMM params stored in the CMA-ES checkpoint.
Also, merged weights cannot be "un-merged" to recover the LoRA decomposition.

**Do this instead:** Always train with LoRA in the unmerged state. The overhead is
negligible (one extra matmul per attention layer).

### Anti-Pattern 3: Detaching KV Cache Between Chunks with Gradient Accumulation

**What people do:** Forget to detach `past_key_values` between chunks when accumulating
gradients over multiple chunks, letting the computation graph grow across the entire
document.

**Why it's wrong:** For a 2048-token document with `chunk_size=512`, this creates
a 4-chunk computation graph that PyTorch holds in memory simultaneously. VRAM usage
scales with document length, causing OOM on long documents.

**Do this instead:** Always detach `past_key_values = detach_kv_cache(past_kv)` between
chunks. LoRA gradient signal comes from within-chunk NTP loss only — this is sufficient
because each chunk contains 512 tokens of meaningful prediction context.

### Anti-Pattern 4: Using TaskSampler.evaluate() for NTP Training Data

**What people do:** Reuse `TaskSampler.evaluate()` to get training batches by calling
it with `train=True`, passing the result as `labels`.

**Why it's wrong:** `TaskSampler.evaluate()` drives `_model_generate()` (autoregressive
generation), not a teacher-forced forward pass. It returns generated text, not logits.
There is no `labels` interface on the generation path.

**Do this instead:** Build `LongBenchNTPDataset` directly from the raw HuggingFace
datasets (same `load_dataset('THUDM/LongBench', ...)` call already in `task_sampler.py`
line 87) but return the **context field** as the training document, tokenize it, and
use it as `input_ids` and `labels`.

### Anti-Pattern 5: Setting `lora_dropout > 0` in gradient training

**What people do:** Use the same `LoraConfig` from v1.0 (`lora_dropout=0.0` for ES)
and forget to reconsider it for gradient training.

**Why it's wrong:** Unlike ES training (no gradient flow, so dropout is irrelevant),
gradient training benefits from regularization. With `lora_dropout=0.0` on small QASPER
training sets (~1000 documents), LoRA will overfit within a few hundred steps.

**Do this instead:** Set `lora_dropout=0.1` (standard for gradient LoRA) when calling
`apply_lora_adapters()` from `LoRAGradTrainer`. This requires adding a `dropout` argument
to `apply_lora_adapters()` with a default of `0.0` (preserving v1.0 behavior for ES use).

---

## Sources

- Direct codebase analysis: `memory_trainer.py`, `memory_llms/llama.py`,
  `memory_llms/base.py`, `memory_evaluator.py`, `task_sampler.py`,
  `memory_evolution/base.py`, `memory_policy/auxiliary_losses.py`
- Project context: `.planning/PROJECT.md`, `.planning/REQUIREMENTS.md`,
  `.planning/ROADMAP.md`
- PyTorch documentation on `@torch.no_grad()` vs `torch.enable_grad()` context manager
  semantics (training cutoff: Aug 2025)
- PEFT 0.11.1 `LoraConfig` parameter reference (lora_dropout for gradient training)

---

*Architecture research for: Gradient-based LoRA integration with NAMM CMA-ES*
*Researched: 2026-03-03*
