# Architecture: Joint NAMM + LoRA-ES Training

**Domain:** Extending existing NAMM CMA-ES system to jointly train LoRA adapters
**Researched:** 2026-03-02
**Confidence:** HIGH (based on direct codebase analysis)

---

## Existing Architecture (What Stays the Same)

```
main.py
  └── MemoryTrainer (memory_trainer.py ~1400 lines)
        ├── CMA_ES (memory_evolution/cma_es.py)  ← evolves NAMM params
        ├── WrappedLlamaForCausalLM (memory_llms/llama.py)  ← frozen LLM
        │     └── LlamaMemoryModel
        │           └── LlamaMemoryAttention (per layer)
        │                 └── ParamMemoryPolicy.set_params()  ← NAMM injects here
        ├── MemoryHFEvaluator (memory_evaluator.py)  ← scores on QASPER/NarrativeQA
        └── TaskSampler (task_sampler.py)
```

**Population evaluation loop (current):**
```
for member in range(pop_size):
    set_memory_params(namm_params[member])  ← apply NAMM
    fitness[member] = evaluate()
cma_es.tell(fitness)
namm_params = cma_es.ask()
```

---

## New Architecture (What Changes)

### Q1 — Where does LoRA get injected?

**After** `WrappedLlamaForCausalLM.__init__()`, before trainer loop:
```python
lora_config = LoraConfig(
    r=lora_rank, lora_alpha=2*lora_rank,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0, bias="none",
    task_type=TaskType.CAUSAL_LM, init_lora_weights=True
)
model = get_peft_model(wrapped_llama, lora_config)
```

**Why post-construction:** `WrappedLlamaForCausalLM.__init__()` replaces standard attention with `LlamaMemoryAttention` modules. PEFT targets `nn.Linear` layers by name — `q_proj`, `v_proj` exist inside these custom attention modules as plain `nn.Linear` in their `forward()` calls, so PEFT finds and wraps them correctly after construction. The NAMM policy reads from `q_proj` outputs via `set_memory_params()` — the LoRA adapter is transparent to this pathway.

**Integration point:** Add `apply_lora_adapters(lora_config)` method to `WrappedLlamaForCausalLM`.

---

### Q2 — LoRA parameter extraction, perturbation, re-application

**Extraction:**
```python
# In memory_llms/base.py: MemoryModelWrapper
def get_lora_params_flat(self) -> torch.Tensor:
    lora_params = [p for n, p in self.named_parameters() if "lora_" in n]
    return torch.nn.utils.parameters_to_vector(lora_params).detach().cpu()

def set_lora_params(self, flat_vec: torch.Tensor) -> None:
    lora_params = [p for n, p in self.named_parameters() if "lora_" in n]
    torch.nn.utils.vector_to_parameters(flat_vec.to(self.device), lora_params)
```

**Per-member application in training loop (serial, like NAMM):**
```python
for member in range(pop_size):
    model.set_memory_params(namm_params[member])  # existing
    model.set_lora_params(lora_perturbations[member])  # new
    fitness[member] = evaluate()
```

LoRA perturbation vectors stored on CPU; move to GPU per member evaluation.
**No need for pop_size GPU copies of the full model.**

---

### Q3 — Mode A vs Mode B Architecture

**Mode B (recommended — implement first):**
```
Training loop:
  namm_members = cma_es.ask()         ← existing CMA-ES
  lora_members = lora_es.ask()        ← new LoRA_ES (sigma-perturbation)

  for member in range(pop_size):
      model.set_memory_params(namm_members[member])
      model.set_lora_params(lora_members[member])
      fitness[member] = evaluate()

  cma_es.tell(fitness)                ← existing
  lora_es.tell(fitness)               ← new (same fitness, different update rule)
```

`LoRA_ES` class (new: `memory_evolution/lora_es.py`):
```python
class LoRA_ES(MemoryEvolution):
    def __init__(self, lora_flat_size, sigma=0.001, pop_size=8):
        self.mean = torch.zeros(lora_flat_size)
        self.sigma = sigma
        self.pop_size = pop_size

    def ask(self):  # antithetic pairs
        noise = [torch.randn_like(self.mean) for _ in range(self.pop_size // 2)]
        return [self.mean + self.sigma * e for e in noise] + \
               [self.mean - self.sigma * e for e in noise]

    def tell(self, fitness):  # weighted mean update
        ...  # OpenES update rule
```

**Mode A (diagonal CMA-ES — deferred, higher complexity):**
```
namm_params || lora_flat → concatenated vector → single CMA-ES
```
Requires: switching `CMA_ES` to diagonal covariance variant (xNES / diagonal-CMA).
CMA-ES `C` matrix replaced by `diag_C` vector → O(N) not O(N²) memory.
Only feasible at small total dims (<5000). Even r=1 q_proj = 65K dims → diagonal CMA-ES still needed.

**Mode selector:** `joint_es_mode` Hydra key ∈ {`A`, `B`, `namm_only`, `lora_only`}.

---

### Q4 — Memory Footprint

| Component | Size | Where |
|-----------|------|-------|
| LoRA weights (r=4, q+v, 16 layers, float32) | ~1.7 MB | GPU (active member) |
| LoRA population buffer (8 members, CPU) | ~13.6 MB | CPU |
| LoRA ES mean vector | ~1.7 MB | CPU |
| Existing NAMM CMA-ES C matrix (~256 dim) | ~0.25 MB | GPU |
| Base LLaMA 3.2-1B (bfloat16) | ~2.5 GB | GPU |

**Total additional VRAM: <2 MB.** VRAM budget is dominated by model activations during generation (unchanged). **No additional VRAM pressure from LoRA-ES in Mode B.**

Follows existing NAMM pattern: pop_size member weights stored on CPU, moved to GPU one at a time.

---

### Q5 — Checkpoint Extension

**Extend `_save_ckpt` (memory_trainer.py ~line 1064):**
```python
ckpt = {
    # existing keys:
    'raw_evolution_algorithm': cma_es.state_dict(),
    'rng_state': ...,
    # new keys:
    'lora_flat': model.get_lora_params_flat(),       # LoRA weights (mean)
    'lora_es_state': lora_es.state_dict(),            # sigma + ES mean
    'lora_config': dataclasses.asdict(lora_config),   # rank, targets
    'joint_es_mode': cfg.joint_es_mode,
}
```

**Extend `_load_ckpt`:**
```python
if 'lora_flat' in ckpt:
    model.set_lora_params(ckpt['lora_flat'])
    lora_es.load_state_dict(ckpt['lora_es_state'])
# else: NAMM-only checkpoint, LoRA stays at initialization
```

**Backward compatible:** existing NAMM checkpoints load without LoRA state — LoRA starts from PEFT initialization (B=0 → identity adapter).

---

## Component Boundaries (What Changes vs. Stays)

| Component | Change Required | Notes |
|-----------|----------------|-------|
| `memory_evolution/cma_es.py` | None (Mode B) / Diagonal variant (Mode A) | Mode B: CMA-ES is unchanged |
| `memory_evolution/lora_es.py` | **New file** | OpenES antithetic; `ask()`/`tell()` interface |
| `memory_llms/llama.py` | Add `apply_lora_adapters()` method | Called once from trainer init |
| `memory_llms/base.py` | Add `get_lora_params_flat()`, `set_lora_params()` | Flat vector interface |
| `memory_trainer.py` | Extend `_train_step`, `_save_ckpt`, `_load_ckpt` | ~50-100 lines new |
| `cfgs/evolution/lora_es.yaml` | **New config** | sigma, antithetic flag |
| `cfgs/run/joint_*.yaml` | **New run configs** | joint_es_mode, lora_rank, lora_targets |
| `main.py` | Conditional LoRA init + LoRA_ES instantiation | ~30 lines |
| `task_sampler.py`, `memory_evaluator.py` | None | Fitness eval unchanged |
| `memory_policy/` | None | NAMM policy untouched |

---

## Data Flow for Joint ES Iteration

```
1. CMA-ES.ask()     → namm_param_matrix [pop_size, namm_dim]  (GPU/CPU)
2. LoRA_ES.ask()    → lora_param_matrix [pop_size, lora_dim]  (CPU)
3. For member i:
   a. model.set_memory_params(namm_param_matrix[i])
   b. model.set_lora_params(lora_param_matrix[i].to(device))
   c. fitness[i] = evaluator.evaluate(model, task_samples)
4. CMA-ES.tell(fitness)   → update NAMM mean + covariance
5. LoRA_ES.tell(fitness)  → update LoRA mean (weighted average of perturbations)
6. namm_param_matrix = CMA-ES.ask()  (next iter)
7. lora_param_matrix = LoRA_ES.ask() (next iter)
```

---

## Suggested Build Order

1. **LoRA injection + extraction/injection methods** — base seam, everything depends on this
2. **Unit tests**: injection succeeded, base weights stable across pop eval, param counts match
3. **Extended checkpoint** — correctness blocker for any training run
4. **LoRA_ES class** — OpenES antithetic, unit test `ask()`/`tell()`
5. **Extend `_train_step`** — Mode B first (no CMA-ES changes needed)
6. **`lora_only` ablation config** — verify LoRA ES trains without NAMM
7. **Joint Mode B training config** — NAMM + LoRA together
8. **Diagnostic logging** — component norms, fitness correlations
9. **Mode A** — only after Mode B produces baseline results

---

## Critical Integration Risk

`LlamaMemoryAttention.forward()` passes `q_proj`, `v_proj` outputs into `ParamMemoryPolicy.forward()`. With PEFT's LoRA wrapper, the forward pass becomes: `q_out = lora_B(lora_A(q_proj_weight @ x)) + q_proj_weight @ x`. This output flows unchanged into NAMM's memory scoring. NAMM sees perturbed query representations — this is **desired behavior** (LoRA changes which tokens NAMM considers important). But it must be verified that `LlamaMemoryAttention.forward()` doesn't do any in-place ops or caching that would break PEFT's hook mechanism.

**Test:** Single forward pass with LoRA enabled, check gradient flow through LoRA params (even though we're not using gradients, PEFT's hooks must not error).
