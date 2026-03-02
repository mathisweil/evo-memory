# Technology Stack

**Project:** Joint NAMM + LoRA-ES Training
**Researched:** 2026-03-02
**Scope:** New components only — what to ADD to the existing NAMM stack. Does not re-document PyTorch, Transformers, Hydra, wandb already present.

---

## New Stack Components

### LoRA Adapter Layer

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| PEFT (already installed) | 0.11.1 | Apply LoRA to `WrappedLlamaForCausalLM` | Already in `th2`; `LoraConfig` + `get_peft_model()` is the standard HF pattern |

**LoRA application pattern:**
```python
from peft import LoraConfig, TaskType, get_peft_model

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,                 # alpha = 2*r (no scaling amplification)
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,             # No dropout for ES — sigma handles exploration
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    init_lora_weights=True,       # B=0 init: LoRA is no-op at start (essential for ES)
)
# Apply AFTER WrappedLlamaForCausalLM is constructed
wrapped_model = get_peft_model(wrapped_llama, lora_config)
```

**Why apply after wrapper construction:** `WrappedLlamaForCausalLM.__init__` replaces attention
layers. PEFT targets `nn.Linear` by name — these exist inside the custom attention layers,
so PEFT finds them correctly after construction.

---

### Parameter Flattening (both modes)

| Technology | Version | Purpose |
|------------|---------|---------|
| `torch.nn.utils.parameters_to_vector` | PyTorch (installed) | Flatten LoRA params to 1D for ES |
| `torch.nn.utils.vector_to_parameters` | PyTorch (installed) | Write ES-perturbed vector back to LoRA params |

```python
lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
flat_lora = torch.nn.utils.parameters_to_vector(lora_params)
torch.nn.utils.vector_to_parameters(perturbed_flat, lora_params)
```

---

### Mode B: Sigma-ES for LoRA (OpenES / Antithetic Sampling)

**No new libraries needed.** Implement as `LoRA_ES(MemoryEvolution)` in `memory_evolution/lora_es.py`,
extending the existing `ask()`/`tell()` interface. Antithetic pairs (+ε, −ε) are more sample-efficient
than one-sided perturbation.

---

## Recommended LoRA Configuration Parameters

### Rank Selection by ES Mode

| Mode | Recommended Rank | Target Modules | LoRA Param Count | Rationale |
|------|-----------------|----------------|-----------------|-----------|
| Mode A (joint CMA-ES) | r=1, q_proj only | ~65k total | Requires diagonal CMA-ES — existing code caps at 40k |
| Mode B (recommended) | r=4 | q_proj, v_proj | ~327k | OpenES scales linearly; safe on 4070Ti |
| Mode B (richer) | r=8 | q_proj, v_proj, o_proj | ~1.5M | Valid if VRAM allows |

**Critical note on Mode A:** `memory_evolution/cma_es.py:36` clips covariance dim at 40k.
A joint CMA-ES over [NAMM ≈256 || LoRA ≥65k] is numerically infeasible at full covariance.
Mode A requires a **diagonal/separable CMA variant** — this is an additional research task.
**Start with Mode B.** Mode A diagonal-CMA is a later-phase research extension.

---

## Parameter Count Reference (LLaMA 3.2-1B)

16 layers, hidden_size=2048, num_kv_heads=8, head_dim=64 → v_proj/k_proj output=512 (GQA).

| LoRA Config | Params per Layer | Total (16 layers) |
|-------------|-----------------|-------------------|
| r=4, q+v_proj | 20,480 | **327,680** |
| r=4, q+k+v+o | 53,248 | 851,968 |
| r=8, q+v_proj | 40,960 | 655,360 |
| r=1, q_proj only | 4,096 | 65,536 |

---

## Sigma Initialization (Mode B)

**Recommended starting sigma: 0.001**
- Reference: `shr1ram/es-fine-tuning-paper` uses σ=0.001 for LLM weight perturbation
- At r=4, q+v (327k params): initial L2 perturbation ≈ 0.001 × √327k ≈ 0.57 — well-conditioned
- Existing NAMM: σ_init=0.065 for ~256 dims → per-param scale ≈ 0.004; LoRA σ=0.001 is same order
- If ES stagnates for 50 iters: increase σ by 10×

---

## Compute Timing Reference (4070Ti, QASPER)

**Measured from wandb run-20260225_183712 (pre-optimisation baseline):**

| Config | Runtime | Per-iter |
|--------|---------|---------|
| NAMM-only (old, pre-optimisation) | 10.7h (38,480s) | ~192s / iter |
| **NAMM-only (current, ~4x speedup via batch_size=4)** | **~2.7h** | **~48s / iter** |
| Eval only (recency/namm, cs=256-1024) | 6-8 min | — |

**Implication for joint training:** LoRA-ES overhead per iteration = parameter perturbation only
(O(lora_params), sub-second). Runtime dominated by same generation pass as NAMM-only.
**Joint training should not significantly increase per-iter time.**

For a 200-iter joint run: estimate **~3h** on 4070Ti. Feasible for daytime runs.
4 ablation conditions × 200 iters ≈ **12h total** — a single GPU-day.

---

## Why NOT These Alternatives

| Alternative | Why Not |
|-------------|---------|
| PEFT version upgrade | Breaking risk — Transformers 4.41.2 + PEFT 0.11.1 is tested; newer PEFT may conflict with `WrappedLlamaForCausalLM` |
| QLoRA (4-bit) | bitsandbytes quantization incompatible with per-perturbation re-quantization |
| DoRA | Larger overhead, no advantage for gradient-free optimization |
| LoRA on MLP layers | 3× more params, marginal gain for long-context retrieval; NAMM bottleneck is attention KV |
| Full weight perturbation | 1B params × 8 pop members = 32GB VRAM minimum |
| JAX/HyperscaleES | Framework incompatibility; RWKV-specific noiser, not portable |
| Full-covariance CMA-ES for Mode A | 65k+ dims → covariance matrix ≥17GB; existing code caps at 40k |

---

## PEFT Integration Points in Existing Codebase

| Location | What Changes |
|----------|-------------|
| `memory_llms/llama.py: WrappedLlamaForCausalLM.__init__` | Add `apply_lora_adapters(self, lora_config)` method |
| `memory_llms/base.py: MemoryModelWrapper` | Add `set_lora_params(flat_vec)` and `get_lora_params_flat()` |
| `memory_trainer.py: _save_ckpt` | Add `'lora_flat': model.get_lora_params_flat()` to ckpt dict |
| `memory_trainer.py: _load_ckpt` | Call `model.set_lora_params(ckpt['lora_flat'])` after loading |
| `memory_trainer.py: _train_step` | Mode B: `lora_es.ask()` → `model.set_lora_params()` before eval |
| New: `memory_evolution/lora_es.py` | `LoRA_ES(MemoryEvolution)` with OpenES antithetic |
| New: `cfgs/evolution/lora_es.yaml` | `sigma: 0.001`, `pop_size: ${pop_size}`, `antithetic: true` |

---

*Confidence: HIGH for PEFT/PyTorch API and compute estimates (measured). MEDIUM for sigma values (single reference, needs empirical tuning).*
