# Feature Landscape

**Domain:** Joint Evolution Strategy training of KV-cache memory policy (NAMM) + LoRA adapters on LLaMA 3.2-1B
**Researched:** 2026-03-02
**Confidence:** HIGH (primary sources: codebase direct inspection, PROJECT.md)

---

## Existing System Baseline

Already implemented in `dev/llama_1b_namm`. New work extends these — does not duplicate.

| Existing Feature | Location |
|-----------------|----------|
| CMA-ES training loop (ask/tell) | `memory_evolution/cma_es.py`, `memory_trainer.py` |
| QASPER / NarrativeQA / PassageRetrieval eval | `task_sampler.py`, `memory_evaluator.py` |
| Hydra config composition | `cfgs/` hierarchy |
| wandb logging (fitness, evo_stats, param_stats) | `memory_trainer.py:945,1237` |
| Checkpoint save/load with RNG state | `memory_trainer.py:_save_ckpt/_load_ckpt` |
| Full-cache and recency baselines | `cfgs/run/` |
| PEFT v0.11.1 installed | `th2` conda env |

---

## Table Stakes

### 1. LoRA Integration

| Feature | Complexity | Notes |
|---------|------------|-------|
| PEFT LoRA wrapping of LLaMA 3.2-1B | Medium | `peft.get_peft_model(model, LoraConfig(...))` in `memory_llms/llama.py` |
| Configurable LoRA rank (r) | Low | Hydra key: `lora_rank` |
| Configurable LoRA target modules | Low | Hydra key: `lora_target_modules` |
| LoRA alpha scaling | Low | Convention: `lora_alpha = 2 * lora_rank` |
| LoRA weight extraction as flat vector | Medium | Filter PEFT named_parameters by `lora_` prefix |
| LoRA weight injection from flat vector | Medium | Inverse; must handle pop_size batching |

### 2. Joint ES Training Loop

| Feature | Complexity | Notes |
|---------|------------|-------|
| **Mode B**: CMA-ES (NAMM) + sigma-ES (LoRA) | High | Two separate ES objects, shared fitness signal, antithetic perturbations |
| **Mode A**: single CMA-ES over [namm_params \|\| lora_flat] | High | Feasible ONLY at r≤2 on q_proj only (~16k total dims); existing CMA caps at 40k |
| Configurable mode selector | Low | Hydra key: `joint_es_mode` ∈ {A, B, namm_only, lora_only} |
| Parameter count logging at startup | Low | Print + wandb.config: `namm_param_size`, `lora_param_size`, `joint_param_size` |

### 3. Diagnostic Logging — "Are Both Components Contributing?"

**Highest-priority new feature.** Without it the paper cannot make mechanistic claims.

| Metric | What It Reveals | Wandb Key |
|--------|----------------|-----------|
| NAMM param L2 norm (mean over pop) | Whether NAMM weights move | `component_norm/namm_l2_mean` |
| LoRA param L2 norm (mean over pop) | Whether LoRA weights move | `component_norm/lora_l2_mean` |
| NAMM perturbation ↔ fitness correlation | Is NAMM driving improvement? | `signal/namm_fitness_correlation` |
| LoRA perturbation ↔ fitness correlation | Is LoRA driving improvement? | `signal/lora_fitness_correlation` |
| Normalized perturbation magnitude | Is one component dominating? | `signal/namm_rel_perturbation`, `signal/lora_rel_perturbation` |
| sigma_lora (Mode B) | LoRA ES step size track | `evo_stats/sigma_lora` |
| Cache utilization rate | Is NAMM still evicting tokens? | `memory/cache_utilization` |

### 4. Ablation Modes (Required for credible paper)

| Condition | What It Tests | Config |
|-----------|---------------|--------|
| NAMM-only | Baseline (already done) | Existing 200-iter runs |
| LoRA-ES-only | Does LoRA alone improve? | `joint_es_mode=lora_only`, `policy=none`, `cache_size=4096` |
| Joint Mode A | Does joint CMA-ES outperform either alone? | `joint_es_mode=A`, `lora_rank=2` |
| Joint Mode B | Does decoupled ES scale to larger LoRA? | `joint_es_mode=B`, `lora_rank=8` |

### 5. Task Transferability Infrastructure

| Feature | Complexity | Notes |
|---------|------------|-------|
| Multi-task eval from single checkpoint | Low | Already possible via `eval_only=true` + `init_from`; needs multi-task config |
| Transferability result matrix | Low | Grid: rows=train-task, cols=eval-task |
| Countdown task integration | Medium | New `cfgs/task/countdown.yaml` + metric |
| Zero-shot baseline at iter=0 | Low | `eval_only=true`, `init_from=None`, `policy=none` |

### 6. Extended Checkpointing (Correctness Blocker)

| Feature | Complexity | Notes |
|---------|------------|-------|
| LoRA state dict in checkpoint | Low | Extend `_save_ckpt`; without this eval after training is impossible |
| NAMM + LoRA param counts in checkpoint | Low | Detect config mismatch before long runs |
| `joint_es_mode` in checkpoint | Low | Prevent loading Mode B ckpt with Mode A config |

---

## Differentiators

| Feature | Value | Complexity |
|---------|-------|------------|
| Per-component fitness attribution (ablate one component, measure fitness drop) | Answers "cooperate or interfere?" | High |
| LoRA weight norm per layer heatmap | Which layers adapt most | Low |
| NAMM eviction pattern with vs without LoRA | Mechanistic insight | Medium |
| Adaptive sigma scheduling (warmup + decay) for Mode B | Broader exploration early | Medium |
| Population diversity metric (`pop/diversity_l2`) | Detect premature convergence | Medium |

---

## Anti-Features

| Anti-Feature | Why Avoid |
|--------------|-----------|
| Gradient-based fine-tuning (Adam/SGD) | Different paper |
| vLLM acceleration | Incompatible with custom attention code |
| JAX/HyperscaleES patterns | RWKV-specific, different framework |
| Automated HPO | Runs take hours; manual grid suffices |
| Multi-model experiments (>1B) | Breaks comparability with NAMM baselines |
| Full-weight perturbation | 1B params × 8 pop = 32GB VRAM minimum |
| QLoRA / DoRA | Incompatible with ES perturbation patterns |
| Task-specific LoRA heads | Prevents transferability evaluation |
| Multi-GPU DDP for LoRA ES | Single GPU sufficient; DDP already complex |

---

## MVP Feature Set (Phase 1 Target)

1. LoRA PEFT wrapping (configurable rank + targets)
2. LoRA flat-vector extraction + injection (everything depends on this)
3. Mode B (separate ES objects — no CMA-ES dim pressure)
4. Parameter count logging at startup (sanity gate)
5. NAMM L2 + LoRA L2 norm per step (minimum diagnostic)
6. Extended checkpoint saving LoRA state dict (correctness blocker)
7. LoRA-only ablation config (key comparison, low extra complexity)

**Defer to Phase 2+:** Mode A, fitness correlation signal metrics, Countdown task, secondary ablations, transferability matrix, population diversity metric.

---

## Configuration Knobs That Matter

| Knob | Values to Test | Hydra Key |
|------|---------------|-----------|
| `lora_rank` | 2, 4, 8 | `lora_rank` |
| `lora_target_modules` | `["q_proj"]`, `["q_proj","v_proj"]` | `lora_target_modules` |
| `joint_es_mode` | A, B, namm_only, lora_only | `joint_es_mode` |
| `sigma_lora` (Mode B) | 0.001, 0.01, 0.05 | `sigma_lora` |
| `cache_size` | 128, 256, 512, 1024 | `cache_size` |

**Leave fixed:** `pretrained_llm_name`, `dtype=bfloat16`, `lora_dropout=0.0`, `max_iters=200`, `pop_size=8`
