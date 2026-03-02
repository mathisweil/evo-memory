# Project Research Summary

**Project:** Joint NAMM + LoRA-ES Training
**Domain:** Gradient-free joint optimization of KV-cache memory policy (NAMM) and LoRA adapters on LLaMA 3.2-1B
**Researched:** 2026-03-02
**Confidence:** HIGH (stack/architecture from direct codebase inspection; MEDIUM on ES hyperparameters)

## Executive Summary

This project extends an existing, working NAMM CMA-ES training system to jointly co-evolve LoRA adapter weights alongside the NAMM memory policy, using Evolution Strategies on a single NVIDIA 4070Ti GPU. The research shows that **Mode B (decoupled ES) is the only viable implementation path from day one**: CMA-ES full-covariance cannot scale to LoRA parameter counts (r=4 alone is 327k dims, producing an ~340 GB covariance matrix), while a separate sigma-ES (OpenES with antithetic sampling) for LoRA adds negligible VRAM overhead (<2 MB above the existing 2.5 GB LLaMA base), and keeps the existing CMA-ES loop entirely unchanged. Per-iteration runtime stays at ~48s since LoRA perturbation is a CPU-side O(lora_params) operation; a full 200-iter joint run is estimated at ~3h, and all four ablation conditions fit in a single GPU-day.

The recommended implementation strategy is strictly bottom-up: add PEFT wrapping and flat-vector injection first, validate with unit tests before touching the training loop, then build the LoRA_ES class, then extend _train_step, then logging, then ablation configs. The most important correctness invariants are: (1) PEFT must be injected *after* WrappedLlamaForCausalLM construction, (2) merge_adapter() must never be called during population evaluation, and (3) LoRA weights must be stored in float32 to avoid bfloat16 underflow at low ranks. These three issues are silent — they produce no errors but corrupt training silently.

The primary scientific risk is not implementation but signal: with pop_size=8 and NAMM already converged on QASPER, LoRA perturbations may produce too weak an ES signal to escape their zero-init safe zone, since joint fitness cannot attribute credit separately. The mitigations are (a) freezing NAMM for the first N iters of joint training to let LoRA find its direction, (b) using multi-task fitness (QASPER + NarrativeQA + Countdown) to widen the signal landscape, and (c) logging per-component L2 norms and perturbation-fitness correlations as the primary diagnostic. Without these diagnostics the paper cannot make mechanistic claims about whether NAMM and LoRA cooperate or interfere.

---

## Key Findings

### Recommended Stack

The existing `th2` conda environment already contains every required library: PEFT v0.11.1 for LoRA, `torch.nn.utils.parameters_to_vector`/`vector_to_parameters` for flat-vector ES perturbation, and PyTorch bfloat16 generation. No new packages need to be installed. The only new code artifacts are: one new Python file (`memory_evolution/lora_es.py`), two new Hydra configs (`cfgs/evolution/lora_es.yaml`, `cfgs/run/joint_*.yaml`), and extensions to ~5 existing files totalling ~200 lines. Mode A (joint CMA-ES) is architecturally infeasible without a full diagonal-CMA research extension and should be treated as a deferred research task, not a Phase 1 goal.

**Core technologies:**
- **PEFT v0.11.1 (already installed):** Apply LoRA to LLaMA 3.2-1B — standard `LoraConfig` + `get_peft_model()`, applied post-wrapper-construction
- **`parameters_to_vector` / `vector_to_parameters` (PyTorch built-in):** Flatten LoRA params to 1D for ES perturbation and re-inject per population member
- **OpenES with antithetic sampling (new `LoRA_ES` class):** Sigma-perturbation ES for LoRA weights; antithetic pairs are more sample-efficient than one-sided at small pop sizes
- **Existing CMA-ES (unchanged in Mode B):** Continues to evolve NAMM params independently on same fitness signal

**Critical parameter recommendation:** r=4, target_modules=["q_proj", "v_proj"], sigma_lora=0.001 (same order as NAMM per-param scale ~0.004). Set lora_dtype=float32 explicitly.

See `.planning/research/STACK.md` for parameter counts, compute timing, and full API patterns.

### Expected Features

The feature set divides cleanly into a correctness-blocked MVP (Phase 1), an ES-training core (Phase 2), and scientific evaluation infrastructure (Phase 3).

**Must have — Phase 1 (without these nothing works):**
- PEFT LoRA wrapping of LLaMA 3.2-1B with explicit target_modules — prerequisite for everything
- LoRA flat-vector extraction + injection methods in `MemoryModelWrapper` — the seam all ES code calls
- Extended checkpoint saving/loading LoRA state dict — correctness blocker; without this eval after training is impossible
- Unit tests: PEFT injection confirmed, base weights stable across pop eval, param counts match expectations
- bfloat16 guard: `lora_dtype=torch.float32` — silent corruption without this
- VRAM profile before training to confirm <2 MB overhead holds

**Must have — Phase 2 (core ES training):**
- `LoRA_ES` class with antithetic OpenES ask()/tell() interface
- Extended `_train_step` for Mode B (two ES objects, shared fitness)
- Per-component diagnostic logging: NAMM L2 norm, LoRA L2 norm, perturbation-fitness correlations per component
- `lora_only` ablation mode (key comparison baseline, low complexity)
- `joint_es_mode` Hydra selector (A/B/namm_only/lora_only)

**Should have — Phase 2 (differentiators):**
- Adaptive sigma scheduling for LoRA_ES (warmup + decay)
- Population diversity metric to detect premature convergence
- LoRA weight norm per-layer heatmap (which layers adapt most)
- NAMM eviction pattern comparison with vs without LoRA

**Defer to Phase 3+:**
- Mode A (diagonal CMA-ES over joint vector) — needs new research into diagonal-CMA implementation
- Countdown task integration — custom data pipeline required
- Task transferability result matrix — depends on all training conditions completing
- Multi-seed runs and secondary ablations

**Explicit anti-features (do not implement):**
- Gradient-based fine-tuning (different paper)
- vLLM, QLoRA, DoRA, full-weight perturbation, JAX patterns
- Multi-GPU DDP, models larger than 1B

See `.planning/research/FEATURES.md` for full configuration knob table and complexity estimates.

### Architecture Approach

The architecture change is deliberately minimal: Mode B keeps the existing CMA-ES loop entirely unchanged and adds a parallel LoRA_ES object that receives the same scalar fitness signal and updates its own mean and sigma independently. The training loop gains two lines per member evaluation (set_memory_params already present; add set_lora_params), one ask() call before the loop, and one tell() call after. All LoRA population weights live on CPU and are moved to GPU one member at a time, exactly mirroring the existing NAMM pattern.

**Major components:**
1. **`LoRA_ES` (new: `memory_evolution/lora_es.py`)** — OpenES antithetic; ask()/tell() interface matching `MemoryEvolution`; stores mean and sigma on CPU
2. **`WrappedLlamaForCausalLM.apply_lora_adapters()` (new method in `memory_llms/llama.py`)** — called once at trainer init; injects PEFT post-construction
3. **`MemoryModelWrapper.get/set_lora_params_flat()` (new methods in `memory_llms/base.py`)** — flat-vector seam between LoRA_ES and PEFT parameter tensors
4. **Extended `memory_trainer.py` (~100 lines)** — Mode B branch in `_train_step`; LoRA state in `_save_ckpt`/`_load_ckpt`
5. **`main.py` conditional init (~30 lines)** — instantiate LoRA_ES when `joint_es_mode != namm_only`; apply PEFT after model construction

**Data flow per ES iteration:**
```
CMA-ES.ask()    → namm_param_matrix [pop_size, namm_dim]   (CPU)
LoRA_ES.ask()   → lora_param_matrix [pop_size, lora_dim]   (CPU)
for member i:
  model.set_memory_params(namm_param_matrix[i])
  model.set_lora_params(lora_param_matrix[i].to(device))
  fitness[i] = evaluator.evaluate(model, task_samples)
CMA-ES.tell(fitness)   → update NAMM mean + covariance
LoRA_ES.tell(fitness)  → update LoRA mean (weighted avg of perturbations)
```

**Suggested build order:** PEFT injection + extraction methods → unit tests → extended checkpoint → LoRA_ES class → extend _train_step → lora_only ablation → joint Mode B → diagnostic logging → Mode A (much later).

See `.planning/research/ARCHITECTURE.md` for complete code stubs and component boundary table.

### Critical Pitfalls

1. **CMA-ES covariance explosion in Mode A** — even r=1 (65k dims) exceeds CMA-ES feasibility; add hard assertion `assert joint_param_size < 5000` before any Mode A instantiation. **Mode B is required for all real LoRA configs.** (Phase 1)

2. **PEFT injection ordering corrupts model loading** — apply PEFT *after* `WrappedLlamaForCausalLM` construction; assert `any('lora_A' in n ...)` immediately after; silent failure produces LoRA norm=0 and training identical to NAMM-only. (Phase 1)

3. **PEFT merge_adapter() corrupts base weights** — never call merge during population evaluation; use `disable_adapter_layers()`/`enable_adapter_layers()`; assert base weight bit-identity before and after a full pop eval. (Phase 1)

4. **PEFT silent zero-injection on custom model class** — always specify `target_modules=["q_proj","v_proj"]` explicitly; assert `lora_count == num_layers * len(target_modules)` after injection. (Phase 1)

5. **LoRA ES signal too weak at pop_size=8** — NAMM already converged; LoRA perturbations have small marginal fitness effect; mitigate by freezing NAMM for first N iters, using multi-task fitness, and logging perturbation-fitness correlation per component. Budget 500+ iters for LoRA-only runs. (Phase 2)

6. **bfloat16 LoRA underflow at low rank** — set `lora_dtype=torch.float32` in LoraConfig; silent zero-rounding in 7-bit mantissa destroys LoRA expressivity at r=1-2. (Phase 1)

See `.planning/research/PITFALLS.md` for all 13 pitfalls with detection signals and phase assignments.

---

## Implications for Roadmap

Based on research, the work divides into four natural phases driven by hard dependencies and risk ordering.

### Phase 1: LoRA Seam + Correctness Infrastructure

**Rationale:** Every subsequent phase depends on PEFT injection working correctly, flat-vector extract/inject being correct, and checkpoints saving LoRA state. Three critical pitfalls (ordering, merge corruption, silent zero-injection, bfloat16) are silent-failure traps that must be caught by unit tests before any training run.

**Delivers:** A model that can accept LoRA weights from an ES vector; confirmed-correct checkpoint format; unit test suite that acts as a regression gate for all future work.

**Addresses:** LoRA PEFT wrapping, flat-vector extraction/injection, extended checkpoint (table stakes features 1, 2, 6 from FEATURES.md).

**Avoids Pitfalls:** 2 (injection ordering), 5 (merge corruption), 7 (silent zero-injection), 12 (bfloat16), 8 (VRAM profile).

**Research flag:** Standard patterns — PEFT API is well-documented; no additional research needed beyond STACK.md.

---

### Phase 2: LoRA_ES Class + Mode B Training

**Rationale:** With the seam established and tested, LoRA_ES can be built against the flat-vector interface and plugged into the existing training loop with minimal changes to CMA-ES code. Mode B (separate ES objects) must come before Mode A because it avoids all covariance-dimension pitfalls and validates the core hypothesis with zero CMA-ES risk.

**Delivers:** A working joint Mode B training run producing a checkpoint with both NAMM and LoRA state; lora_only ablation run; per-component diagnostic logs in wandb.

**Addresses:** LoRA_ES class (Mode B), extended _train_step, lora_only ablation config, parameter count logging, component norm logging (table stakes features 2, 3, 4, 5, 7 from FEATURES.md).

**Avoids Pitfalls:** 4 (credit attribution — diagnostic logging required before claims), 9 (signal-to-noise — freeze NAMM strategy + multi-task fitness consideration), 13 (Hydra param_size conflict in Mode B: unchanged).

**Research flag:** Needs deeper research on OpenES antithetic update rule variants and sigma scheduling. The existing es-fine-tuning-paper reference (sigma=0.001, pop=30) is a single data point; sigma and pop_size tuning will require empirical iteration beyond budget estimates.

---

### Phase 3: Ablation Suite + Task Transferability

**Rationale:** Once Mode B trains successfully, the paper requires: (a) NAMM-only baseline, (b) LoRA-only, (c) joint Mode B — all on the same tasks. Transferability (train on QASPER, eval on NarrativeQA/Countdown) requires all training conditions to have valid checkpoints. Countdown integration is a separate sub-task with its own pipeline risk.

**Delivers:** Complete ablation table across 3 tasks × 4 conditions; transferability result matrix; paper-ready numbers.

**Addresses:** Ablation modes (namm_only, lora_only, joint B), multi-task eval configs, Countdown task integration, zero-shot baseline at iter=0, transferability infrastructure (FEATURES.md differentiators and Phase 2+ features).

**Avoids Pitfalls:** 10 (QASPER training bias — multi-task fitness from start, 3+ seeds, always run NAMM-only baseline on all tasks first), 11 (Countdown pipeline — implement and test independently before adding to ES loop).

**Research flag:** Countdown task integration needs dedicated research — it is not in LongBench, requires custom data pipeline and metric. Flag for `/gsd:research-phase` during planning.

---

### Phase 4: Mode A (Diagonal CMA-ES) — Research Extension

**Rationale:** Mode A (joint vector over NAMM + LoRA params via a single ES) is an interesting research question but requires a non-trivial algorithmic change (diagonal/separable CMA-ES variant replacing full-covariance CMA-ES). It should only be attempted after Mode B produces clean baseline results. If Mode B already shows cooperative improvement, Mode A may not be necessary for the paper.

**Delivers:** Joint single-optimizer baseline for comparison; evidence on whether covariance coupling across components helps or hurts.

**Addresses:** Mode A feature (deferred from FEATURES.md), diagonal CMA-ES implementation.

**Avoids Pitfalls:** 1 (covariance OOM — requires diagonal variant, NOT the existing `cma_es.py`), 3 (mixed-scale parameter vector — requires uniform init of LoRA A/B to NAMM scale), 6 (checkpoint migration — Mode A changes CMA-ES param_size, needs partial load).

**Research flag:** Definitely needs `/gsd:research-phase` — diagonal CMA-ES (xNES or sep-CMA-ES) is an active research area with no existing implementation in this codebase. STACK.md explicitly flags this as an "additional research task".

---

### Phase Ordering Rationale

- **Phases 1 before 2:** The flat-vector seam is a hard technical dependency; ES training cannot proceed without it. Unit tests are not optional — three of the critical pitfalls are silent failures that produce plausibly-looking but incorrect results.
- **Mode B before Mode A:** Mode A requires algorithmic research that Mode B does not. Mode B produces valuable baselines and de-risks the joint training concept before the harder optimization problem.
- **Phase 3 after Phase 2:** Transferability experiments require multiple trained checkpoints from different conditions. Countdown requires independent pipeline work that can start in parallel but integrates in Phase 3.
- **Phase 4 last (optional):** Mode A adds complexity with uncertain scientific payoff. It's a research extension, not a requirement for the paper.

### Research Flags

Phases likely needing deeper `/gsd:research-phase` during planning:
- **Phase 2:** OpenES antithetic update rule, sigma scheduling strategy, signal-to-noise implications of pop_size=8 for LoRA. The single reference (es-fine-tuning-paper, sigma=0.001, pop=30) needs corroboration or empirical calibration.
- **Phase 3:** Countdown task data pipeline and evaluation metric. Not in LongBench; requires custom implementation and independent validation before ES integration.
- **Phase 4:** Diagonal/separable CMA-ES algorithm selection and implementation. xNES, sep-CMA-ES, or PGPE are candidates — no existing code in repo, active research area.

Phases with standard patterns (skip research-phase):
- **Phase 1:** PEFT API is thoroughly documented (v0.11.1 docs, direct codebase analysis confirms all target module paths). PyTorch flat-vector utilities are stable. No research needed.
- **Phase 3 (eval infrastructure):** Multi-task eval and transferability matrix are straightforward extensions of existing `MemoryHFEvaluator` patterns. Standard practice.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Direct codebase inspection; PEFT v0.11.1 confirmed installed in `th2`; PyTorch API is stable; compute timing from measured wandb runs |
| Features | HIGH | Primary sources: codebase direct inspection, PROJECT.md answers; feature boundaries are clear with explicit anti-feature reasoning |
| Architecture | HIGH | Based on direct codebase analysis of all modified files; code stubs provided for all integration points; data flow is an extension of existing patterns |
| Pitfalls | MEDIUM | Critical pitfalls 1-7 are high-confidence from codebase analysis and PEFT docs; ES signal quality (pitfall 9) and transferability confounds (pitfall 10) are based on ES literature + inference, not empirical data |

**Overall confidence:** HIGH for implementation decisions; MEDIUM for ES hyperparameter choices and scientific outcomes.

### Gaps to Address

- **LoRA sigma calibration:** The sigma_lora=0.001 recommendation comes from a single reference (shr1ram/es-fine-tuning-paper) that used pop=30, not pop=8. At pop=8, sigma may need to be an order of magnitude larger (0.01-0.1) to produce meaningful signal. Plan for sigma search in Phase 2 hyperparameter tuning.

- **NAMM freeze strategy for credit attribution:** PITFALLS.md recommends freezing NAMM for first N iters to let LoRA find its direction, but the optimal N is unknown. This needs an empirical pilot run (50-iter LoRA-only from NAMM-pretrained weights) before committing to a schedule.

- **Countdown task data format:** No concrete information on Countdown dataset location, format, or existing eval infrastructure in the codebase. This must be investigated before Phase 3 planning. If it requires a LongBench-incompatible format, the pipeline effort is non-trivial.

- **Mode A diagonal CMA-ES algorithm:** The choice between sep-CMA-ES, xNES, and PGPE is unresolved. All are viable but have different convergence properties and implementation complexity. Defer until Phase 3 results justify the investment.

- **Multi-task fitness weighting (α·qasper + β·narrativeqa + γ·countdown):** Optimal weights are unknown and task scores are on different scales. Normalization strategy (z-score vs. min-max) needs a design decision before Phase 3 training begins.

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection (`memory_trainer.py`, `memory_evolution/cma_es.py`, `memory_llms/llama.py`, `memory_llms/base.py`) — architecture, pitfall detection, parameter counts
- PEFT v0.11.1 documentation — LoraConfig API, target_modules behavior, merge_adapter() semantics
- Measured wandb run-20260225_183712 — compute timing (48s/iter post-optimisation)
- `th2` conda environment inspection — installed packages confirmed

### Secondary (MEDIUM confidence)
- `shr1ram/es-fine-tuning-paper` (warming-up branch) — sigma=0.001 recommendation; caveats: pop=30 vs our pop=8, full-weight not LoRA
- ES literature (OpenES antithetic sampling, NES weight perturbation) — signal-to-noise analysis at small pop_size
- PEFT behavior on custom model classes — inference from docs + known limitation patterns

### Tertiary (LOW confidence)
- Countdown task format assumptions — not verified; treat as unknown until investigated
- Mode A diagonal CMA-ES viability — theoretical analysis only; no implementation tested

---
*Research completed: 2026-03-02*
*Ready for roadmap: yes*
