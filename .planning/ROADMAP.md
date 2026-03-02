# Roadmap: Joint NAMM + LoRA-ES Training

## Overview

Starting from a working LLaMA 3.2-1B NAMM CMA-ES trainer, this roadmap extends the system to jointly evolve LoRA adapter weights alongside the NAMM memory policy using Evolution Strategies. The work proceeds bottom-up: git branch first, then a correctness-gated LoRA seam, then ES algorithm implementations, then Mode B training, then diagnostic observability, then evaluation infrastructure, then the ablation and transferability experiments that constitute the scientific contribution. Every phase delivers a verifiable capability; no phase begins before its predecessor's unit tests pass.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Branch Setup** - Create the working branch and verify the baseline NAMM trainer runs on it
- [ ] **Phase 2: LoRA Seam + Correctness Gate** - Inject PEFT LoRA into LLaMA, add flat-vector extract/inject, extend checkpoints, and lock correctness with unit tests
- [ ] **Phase 3: OpenES Implementation** - Build the LoRA_ES class (OpenES with antithetic sampling) and Hydra variant selector
- [ ] **Phase 4: EggRoll Implementation** - Build the LoRA_EggRoll class (structured rank-r noise variant, PyTorch rewrite)
- [ ] **Phase 5: Mode B Training Loop** - Extend _train_step for joint NAMM+LoRA Mode B and lora_only ablation mode with run configs
- [ ] **Phase 6: Diagnostic Logging** - Add per-component wandb metrics for observability: param counts, L2 norms, fitness correlations, diversity, sigma
- [ ] **Phase 7: Evaluation Harness Extension** - Extend evaluator to load joint checkpoints, multi-task eval config, zero-shot baseline
- [ ] **Phase 8: Baseline Ablation Runs** - Execute and record NAMM-only, LoRA-only, and joint Mode B training runs on QASPER with both ES variants
- [ ] **Phase 9: Task Transferability Experiments** - Train on one task, evaluate on all three; collect transferability matrix
- [ ] **Phase 10: Analysis and Synthesis** - Interpret diagnostic logs, compare conditions, document findings, produce paper-ready result tables

## Phase Details

### Phase 1: Branch Setup
**Goal**: The working branch exists, the existing NAMM trainer runs cleanly on it, and there is a verified starting point for all subsequent work.
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01
**Success Criteria** (what must be TRUE):
  1. `git branch dev/joint-namm-lora-es` exists locally, forked from `dev/llama_1b_namm` at commit 21b2880
  2. The existing QASPER NAMM eval script runs to completion on the new branch and reproduces the known QASPER score (within 0.1 F1) without any code changes
  3. `git log --oneline` shows the new branch head is one commit ahead of `dev/llama_1b_namm` (branch-creation commit only, no stray changes)
**Plans**: 1 plan

Plans:
- [x] 01-01-PLAN.md — Verify branch state, run NAMM smoke eval on GPU, commit phase-1 marker

### Phase 2: LoRA Seam + Correctness Gate
**Goal**: LLaMA 3.2-1B accepts LoRA weights injected from a flat float32 vector, checkpoints save and restore LoRA state, and four critical correctness pitfalls (injection ordering, merge corruption, silent zero-injection, bfloat16 underflow) are caught by passing unit tests before any training attempt.
**Depends on**: Phase 1
**Requirements**: LORA-01, LORA-02, LORA-03, LORA-04
**Success Criteria** (what must be TRUE):
  1. Running the unit test suite produces zero failures; the test log shows the expected LoRA module count (`num_layers * len(target_modules)`) = 32 for the default q_proj+v_proj config (LLaMA 3.2-1B has 16 hidden layers, not 14)
  2. Saving a checkpoint then loading it into a fresh model instance yields bit-identical LoRA weights and restores the LoRA config (rank, target modules); a NAMM-only checkpoint loads without error under the same code (graceful fallback verified)
  3. Running a full population evaluation cycle (pop_size=8, batch_size=4) leaves base model weights bit-for-bit unchanged, confirmed by a checksum assertion in the test suite
  4. LoRA weights are stored in float32 in the checkpoint (not bfloat16), confirmed by inspecting `ckpt['lora_state_dict']` dtypes in the test log
**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md — Add apply_lora_adapters() to WrappedLlamaForCausalLM; add get_lora_params_flat() / set_lora_params() to MemoryModelWrapper (LORA-01, LORA-02) [2026-03-02]
- [ ] 02-02-PLAN.md — Extend _save_ckpt / _load_ckpt with LoRA state dict, config, joint_es_mode; graceful fallback for NAMM-only checkpoints (LORA-03)
- [ ] 02-03-PLAN.md — Write tests/test_lora_seam.py with 5 pytest tests covering all LORA-04 assertions; requires GPU on sideswipe/prowl (LORA-04)

### Phase 3: OpenES Implementation
**Goal**: The `LoRA_ES` class implements OpenES with antithetic sampling against the flat-vector seam, is configurable via Hydra, and the variant selector key (`lora_es_variant=openES`) routes to it correctly.
**Depends on**: Phase 2
**Requirements**: ES-01, ES-04
**Success Criteria** (what must be TRUE):
  1. Calling `lora_es.ask()` returns a population matrix of shape `[pop_size, lora_dim]` where adjacent pairs are antithetic (`row[2i] + row[2i+1] ≈ 2 * mean` within float32 tolerance)
  2. Calling `lora_es.tell(fitness)` updates the internal mean in the direction predicted by z-score-normalized fitness-weighted perturbations, verified by a unit test on a 2D quadratic fitness landscape where the mean moves toward the optimum after 10 steps
  3. `lora_es_variant: openES` in a Hydra config instantiates `LoRA_ES` (not `LoRA_EggRoll`); confirmed by checking the class name in the instantiated object's `__class__.__name__`
  4. The `cfgs/evolution/lora_es.yaml` config file exists and contains `sigma`, `pop_size`, and `lora_es_variant` keys with documented defaults
**Plans**: TBD

Plans:
- [ ] 03-01: Copy `utils/worker_extn.py` (seeded noise + ES update math) from `shr1ram/es-fine-tuning-paper` (warming-up branch) into `memory_evolution/`; adapt parameter access from vLLM to standard `model.named_parameters()`
- [ ] 03-02: Build `LoRA_ES` class in `memory_evolution/lora_es.py` wrapping the ported update math; add antithetic pairs on top (+σε / −σε per seed); expose `ask()`/`tell()` interface matching `MemoryEvolution`
- [ ] 03-03: Write `cfgs/evolution/lora_es.yaml` Hydra config
- [ ] 03-04: Implement `lora_es_variant` dispatch; write unit tests for antithetic pairing and tell() convergence

### Phase 4: EggRoll Implementation
**Goal**: The `LoRA_EggRoll` class implements structured rank-r noise perturbation (rewritten in PyTorch from the HyperscaleES eggroll concept), exposes the same `ask()`/`tell()` interface as `LoRA_ES`, and routes correctly from the variant selector.
**Depends on**: Phase 3
**Requirements**: ES-02
**Success Criteria** (what must be TRUE):
  1. `lora_eggroll.ask()` returns perturbations of the form `ΔW = A @ B.T` where A and B are seeded random matrices with shapes matching the LoRA rank; the flat perturbation vector is reproducible given the same seed
  2. `lora_es_variant: eggroll` in a Hydra config instantiates `LoRA_EggRoll`; swapping variant between `openES` and `eggroll` in the same training config produces different perturbation patterns but the same tell() interface
  3. A 200-step LoRA-only dry run (no NAMM, full cache, dummy fitness) completes without error using the eggroll variant, and the LoRA mean drifts non-zero from its zero init
**Plans**: TBD

Plans:
- [ ] 04-01: Study `shr1ram/HyperscaleES` (warming-up) `src/hyperscalees/noiser/eggroll.py` and `alteggroll.py` to understand the rank-r noise design; port to PyTorch in `memory_evolution/lora_eggroll.py` staying as close to his class structure and naming as the JAX→PyTorch translation allows (replace `jax.random` with seeded `torch.Generator`, `jax.tree.map` with parameter iteration, `optax` update with direct `param.data +=`)
- [ ] 04-02: Wire eggroll into the variant dispatch; write unit tests for structured perturbation shape correctness and interface compatibility

### Phase 5: Mode B Training Loop
**Goal**: The training loop supports `joint_es_mode=B` (NAMM CMA-ES + LoRA ES simultaneously on shared fitness) and `joint_es_mode=lora_only` (LoRA ES only, policy=none, full cache); run configs exist for both; a joint Mode B training run produces a checkpoint containing both NAMM and LoRA state.
**Depends on**: Phase 4
**Requirements**: ES-03, ES-05, INFRA-02, INFRA-03
**Success Criteria** (what must be TRUE):
  1. Launching `cfgs/run/joint_namm_lora_b.yaml` runs 10 training iterations without error, and the output checkpoint at iter 10 contains both `namm_state` and `lora_state_dict` keys; loading that checkpoint into an evaluator reproduces non-trivial (non-zero) scores on QASPER
  2. Launching `cfgs/run/lora_only.yaml` runs 10 training iterations without error; `namm_param_size` in the wandb config shows 0 or absent; the checkpoint contains `lora_state_dict` but no NAMM params
  3. In both modes, each population member receives a distinct combination of NAMM params (or none) and LoRA params per iteration, confirmed by asserting `lora_param_matrix[0] != lora_param_matrix[1]` in a debug log
  4. The `joint_es_mode` Hydra key is the only switch needed to move between `namm_only`, `lora_only`, and `B`; no other config keys require manual change
**Plans**: TBD

Plans:
- [ ] 05-01: Extend `_train_step` with Mode B branch: `cma_es.ask()` + `lora_es.ask()` before loop, `set_memory_params` + `set_lora_params` per member, `cma_es.tell()` + `lora_es.tell()` after
- [ ] 05-02: Add `lora_only` mode branch: skip CMA-ES, run `lora_es.ask()/tell()` only, policy=none
- [ ] 05-03: Write `cfgs/run/joint_namm_lora_b.yaml` and `cfgs/run/lora_only.yaml`
- [ ] 05-04: Integration test: 10-iter run in each mode; verify checkpoint structure and score

### Phase 6: Diagnostic Logging
**Goal**: Per-run and per-step wandb metrics give full observability into NAMM and LoRA evolution: parameter sizes, L2 norms, perturbation-fitness correlations, population diversity, and LoRA sigma — enabling mechanistic claims about whether the two components cooperate or interfere.
**Depends on**: Phase 5
**Requirements**: DIAG-01, DIAG-02, DIAG-03, DIAG-04, DIAG-05
**Success Criteria** (what must be TRUE):
  1. The wandb run page for a joint Mode B run shows `namm_param_size`, `lora_param_size`, `joint_param_size`, `lora_rank`, and `lora_target_modules` populated in the Config tab at run start
  2. The wandb Metrics tab shows `component_norm/namm_l2_mean` and `component_norm/lora_l2_mean` as time series over training steps, with non-constant values indicating active evolution in both components
  3. The wandb Metrics tab shows `signal/namm_fitness_correlation` and `signal/lora_fitness_correlation` as time series; in a LoRA-only run, `signal/namm_fitness_correlation` is absent or NaN (not a spurious zero)
  4. `pop/diversity_l2` and `evo_stats/sigma_lora` appear in wandb metrics every step during Mode B training, and `evo_stats/sigma_lora` is non-zero and non-constant
**Plans**: TBD

Plans:
- [ ] 06-01: Add param count logging to `wandb.config` at run start
- [ ] 06-02: Add per-step L2 norm logging for NAMM and LoRA population means
- [ ] 06-03: Add per-step Pearson r correlation logging (perturbation direction vs. fitness rank)
- [ ] 06-04: Add population diversity (mean pairwise L2) and sigma_lora logging
- [ ] 06-05: Verify all metrics appear correctly in wandb for Mode B and lora_only runs

### Phase 7: Evaluation Harness Extension
**Goal**: The evaluator loads joint NAMM+LoRA checkpoints correctly; a single config invocation evaluates all three tasks (QASPER, NarrativeQA, PassageRetrieval) from one checkpoint; and a zero-shot baseline (iter=0, no NAMM, full cache, no LoRA) is measured to establish the pre-adaptation reference.
**Depends on**: Phase 6
**Requirements**: EVAL-01, EVAL-02, EVAL-03
**Success Criteria** (what must be TRUE):
  1. Running `cfgs/run/joint_eval_all_tasks.yaml` against the Phase 5 joint Mode B checkpoint produces scores on all three tasks in a single script invocation without manually editing the config between tasks
  2. The eval log confirms LoRA weights are loaded and active during evaluation (LoRA norm non-zero in eval-time diagnostic output or wandb eval config)
  3. A zero-shot eval run (no checkpoint, `init_from=null`, `cache_size=4096`, no LoRA) completes and logs scores for all three tasks to the `Llama-3.2-1B/joint-es` wandb group under a `zero_shot_baseline` run name
  4. The evaluator handles a NAMM-only checkpoint (no `lora_state_dict` key) without error, falling back to full-cache evaluation — confirming backward compatibility
**Plans**: TBD

Plans:
- [ ] 07-01: Extend `MemoryHFEvaluator._load_checkpoint()` to restore LoRA state if present
- [ ] 07-02: Write `cfgs/run/joint_eval_all_tasks.yaml` multi-task eval config
- [ ] 07-03: Run and record zero-shot baseline (iter=0) on all three tasks
- [ ] 07-04: Verify backward compatibility with NAMM-only checkpoints

### Phase 8: Baseline Ablation Runs
**Goal**: Controlled training runs across all four conditions (NAMM-only, LoRA-only/openES, LoRA-only/eggroll, joint Mode B/openES, joint Mode B/eggroll) on QASPER produce valid checkpoints and comparable eval scores that form the core result table of the paper.
**Depends on**: Phase 7
**Requirements**: None (v1 reqs fully satisfied by Phase 7; this phase serves the PROJECT.md core value: "verified through controlled ablations")
**Success Criteria** (what must be TRUE):
  1. All five ablation conditions complete 200 training iterations on QASPER without crashing; wandb shows full training curves for all conditions in the `Llama-3.2-1B/joint-es` group
  2. Each completed run produces a valid checkpoint that loads cleanly into the evaluator and scores non-trivially on at least one task (QASPER score > 1.0 for all conditions)
  3. The eval table shows measurable differences between conditions (at least two conditions differ by > 0.5 QASPER F1), providing scientific signal for the ablation comparison
  4. NAMM-only condition (run from existing checkpoint, re-evaluated) scores within 0.2 of the previously recorded 6.46 QASPER F1, confirming reproducibility of the baseline
**Plans**: TBD

Plans:
- [ ] 08-01: Run NAMM-only (existing checkpoint re-eval) on all three tasks to confirm baseline
- [ ] 08-02: Run LoRA-only 200 iters with openES variant; eval on all three tasks
- [ ] 08-03: Run LoRA-only 200 iters with eggroll variant; eval on all three tasks
- [ ] 08-04: Run joint Mode B 200 iters with openES variant; eval on all three tasks
- [ ] 08-05: Run joint Mode B 200 iters with eggroll variant; eval on all three tasks
- [ ] 08-06: Collect all scores into a comparison table; log to wandb summary

### Phase 9: Task Transferability Experiments
**Goal**: Training on QASPER and evaluating on NarrativeQA and PassageRetrieval (and vice versa) reveals whether joint NAMM+LoRA representations transfer across tasks, completing the transferability matrix that is a stated goal in PROJECT.md.
**Depends on**: Phase 8
**Requirements**: None (extends Phase 8 checkpoints; serves PROJECT.md "Task transferability evaluation" goal)
**Success Criteria** (what must be TRUE):
  1. The QASPER-trained joint Mode B checkpoint scores non-trivially on NarrativeQA (> 1.0 F1) without any additional training, confirming cross-task transfer is at least partially present
  2. A NarrativeQA-trained joint Mode B run completes 200 iters and produces a checkpoint; evaluating it on QASPER gives a score > the zero-shot baseline, confirming bidirectional transfer possibility
  3. The full transferability matrix (2 training tasks × 3 eval tasks) is recorded in wandb and a local CSV, with clear labels for which checkpoint was used for each row
**Plans**: TBD

Plans:
- [ ] 09-01: Evaluate QASPER-trained checkpoints (NAMM-only, joint B) on all three tasks
- [ ] 09-02: Run joint Mode B training on NarrativeQA; eval on all three tasks
- [ ] 09-03: Compile transferability matrix; identify best-transferring condition

### Phase 10: Analysis and Synthesis
**Goal**: Diagnostic logs are interpreted, conditions are compared mechanistically (do NAMM and LoRA cooperate or interfere?), and findings are organized into a coherent narrative with paper-ready tables and figures.
**Depends on**: Phase 9
**Requirements**: None (synthesis phase; no new code requirements)
**Success Criteria** (what must be TRUE):
  1. The `signal/namm_fitness_correlation` and `signal/lora_fitness_correlation` time series from wandb are plotted side-by-side for joint Mode B runs; the plot shows at least one condition where both correlations are positive (indicating cooperation) or reveals interference patterns that explain performance differences
  2. A result table exists that lists all ablation conditions × all eval tasks with scores, standard deviations (if multi-seed), and the delta from the zero-shot baseline — formatted to drop into a paper or report
  3. Key findings are recorded as numbered conclusions in a FINDINGS.md file under `.planning/`, each citing the specific wandb metric or eval score that supports it
**Plans**: TBD

Plans:
- [ ] 10-01: Export and plot diagnostic time series from wandb for all training runs
- [ ] 10-02: Compile final result table (all conditions × all tasks × all metrics)
- [ ] 10-03: Write FINDINGS.md with numbered conclusions and supporting evidence

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Branch Setup | 1/1 | Complete | 2026-03-02 |
| 2. LoRA Seam + Correctness Gate | 2/3 | In Progress|  |
| 3. OpenES Implementation | 0/3 | Not started | - |
| 4. EggRoll Implementation | 0/2 | Not started | - |
| 5. Mode B Training Loop | 0/4 | Not started | - |
| 6. Diagnostic Logging | 0/5 | Not started | - |
| 7. Evaluation Harness Extension | 0/4 | Not started | - |
| 8. Baseline Ablation Runs | 0/6 | Not started | - |
| 9. Task Transferability Experiments | 0/3 | Not started | - |
| 10. Analysis and Synthesis | 0/3 | Not started | - |
