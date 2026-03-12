# TPU Compatibility Plan

This plan is scoped to making the `es-fine-tuning` codebase runnable and stable on TPU, then maps what was already implemented on `tpu`.

## 1) Clear Plan: `es-fine-tuning` -> TPU-Compatible

### Goals
- Run `scripts/run_es.py` end-to-end on TPU with `es_only`, `es_namm`, and `es_recency`.
- Run `scripts/run_eval.py` on TPU for checkpoints produced by TPU training.
- Keep GPU behavior unchanged.
- Make spot TPU runs resumable and operationally robust.

### Non-goals (initial milestone)
- Bit-for-bit numerical parity with GPU.
- Immediate multi-chip parallel ES execution (can be phase 2 optimization).

### Acceptance Criteria
- TPU smoke test (`iter=2`, `pop=2`) completes for all 3 methods.
- 50-iteration run completes on TPU with no shape-related crash.
- Resume from preemption works (checkpoint + RNG state restoration).
- Full eval runs and writes `results.json`.

### Phase A: Device and Runtime Abstraction
1. Remove hardcoded CUDA assumptions in entry points and helpers.
2. Standardize device selection (`TPU > CUDA > CPU`) and sync behavior.
3. Guard CUDA-only performance settings.

Deliverable:
- Device-agnostic training and eval scripts.

### Phase B: XLA Static-Shape Hardening
1. Eliminate data-dependent shape changes in cache eviction paths.
2. Reduce compile-shape explosion from variable sequence lengths.
3. Handle TPU-incompatible indexing/slicing corner cases.

Deliverable:
- Stable set of XLA graphs for fixed config; no recurrent recompilation crashes.

### Phase C: TPU Operations and Reliability
1. TPU setup + activation scripts.
2. Spot preemption handling.
3. Checkpoint + training-state resume.
4. Optional cloud artifact/manifest synchronization.

Deliverable:
- Recoverable long-running TPU jobs.

### Phase D: Validation and Regression Safety
1. Smoke matrix:
   - `es_only`, `es_namm`, `es_recency`
   - train + eval
2. Consistency checks:
   - checkpoint load
   - resumed run equivalence (history continuation)
3. GPU regression check after TPU changes.

Deliverable:
- Repeatable validation checklist with pass/fail outputs.

### Phase E: Performance Optimization (after correctness)
1. Reduce host/device overhead in ES noise path.
2. Consider population parallelism across TPU chips.
3. Optional warmup workflow for compile-heavy configs.

Deliverable:
- Better TPU throughput and reduced wall-clock cost.


## 2) Detailed Changes Already Made on `tpu` (vs `es-fine-tuning`)

This section summarizes implemented changes and where they landed.

### A. Device-Agnostic TPU Enablement
- Added explicit device helper:
  - `es_finetuning/device.py`
  - `get_device()`, `sync_device()`, `empty_cache()`
- Updated scripts to use `.to(device)` instead of `.cuda()`:
  - `scripts/run_es.py`
  - `scripts/run_eval.py`
- Updated evaluator/device behavior:
  - `namm/evaluator.py` (`device='auto'`, lazy `device` property)
- Updated utility movement paths to avoid CUDA-only assumptions:
  - `utils/helpers.py`

### B. ES Runtime Changes for TPU
- Vendored ES package into repo (from external dependency):
  - `es_finetuning/{config.py,noise.py,trainer.py,utils.py,...}`
- TPU-safe ES noise generation:
  - `es_finetuning/noise.py` now generates deterministic noise on CPU and copies to device (XLA generator limitation workaround).

### C. XLA Compilation / Shape Control
- Sequence-length bucketing in tokenizer batch encode:
  - `namm/evaluator.py`
  - bucket set: `[512..131072]`
- Fixed-size eviction adaptation for NAMM on TPU:
  - `namm/policy/base_dynamic.py`
  - `namm/policy/deep.py`
  - `namm/policy/deep_selection.py`
  - `namm/llms/llama.py`
- Introduced `cache_validity_mask` flow:
  - evicted entries zeroed and masked in scoring/attention paths.

### D. TPU Setup / Activation Tooling
- Added TPU setup and activation scripts:
  - `setup/setup_tpu.sh`
  - `setup/setup_tpu_cmd.sh`
  - `setup/activate_tpu.sh`
  - `setup/tpu_restart.sh`
- Added TPU/GCS env defaults in activation path.

### E. Spot VM Resilience + Cloud Workflow
- Added preemption handler:
  - `es_finetuning/preemption.py` (SIGTERM flag)
- Added GCS client and manifest/concurrency helpers:
  - `es_finetuning/gcs.py`
- Added periodic + emergency checkpoint logic and exact resume:
  - `es_finetuning/trainer.py`
- Added run claiming, auto-resume, GCS-backed experiment flow:
  - `scripts/run_es.py`
- Added report/archive scripts for GCS-native workflows:
  - `scripts/generate_report.py`
  - `scripts/archive_experiment.py`

### F. Documentation and Config Updates
- Added TPU notes and updated docs:
  - `docs/tpu_notes.md`
  - `README.md`, `docs/es-ft-guide.md`, `docs/es-ft-namm-guide.md`, `docs/examples.md`
- Added/updated run configs for recency and full-cache ES:
  - `cfgs/run/recency_es_llama32_1b.yaml`
  - `cfgs/run/full_cache_es_llama32_1b.yaml`


## 3) Actionable Plan to Finish TPU Compatibility

The branch already contains most of the foundation. The remaining work is a stabilization pass to close correctness and operability gaps.

### Priority 0 (must fix before long TPU runs)

1. Enforce TPU-safe batching invariants in code (not only docs).
- Add runtime checks in `scripts/run_es.py` for TPU mode:
  - `batch_size` must be fixed integer.
  - `mini_batch_size == batch_size`.
  - sampled set size divisible by batch size, or explicit padding behavior.
- Fail fast with clear error messages.

2. Close documented-vs-code mismatch for recency/negative indexing fixes.
- Validate and patch:
  - `namm/policy/base.py` recency slicing paths.
  - `namm/llms/llama.py` `attention_mask` truncation math.
  - `namm/policy/deep_selection.py` mask slicing.
- Goal: no TPU crash when `cache_size > current_kv_len` and no variable-slice surprises.

3. Fix CLI contract for GCS toggle.
- `scripts/run_es.py` currently uses `--gcs` with `default=True` (always on).
- Add explicit `--no-gcs` behavior and keep docs/examples consistent.

4. Resolve missing warmup artifact mismatch.
- Either add `scripts/warmup_xla_cache.sh` (as documented) or remove all references.
- Prefer adding script so expected workflow remains available.

### Priority 1 (should do next)

5. Reassess XLA persistent cache sync behavior.
- `docs/tpu_notes.md` states deserialize is unsupported with current stack.
- If true for active runtime, make cache sync optional/off by default to reduce confusion and unnecessary ops.

6. Add TPU smoke CI-style script/checklist.
- Single command validating:
  - env activation
  - model load
  - 1-2 iteration ES run
  - eval pass
  - checkpoint save/load

7. Add strict TPU run presets in config.
- Provide known-good TPU configs (fixed batch, fixed cache sizes) to avoid accidental auto modes.

### Priority 2 (performance and scaling)

8. Reduce noise path overhead in ES.
- Current approach creates/copies per-parameter tensors from CPU each perturb/restore/update.
- Investigate batched precomputed noise buffers per seed or per-layer grouping.

9. Multi-chip population parallelism.
- Parallelize population member evaluation across TPU chips (e.g., `xmp.spawn` model replica workers + reward gather).
- Keep single-chip path as default fallback.


## 4) Execution Checklist

### Milestone M1: Correctness
- [ ] TPU batch invariant checks implemented.
- [ ] Negative indexing/recency fixes implemented and tested.
- [ ] `--no-gcs` added and documented.
- [ ] Warmup script mismatch resolved.
- [ ] `es_only` TPU smoke test passes.
- [ ] `es_namm` TPU smoke test passes.
- [ ] `es_recency` TPU smoke test passes.

### Milestone M2: Reliability
- [ ] Forced SIGTERM test confirms emergency checkpoint path.
- [ ] Resume reproduces continuous history from last checkpoint.
- [ ] Eval on resumed checkpoint succeeds.

### Milestone M3: Performance
- [ ] Compile-time overhead measured before/after warmup strategy.
- [ ] ES step-time profiling completed.
- [ ] Optional multi-chip prototype benchmarked.


## 5) Recommended Immediate Next Steps

1. Implement Priority 0 items in one patch set.
2. Run a deterministic TPU smoke matrix (`iter=2`) and record failures.
3. Start first 50-iteration TPU run only after M1 passes.

