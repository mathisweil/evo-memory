# Claude Code Task: Add Threshold-Based Eviction Mode to NAMM Training

## Context

This codebase trains a Neural Attention Memory Model (NAMM) via CMA-ES evolution using `scripts/run_namm.py` with Hydra configuration. The NAMM scores each KV-cache token with a selection score `s_i = m_φ(ω_i^{1:T})` and currently evicts based on a **hard top-k cache size cap** (keeping the `cache_size` highest-scoring tokens). 

The NAMM paper (Cetin et al., ICLR 2025) describes the **original eviction rule as purely threshold-based**: evict all tokens with `s_i < 0`, treating the decision as binary classification with no hard cap. The existing `check_eviction_stats.py` diagnostic already implements this by setting `memory_policy.selection_criteria.cache_size = None` after model construction (when `--cache_size 0` is passed).

The goal is to make `run_namm.py` (and anything it depends on) support **both modes** cleanly, selectable via a Hydra config flag.

---

## Step 1 — Explore and Plan (do this before making any changes)

Before writing a single line of code, do the following:

1. **Read `scripts/run_namm.py`** in full.
2. **Read `scripts/check_eviction_stats.py`** in full — this is the reference implementation for threshold-only mode.
3. **Find `namm/run_utils.py`** and read `make_eval_model()` — understand how `memory_policy`, `memory_evaluator`, and related objects are constructed from the Hydra config.
4. **Find the `selection_criteria` class** (search for `SelectionCriteria`, `cache_size`, `topk` across `namm/`). Understand what `cache_size = None` actually does at inference time — does it skip the topk step entirely, or is there a separate threshold gate?
5. **Find all Hydra config files** under `config/` that set `cache_size`, `max_memory_length`, or anything on `selection_criteria`. List every place this propagates.
6. **Find the `MemoryTrainer`** (in `trainer.py`) and check if it logs or checkpoints any cache-size-related stats that would need updating.
7. **Search for any fitness/reward shaping** tied to memory size (e.g., a penalty for large cache) — threshold mode produces variable cache sizes per step, so any fixed-size reward term would need to be aware of this.
8. **Check `namm/evaluation.py`** (`MemoryHFEvaluator`) — does it enforce a hard cap independently of `selection_criteria`? If so, that cap must also be lifted in threshold mode.
9. **Check `wandb` logging** in the trainer — is `cache_size` logged as a scalar? In threshold mode it will vary per step; log the mean/max instead.

After reading, produce a **written plan** listing:
- Every file that needs to change and why.
- Any risk of breakage (e.g., downstream scripts that pass `cache_size` to `run_namm.py`).
- The exact approach you will use (see below).

Only proceed to Step 2 after the plan is complete.

---

## Step 2 — Implement

### Primary approach (mirror `check_eviction_stats.py`)

The safest minimal change is a post-construction hook in `run_namm.py`, gated on a new Hydra config flag `threshold_only` (default `false`):

```python
# In run_namm.py, after make_eval_model() returns:
if cfg.get('threshold_only', False):
    if hasattr(memory_policy, 'selection_criteria'):
        memory_policy.selection_criteria.cache_size = None
        if master_process:
            print("[threshold_only=True] selection_criteria.cache_size set to None — "
                  "eviction driven purely by score threshold (s_i < 0), no hard cap.")
    else:
        raise ValueError(
            "threshold_only=True requested but memory_policy has no selection_criteria. "
            "Check your policy config."
        )
```

**Additionally**, check whether `make_eval_model()` itself passes `cache_size` deep into the evaluator or model wrapper in a way that creates a second cap. If it does, that secondary cap must also be lifted when `threshold_only=True`. Follow the call chain all the way down.

### Config changes

1. Add `threshold_only: false` to `config/config.yaml` (top-level, with a comment explaining it).
2. Check whether any run presets under `config/run/` hardcode a `cache_size` value in a way that conflicts. If a preset is used for threshold-mode training, either create a new preset (e.g., `namm_bam_i1_llama32_1b_threshold.yaml`) or document the required override clearly.
3. Do **not** change `max_memory_length` — it is still used to size internal buffers. Only `selection_criteria.cache_size` (the top-k cutoff) is set to `None`.

### Logging changes

In `trainer.py` (or wherever cache stats are logged to wandb):
- If `cache_size` is logged as a fixed scalar, replace it with a dynamic log of the **mean retained tokens per step** when in threshold mode. This is important for tracking whether the threshold is converging.
- Add a `threshold_only` boolean field to the wandb run config so experiments are filterable.

### Validation

After implementing, verify the change is correct by:
1. Running a **dry-run** (1 CMA-ES iteration, small `max_conditioning_length`, 1 sample) in both modes and confirming:
   - In cache-size mode: retained token count == `cache_size` (hard cap enforced).
   - In threshold mode: retained token count varies across tokens and is **not** clamped to a fixed value; some steps may retain more or fewer tokens than a fixed cap would allow.
2. Confirming the model still saves and loads checkpoints correctly in threshold mode (the checkpoint format should be unchanged — `cache_size=None` is runtime state, not saved state).

---

## Step 3 — Update call sites and documentation

1. **`scripts/check_eviction_stats.py`**: add a comment noting that `--cache_size 0` and `threshold_only=True` in training are now equivalent in effect, so diagnostic results are directly comparable to training runs.
2. **`README.md`** (or the relevant section): add a short paragraph explaining the two modes and the Hydra override needed to use threshold mode:
   ```
   python scripts/run_namm.py run=namm_bam_i1_llama32_1b threshold_only=true
   ```
3. **`experiment_specification.md`** (if it exists in the repo): if there are NAMM training entries (M2, M3, etc.), add a note that threshold-mode variants can be run by appending `threshold_only=true`.

---

## Constraints

- **Do not break cache-size mode.** The default (`threshold_only=false`) must behave identically to the current codebase.
- **Do not change the checkpoint format.** Existing checkpoints must remain loadable.
- **Do not add new required config fields.** `threshold_only` must default to `false` so all existing launch commands continue to work without modification.
- **Keep the change minimal.** Prefer the post-construction hook approach over refactoring `make_eval_model()`, unless you discover that a second cap inside the evaluator makes the hook insufficient on its own.
- **DDP-safe.** Any print statements about threshold mode must be gated on `master_process`.

---

## Definition of Done

- [ ] `run_namm.py` supports `threshold_only=true` via Hydra override with no other changes needed.
- [ ] Default behaviour (`threshold_only=false`) is unchanged.
- [ ] `config/config.yaml` has `threshold_only: false` documented with a comment.
- [ ] No second hard cap silently re-applies inside `MemoryHFEvaluator` or `MemoryModelWrapper` when `threshold_only=true`.
- [ ] Wandb logs mean retained token count (not a fixed scalar) when `threshold_only=true`.
- [ ] README updated with usage example.
- [ ] Dry-run passes in both modes.
