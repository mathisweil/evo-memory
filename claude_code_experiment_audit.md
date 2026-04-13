# Claude Code Task: Experiment & Config Audit + Creation

## Context

This is the `evo-memory` project — fine-tuning LLaMA 3.2-1B-Instruct via LoRA while NAMM (Neural Attention Memory Model) manages the KV cache. We're writing an ACL paper studying what happens when you combine learned KV-cache eviction (NAMM) with parameter-efficient fine-tuning (LoRA).

Read these files first to understand the full project:
- `README.md` — project structure, all scripts, CLI args, config system, output layout
- `experiment_specification.md` — every experiment (B0, B1, M1–M4, A1, A4), exact hyperparams, fairness constraints, completion status
- `analysis_specification.md` — the 10 analysis sections (§0–§9), what's done, what's not
- `scripts/configs/` — all existing YAML config presets

Then read the actual source code of all entry-point scripts to understand what CLI args they accept and how configs are loaded:
- `scripts/run_lora.py`
- `scripts/run_namm.py`
- `scripts/run_joint.py`
- `scripts/run_eval.py`
- `scripts/generate_report.py`

Also read the Hydra config tree:
- `config/config.yaml` (main)
- `config/run/` (all run presets, especially any `*5t*` or `*llama32*` ones)
- `config/task/` (especially `rh_multi_qa_5t` if it exists)
- `config/model/`, `config/policy/`, `config/evolution/`, `config/trainer/`

And the existing script configs:
- `scripts/configs/*.yaml` (all of them)

---

## Task 1: Full Experiment Audit

Cross-reference `experiment_specification.md` §6 (Completed Runs) against the actual codebase. For every experiment (B0, B1, M1-r4/r8/r16, M2, M3, M4, A1, A4), determine:

1. **Config existence**: Does the YAML config file referenced in the experiment spec actually exist? For example, `experiment_specification.md` references `scripts/configs/lora_rh_m1_instruct_5t.yaml` and `scripts/configs/lora_rh_m4_instruct_5t.yaml` — do these exist, or only the non-`_5t` variants (`lora_rh_m1_instruct.yaml`, `lora_rh_m4_instruct.yaml`)?

2. **Config correctness**: For each config that exists, do the values inside match the experiment spec exactly? Check every single parameter: `learning_rate`, `num_epochs`, `batch_size`, `gradient_accumulation_steps`, `lora_rank`, `lora_alpha`, `lora_dropout`, `lora_target_modules`, `warmup_ratio`, `weight_decay`, `max_grad_norm`, `sft_mode`, `namm_active`, `cache_size`, `max_seq_len`, `eval_interval`, task list, split params, filtering params, seed.

3. **FAIR-01 compliance**: Verify that every config enforces the fairness constraints: identical 5-task subset, `train_frac=0.7`, `val_frac=0.15`, `split_seed=42`, `min_conditioning_length=4096`, `max_conditioning_length=6500`, `max_answer_tokens=64`, and `temperature=0.0` for evaluation.

4. **Hydra config completeness**: Check if `config/run/namm_bam_i1_llama32_1b_5t.yaml` exists and matches M2 spec. Check if `config/run/full_cache_baseline_llama32_1b.yaml` and `config/run/recency_baseline_llama32_1b.yaml` exist for B0/B1.

5. **Script compatibility**: Read each script's argparse/config loading code to confirm the CLI commands in the experiment spec will actually work. Flag any args that don't exist or are named differently in the code.

Write the audit as `docs/experiment_config_audit.md` with a table per experiment showing: expected config path, exists (Y/N), parameter mismatches, missing CLI args, FAIR-01 violations.

---

## Task 2: Create All Missing Configs

Based on the audit, create every YAML config that is referenced in `experiment_specification.md` but doesn't exist. The configs that likely need creating or updating:

### 2a. `scripts/configs/lora_rh_m1_instruct_5t.yaml`
M1 LoRA-only config for the 5-task QA subset. Must match these params exactly:
- `method`: `rh_m1_lora_instruct_5t`
- `sft_mode`: true
- `learning_rate`: 5e-5
- `num_epochs`: 150
- `batch_size`: 4
- `gradient_accumulation_steps`: 4
- `max_seq_len`: 7000
- `namm_active`: false
- `eval_interval`: 2
- `lora_rank`: 8 (default; overridden to 4/16 for sweep)
- `lora_alpha`: 16
- `lora_dropout`: 0.1
- `lora_target_modules`: [q_proj, v_proj]
- `warmup_ratio`: 0.03
- `weight_decay`: 0.01
- `max_grad_norm`: 1.0
- `seed`: 42
- Task/split/filter params matching FAIR-01

### 2b. `scripts/configs/lora_rh_m4_instruct_5t.yaml`
M3 LoRA + frozen NAMM config. Must match:
- `method`: `rh_m4_frozen_5t`
- `sft_mode`: true
- `learning_rate`: 1e-4
- `num_epochs`: 150
- `batch_size`: 1
- `gradient_accumulation_steps`: 16
- `max_seq_len`: 7000
- `namm_active`: true (requires --namm_checkpoint at runtime)
- `cache_size`: 1024
- `lora_rank`: 8
- `lora_alpha`: 16
- `lora_dropout`: 0.05
- `lora_target_modules`: [q_proj, v_proj]
- `eval_interval`: 2
- `seed`: 42

### 2c. `scripts/configs/joint_lora_m4_5t.yaml` (or update `joint_default.yaml`)
M4 joint training config. Must match:
- `adapter_type`: lora
- `num_outer_loops`: 2
- `namm_iterations_per_stage`: 100
- `lora_epochs_per_stage`: 75
- `lora_rank`: 8
- `lora_alpha`: 16
- `learning_rate`: 5e-5
- `sft_mode`: true
- `cache_size`: 1024
- `eval_after_each_loop`: true
- Task/split/filter matching FAIR-01

### 2d. `scripts/configs/eval_b0_5t.yaml` and `scripts/configs/eval_b1_5t.yaml`
Eval configs for baselines if `eval_default.yaml` doesn't cover them. B0 needs `cache_size: null` (full cache), B1 needs `cache_size: 1024` with recency policy.

### 2e. Hydra run configs
If any of these don't exist in `config/run/`:
- `namm_bam_i1_llama32_1b_5t.yaml`
- `full_cache_baseline_llama32_1b.yaml`
- `recency_baseline_llama32_1b.yaml`

**Important**: Base new configs on existing ones (e.g. `lora_rh_m1_instruct.yaml` → `lora_rh_m1_instruct_5t.yaml`) to maintain consistency with the config loading system. Read the script source to understand which keys are expected.

---

## Task 3: Create a Master Run Script

Create `scripts/run_all_experiments.sh` — a sequential bash script that runs every experiment in dependency order with proper error handling. It should:

1. Accept an `EXPERIMENT_DIR` argument (e.g. `experiments/experiment_5`)
2. Run smoke tests first (abort if any fail)
3. Execute in the order from `experiment_specification.md` §1:
   - B0, B1 (baselines, parallel-safe)
   - M1-r4, M1-r8, M1-r16 (LoRA rank sweep)
   - M2 (standalone NAMM)
   - M3 (LoRA + frozen NAMM — depends on M2 checkpoint)
   - M4 (joint — no dependency on M2)
   - A4 (depends on M4 checkpoint)
4. After each training run, automatically trigger the corresponding eval with `cache_size=1024` and `temperature=0.0`
5. After all runs complete, call `scripts/generate_report.py`
6. Log everything to `$EXPERIMENT_DIR/run_all.log`
7. Include `set -euo pipefail` and trap for cleanup
8. Print a summary table at the end showing which experiments succeeded/failed

---

## Task 4: Validate Compute Budget Matching

The experiment spec defines compute anchors:
- M1: 150 epochs × effective batch 16 on 306 training samples
- M2: 200 CMA-ES generations
- M4: 2 outer loops × (100 NAMM iters + 75 LoRA epochs) = 200 NAMM iters + 150 LoRA epochs (matching M1+M2)

Verify that:
1. The total gradient steps for M1 = `ceil(306/16) × 150` = `20 × 150` = 3000 steps. Check this is consistent with what `run_lora.py` computes.
2. M4's LoRA stages use the same effective batch size as M1 (check `joint_default.yaml` or `run_joint.py`).
3. M3 uses `batch_size=1, grad_accum=16` (effective=16) to fit in memory with NAMM active — confirm `run_lora.py` supports this when NAMM is loaded.

Write findings to `docs/compute_budget_validation.md`.

---

## Task 5: Create Evaluation Configs for All Conditions

Each trained model needs a standardised eval pass. Create `scripts/configs/eval_*.yaml` configs (or a single parameterised one) that covers all conditions from the main results table:

| Condition | es_checkpoint | namm_checkpoint | cache_size | run_config |
|-----------|--------------|-----------------|------------|------------|
| B0 | null | null | null (full) | full_cache_baseline_llama32_1b |
| B1 | null | null | 1024 | recency_baseline_llama32_1b |
| M1-r8 | M1 LoRA adapter path | null | 1024 | — |
| M2 | null | M2 NAMM path | 1024 | — |
| M3 | M3 LoRA adapter path | M2 NAMM path | 1024 | — |
| M4 | M4 LoRA adapter path | M4 NAMM path | 1024 | — |
| A4-on | M4 LoRA adapter path | M4 NAMM path | 1024 | — |
| A4-off | M4 LoRA adapter path | null | null (full) | — |

All evals must use `temperature=0.0`, the 5-task test split (69 samples), and report both F1 and exact match.

---

## Task 6: Fix the Naming Confusion

The experiment spec documents a critical naming issue: WandB uses "M4" (`rh_m4_frozen`) to refer to what the spec calls M3 (LoRA + frozen NAMM). And `results/main_table_5t/M4/` actually contains M3 results.

1. Read `scripts/organize_eval_results.py` (if it exists) and document every place where "M4" is used to mean M3.
2. Create a `docs/naming_mapping.md` that provides the definitive mapping:
   - Experiment spec name → WandB name → config method name → results directory
3. Check if any config files use `rh_m4` to mean frozen-NAMM (M3) vs joint (M4) and flag the ambiguity.
4. Recommend whether to rename things in the codebase or just document the mapping (renaming is risky if existing checkpoints reference old names).

---

## Task 7: Gap Analysis for Paper Readiness

Based on all of the above, produce `docs/paper_readiness_gaps.md` covering:

1. **Missing experiments**: What still needs to run? (M4 joint, M1-r4, M1-r16, re-run A4 on M4 instead of M3)
2. **Missing configs**: Summary of what was created in Task 2
3. **Missing analyses**: §8 (probing) and §9 (gradient flow) from `analysis_specification.md` — are these feasible given time? What code changes would they require?
4. **Results inconsistencies**: The completion summary in the experiment spec shows two different versions of status (one says B0/B1 "not started", the later section says "done"). Reconcile these — which is the ground truth?
5. **Broken runs**: M1_recency shows all zeros — investigate `run_lora.py` and `run_eval.py` to hypothesise why LoRA + recency eviction fails. Is this a code bug or expected behaviour?
6. **M3 training completeness**: cs1024 and cs2048 crashed before 150 epochs. Check if the best checkpoints (step 340, step 244) are from early enough that more training might help, or if the learning curves had plateaued.
7. **Estimated compute time**: Based on the README's estimates (~6h for NAMM cs1024, LoRA timing from WandB), estimate total GPU-hours needed to complete all remaining experiments.

---

## Output Summary

When done, you should have created/modified:

```
docs/
├── experiment_config_audit.md      # Task 1: full audit table
├── compute_budget_validation.md    # Task 4: step count verification
├── naming_mapping.md               # Task 6: M3/M4 confusion resolution
└── paper_readiness_gaps.md         # Task 7: what's left to do

scripts/
├── configs/
│   ├── lora_rh_m1_instruct_5t.yaml    # Task 2a (if missing)
│   ├── lora_rh_m4_instruct_5t.yaml    # Task 2b (if missing)
│   ├── joint_lora_m4_5t.yaml          # Task 2c (if missing)
│   ├── eval_b0_5t.yaml                # Task 2d (if missing)
│   ├── eval_b1_5t.yaml                # Task 2d (if missing)
│   └── eval_main_table.yaml           # Task 5: unified eval config
├── run_all_experiments.sh              # Task 3: master runner
```

Plus any Hydra configs in `config/run/` that were missing.

---

## Important Notes

- **Do not modify any existing config that has already been used for a completed run** — this would invalidate reproducibility. Create new `_5t` variants instead.
- **Read the actual Python source** before creating configs. The YAML keys must match what the scripts expect (e.g. `run_lora.py` may call it `lr` not `learning_rate`).
- **The `_5t` suffix** distinguishes configs for the 5-task QA subset from older single-task or full-LongBench configs. Existing `lora_rh_m1_instruct.yaml` may target a different task set.
- **Hydra configs** (`config/`) use a different system from script configs (`scripts/configs/`). Don't mix them up — `run_namm.py` uses Hydra, everything else uses argparse + YAML.
- When in doubt about a parameter value, the **experiment specification is the source of truth** — configs must match it, not the other way around.
