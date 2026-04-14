# Config Comparison Matrix

**Date:** 2026-04-14 (post Part C fixes)

Cross-condition hyperparameter table. All values come from the YAML files as
they stand after the config review. "Match?" flags any inter-condition
difference that is unexplained (i.e. not forced by NAMM memory constraints).

---

## LoRA Hyperparameters

| Parameter | M1 `_5t.yaml` | M3 `_5t.yaml` | M4 `joint_lora_m4_5t.yaml` | Match? | Notes |
|---|---|---|---|---|---|
| `learning_rate` | 5e-5 | 5e-5 | 5e-5 | YES | Was 1e-4 in M3 pre-fix — a 2x confound |
| `lora_rank` | 8 | 8 | 8 | YES | |
| `lora_alpha` | 16 | 16 | 16 | YES | alpha = 2 * rank |
| `lora_dropout` | 0.1 | 0.1 | 0.1 | YES | Was 0.05 in M3 pre-fix |
| `lora_target_modules` | [q_proj, v_proj] | [q_proj, v_proj] | [q_proj, v_proj] | YES | |
| `num_epochs` (total) | 150 | 150 | 150 (2 × 75/stage) | YES | |
| `batch_size` | 1 | 1 | 1 (`lora_batch_size`) | YES | M1 matches M3/M4 despite having memory headroom (controlled comparison) |
| `gradient_accumulation_steps` | 16 | 16 | 16 | YES | effective batch = 16 everywhere |
| `max_seq_len` | 7000 | 7000 | 7000 | YES | |
| `sft_mode` | true | true | true | YES | |
| `warmup_ratio` | 0.03 | 0.03 | 0.03 | YES | Warmup restarts per LoRA stage in M4 |
| `weight_decay` | 0.01 | 0.01 | 0.01 | YES | |
| `max_grad_norm` | 1.0 | 1.0 | 1.0 | YES | |
| `dtype` | bfloat16 | bfloat16 | bfloat16 | YES | |

## Evaluation

| Parameter | M1 | M3 | M4 | Match? | Notes |
|---|---|---|---|---|---|
| `eval_interval` (LoRA) | 14 | 14 | 14 (`lora_eval_interval`) | YES | Was 999999 in M4 pre-fix (no mid-stage eval) |
| `log_interval` | 2 | 2 | 10 (`lora_log_interval`) | DIFFERS | Cosmetic only; logging cadence, not evaluation |
| `batch_size_eval` | 2 | 2 | (via run_config batch=4) | YES | M1/M3 were 1/2 pre-fix |
| `early_stopping_patience` | 20 | 20 | (not used in joint loop) | YES for M1/M3 | Was 5 pre-fix |
| `always_save_checkpoint` | true | true | true | YES | |

## Data / Splitting (FAIR-01)

All values come from the Hydra `run_config: namm_bam_i1_llama32_1b_5t` preset
for LoRA configs (M1, M3) and from explicit YAML keys for joint configs (M4).

| Parameter | M1 | M3 | M4 | B0 | B1 | M2 | Match? |
|---|---|---|---|---|---|---|---|
| `train_frac` | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 | YES |
| `val_frac` | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 | YES |
| `split_seed` | 42 | 42 | 42 | 42 | 42 | 42 | YES |
| `min_conditioning_length` | 4096 (Hydra) | 4096 (Hydra) | 4096 (YAML) | 4096 | 4096 | 4096 | YES |
| `max_conditioning_length` | 6500 (Hydra) | 6500 (Hydra) | 6500 (YAML) | 6500 | 6500 | 6500 | YES |
| `max_answer_tokens` | 64 | 64 | 64 | 64 | 64 | 64 | YES |
| `run_config` | 5t | 5t | 5t | full_cache | recency | 5t | YES (different run_config for baselines is intentional) |

## NAMM / Memory

| Parameter | M1 | M3 | M4 | B0 | B1 | M2 |
|---|---|---|---|---|---|---|
| `namm_active` | false | true | true | false | false | true |
| `namm_checkpoint` | null | M2 ckpt (CLI) | null (cold start) | null | null | N/A (training) |
| `cache_size` (training) | null (full) | 1024 | 1024 | N/A | N/A | 1024 |
| `cache_size` (eval) | 1024 (FAIR-01) | 1024 | 1024 | null (B0 by definition) | 1024 | 1024 |

## CMA-ES (NAMM stages in M2, M4)

All values come from `config/run/namm_bam_i1_llama32_1b_5t.yaml`.

| Parameter | M2 | M4 NAMM stages | Match? |
|---|---|---|---|
| `pop_size` | 8 | 8 | YES |
| `elite_ratio` | 0.5 | 0.5 | YES |
| `init_sigma` | 0.065 | 0.065 | YES |
| `memory_policy_fixed_delay` | 256 | 256 | YES |
| `max_memory_length` | 1024 | 1024 | YES |
| `samples_batch_size` | 8 | 8 | YES |
| `batch_size` | 4 | 4 | YES |
| `max_iters` (total across stages) | 200 | 200 (100/stage × 2) | YES |
| `seed` | 1337 | 1337 | YES |

## Seeds

| Purpose | Value | Conditions |
|---|---|---|
| LoRA training seed (via `split_seed`) | 42 | M1, M3, M4 LoRA stages |
| NAMM training seed (Hydra `seed`) | 1337 | M2, M4 NAMM stages |
| Data split seed | 42 | All conditions |

---

## Remaining inter-condition deltas (all intentional)

1. **`namm_active` + `cache_size` (training)**: false/null for M1, true/1024
   for M3 and M4. This is the independent variable under test — not a confound.
2. **`log_interval`** differs between LoRA configs (2) and joint config (10).
   Logging cadence has no effect on training or evaluation; flagged for
   completeness only.
3. **M4 warmup+optimizer restart per LoRA stage**: M1/M3 run a single 150-epoch
   warmup+decay schedule; M4 runs two 75-epoch schedules back-to-back. See
   `docs/m4_readiness_review.md` for treatment.
4. **M4 `early_stopping_patience` is not plumbed through the joint loop**.
   The joint `_run_lora_stage` constructs a fresh `LoRATrainerConfig` without
   setting `early_stopping_patience`, so it defaults to 0 (off). Each LoRA
   stage runs for its full `lora_epochs_per_stage=75` epochs. This is
   intentional — you cannot stop early inside a stage without breaking the
   outer-loop budget accounting — but it is a difference from M1/M3 worth
   being aware of.

## Legend

| Flag | Meaning |
|---|---|
| YES | Values match across conditions as intended |
| DIFFERS | Values differ but the difference does not affect results |
| N/A | Parameter does not apply to this condition |
