# Config Comparison Matrix

**Date:** 2026-04-13

Cross-condition hyperparameter table. "Actual" columns come from the YAML files. "Match?" flags any inter-condition difference that is unexplained.

---

## LoRA Hyperparameters

| Parameter | M1 (spec) | M1 (actual _5t.yaml) | M3 (spec) | M3 (actual _5t.yaml) | M4 (spec) | M4 (joint_lora_m4_5t.yaml) | Match? | Notes |
|-----------|-----------|---------------------|-----------|---------------------|-----------|---------------------------|--------|-------|
| `learning_rate` | 5e-5 | 5e-5 | 1e-4 | 1e-4 | 5e-5 | 5e-5 | M3 differs | **Confound.** M3 uses 2x M1's LR |
| `lora_rank` | 8 | 8 | 8 | 8 | 8 | 8 | YES | |
| `lora_alpha` | 16 | 16 | 16 | 16 | 16 | 16 | YES | |
| `lora_dropout` | 0.1 | 0.1 | 0.05 | 0.05 | ? | **0.0** | M3, M4 differ | **Confound.** M4 should be 0.1 |
| `lora_target_modules` | [q_proj, v_proj] | [q_proj, v_proj] | [q_proj, v_proj] | [q_proj, v_proj] | [q_proj, v_proj] | [q_proj, v_proj] | YES | |
| `num_epochs` | 150 | 150 | 150 | 150 | 75/stage (150 total) | 75/stage (150 total) | YES | |
| `batch_size` | 4 | 1* | 1 | 1 | 1 | 1 (hardcoded) | YES | *M1 modified for 8GB VRAM |
| `gradient_accumulation_steps` | 4 | 16* | 16 | 16 | 16 | 16 | YES | *Compensated; eff batch=16 for all |
| `effective_batch` | 16 | 16 | 16 | 16 | 16 | 16 | YES | |
| `max_seq_len` | 7000 | 7000 | 7000 | 7000 | ? | **3500** | **M4 BROKEN** | Truncates all prompts; zero-loss training |
| `sft_mode` | true | true | true | true | true | true | YES | |
| `warmup_ratio` | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | YES | |
| `weight_decay` | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | YES | |
| `max_grad_norm` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | YES | |

## Data / Splitting

| Parameter | M1 | M3 | M4 | B0 | B1 | M2 | Match? |
|-----------|----|----|----|----|----|----|--------|
| `train_frac` | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 | YES |
| `val_frac` | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 | YES |
| `split_seed` | 42 | 42 | 42 | 42 | 42 | 42 | YES |
| `min_conditioning_length` | 4096 | 4096 | 4096 | 4096 | 4096 | 4096 | YES |
| `max_conditioning_length` | 6500 | 6500 | 6500 | 6500 | 6500 | 6500 | YES |
| `max_answer_tokens` | 64 | 64 | 64 | 64 | 64 | 64 | YES |
| `run_config` | _5t | _5t | _5t | full_cache | recency | _5t | YES |

## NAMM / Memory

| Parameter | M1 | M3 | M4 | B0 | B1 | M2 |
|-----------|----|----|----|----|----|----|
| `namm_active` | false | true | true | false | false | true |
| `namm_checkpoint` | null | M2 ckpt | null (cold start) | null | null | N/A (training) |
| `cache_size` (training) | null (full) | 1024 | 1024 | N/A | N/A | 1024 |
| `cache_size` (eval) | full | 1024 | 1024 | full | 1024 | 1024 |

## CMA-ES (M2 and M4 NAMM stages)

| Parameter | M2 (spec) | M2 (Hydra _5t) | M4 (spec) | M4 (joint_lora_m4_5t) | Match? |
|-----------|----------|----------------|-----------|----------------------|--------|
| `pop_size` | 8 | 8 | 8 (from Hydra) | 8 (from Hydra) | YES |
| `elite_ratio` | 0.5 | (in CMA-ES config) | 0.5 | (from Hydra) | YES |
| `init_sigma` | 0.065 | (in CMA-ES config) | 0.065 | (from Hydra) | YES |
| `memory_policy_fixed_delay` | 256 | 256 | 256 | (from Hydra) | YES |
| `max_memory_length` | 1024 | 1024 | 1024 | (from Hydra) | YES |
| `samples_batch_size` | 8 | 8 | 8 | (from Hydra) | YES |
| `batch_size` | 4 | 4 | 4 | (from Hydra) | YES |
| `max_iters` | 200 | 200 | 100/stage (200 total) | 100/stage (200 total) | YES |
| `seed` | 1337 | 1337 | 1337 | (from Hydra) | YES |

## Evaluation

| Parameter | All Conditions | Notes |
|-----------|---------------|-------|
| `temperature` | 0.0 | Greedy decoding |
| `num_samples` | 1 | Single sample per prompt |
| `batch_size_eval` | 1 (M1, M3 _5t configs) | Was 4, changed for 8GB VRAM |
| `early_stopping_patience` | 5 | Same across M1, M3 |

## Seeds

| Seed Purpose | Value | Conditions |
|-------------|-------|------------|
| LoRA training seed | 42 (via `split_seed`) | M1, M3, M4 LoRA stages |
| NAMM training seed | 1337 (Hydra `seed`) | M2, M4 NAMM stages |
| Data split seed | 42 (`split_seed`) | All conditions |

---

## Flags Summary

| Flag | Description |
|------|-------------|
| YES | Values match across conditions as intended |
| M3 differs | M3 uses a different value than M1; documented confound |
| M4 BROKEN | M4 config has a value that will cause silent failure |
| *modified | Value was changed from spec for hardware reasons (8GB VRAM) |
