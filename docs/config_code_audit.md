# Config-vs-Code Audit

**Date:** 2026-04-13

This document verifies that every config key is read by its target script, checks defaults, types, and FAIR-01 compliance.

---

## Active Configs (FAIR-01 compliant)

### `lora_rh_m1_instruct_5t.yaml` (M1)

Target script: `scripts/run_lora.py`

| Config Key | Argparse Name | Script Line | Read? | Type OK? | Notes |
|------------|--------------|-------------|-------|----------|-------|
| `run_name` | `--run_name` | 59 | YES | str | |
| `experiment` | `--experiment` | 60 | YES | int/None | |
| `method` | `--method` | 61 | YES | str | |
| `lora_rank` | `--lora_rank` | 65 | YES | int | |
| `lora_target_modules` | `--lora_target_modules` | 66 | YES | list[str] | |
| `lora_alpha` | `--lora_alpha` | 67 | YES | int/None | |
| `lora_dropout` | `--lora_dropout` | 68 | YES | float | |
| `learning_rate` | `--learning_rate` | 69 | YES | float | |
| `weight_decay` | `--weight_decay` | 70 | YES | float | |
| `max_grad_norm` | `--max_grad_norm` | 71 | YES | float | |
| `warmup_ratio` | `--warmup_ratio` | 72 | YES | float | |
| `num_epochs` | `--num_epochs` | 73 | YES | int | |
| `batch_size` | `--batch_size` | 74 | YES | int | |
| `gradient_accumulation_steps` | `--gradient_accumulation_steps` | 75 | YES | int | |
| `max_seq_len` | `--max_seq_len` | 76 | YES | int | |
| `sft_mode` | `--sft_mode` | 77 | YES | bool | `action="store_true"` in argparse |
| `namm_active` | `--namm_active` | 79 | YES | bool | `action="store_true"` in argparse |
| `namm_checkpoint` | `--namm_checkpoint` | 80 | YES | str/None | |
| `run_config` | `--run_config` | 81 | YES | str | |
| `cache_size` | `--cache_size` | 82 | YES | int/None | |
| `filter_by_tokens` | `--filter_by_tokens` | 85 | YES | int/None | |
| `filter_answers_by_tokens` | `--filter_answers_by_tokens` | 86 | YES | int | |
| `train_split` | `--train_split` | 87 | YES | float | |
| `val_split` | `--val_split` | 88 | YES | float | |
| `split_seed` | `--split_seed` | 89 | YES | int | |
| `eval_interval` | `--eval_interval` | 92 | YES | int | |
| `log_interval` | `--log_interval` | 93 | YES | int | |
| `batch_size_eval` | `--batch_size_eval` | 94 | YES | int/None | |
| `early_stopping_patience` | `--early_stopping_patience` | 95 | YES | int | |
| `wandb_entity` | `--wandb_entity` | 107 | YES | str | |
| `wandb_project` | `--wandb_project` | 108 | YES | str | |
| `gcs` | `--gcs` | 101 | YES | bool | `BooleanOptionalAction` |
| `resume_checkpoint` | `--resume_checkpoint` | 103 | YES | str/None | |
| `override` | `--override` | 113 | YES | list[str] | |

**FAIR-01 compliance:**
- `split_seed=42`: YES
- `train_split=0.7`: YES (`train_frac` maps to `train_split`)
- `val_split=0.15`: YES
- `min_conditioning_length=4096`: NOT in config, but `run_config=namm_bam_i1_llama32_1b_5t` sets it in Hydra config
- `max_conditioning_length=6500`: NOT in config, set in Hydra config
- `max_answer_tokens=64`: YES (via `filter_answers_by_tokens=64`)

**Issues found:**
1. **`sft_mode` type mismatch:** Config has `sft_mode: true` (bool), but argparse defines `--sft_mode` as `action="store_true"` with `default=False`. When loaded via `load_config_defaults` -> `set_defaults`, the YAML `true` value overrides the default correctly. No bug, but if someone passes `--sft_mode false` on CLI it won't work (store_true doesn't take arguments). This is a minor issue since the config always sets it.

2. **`batch_size` discrepancy:** Config says `batch_size: 1` (recently changed for 8GB VRAM), but the spec says `batch_size: 4` with `gradient_accumulation_steps: 4`. The user modified this for their hardware. The original M1 run used batch_size=4, grad_accum=4 on RTX 3090 Ti.

---

### `lora_rh_m4_instruct_5t.yaml` (M3)

Target script: `scripts/run_lora.py`

Same key mapping as M1 (same script). All keys are read. Types are correct.

**FAIR-01 compliance:**
- `split_seed=42`: YES
- `train_split=0.7`: YES
- `val_split=0.15`: YES
- `min_conditioning_length`: Set in Hydra config via `run_config`
- `max_conditioning_length`: Set in Hydra config via `run_config`
- `filter_answers_by_tokens=64`: YES
- `cache_size=1024`: YES

**Issues found:**
1. **`learning_rate: 1e-4`** does not match M1's `5e-5`. Documented confound (see critical review 1c).
2. **`lora_dropout: 0.05`** does not match M1's `0.1`. Documented confound.
3. **`eval_interval: 2`** — very frequent eval. With batch_size=1 and grad_accum=16, this means eval every 2 optimization steps (32 forward passes). This is expensive but ensures fine-grained checkpoint selection.

---

### `joint_lora_m4_5t.yaml` (M4)

Target script: `scripts/run_joint.py`

| Config Key | Argparse Name | Script Line | Read? | Type OK? | Notes |
|------------|--------------|-------------|-------|----------|-------|
| `run_name` | `--run_name` | 159 | YES | str | |
| `experiment` | `--experiment` | 161 | YES | int/None | |
| `adapter_type` | `--adapter_type` | 165 | YES | str | |
| `num_outer_loops` | `--num_outer_loops` | 168 | YES | int | |
| `namm_iterations_per_stage` | `--namm_iterations_per_stage` | 171 | YES | int | |
| `adapter_iterations_per_stage` | `--adapter_iterations_per_stage` | 174 | YES | int | |
| `lora_epochs_per_stage` | `--lora_epochs_per_stage` | 176 | YES | int | |
| `eval_after_each_loop` | `--eval_after_each_loop` | 179 | YES | bool | |
| `run_config` | `--run_config` | 188 | YES | str | |
| `namm_checkpoint` | `--namm_checkpoint` | 191 | YES | str/None | |
| `cache_size` | `--cache_size` | 193 | YES | int/None | |
| `namm_eval_interval` | `--namm_eval_interval` | 194 | YES | int | |
| `sigma` | `--sigma` | 198 | YES | float | |
| `alpha` | `--alpha` | 199 | YES | float | |
| `population_size` | `--population_size` | 200 | YES | int | |
| `noise_mode` | `--noise_mode` | 201 | YES | str | |
| `initial_seed` | `--initial_seed` | 203 | YES | int | |
| `mini_batch_size` | `--mini_batch_size` | 204 | YES | int | |
| `lora_rank` | `--lora_rank` | 207 | YES | int | |
| `lora_target_modules` | `--lora_target_modules` | 208 | YES | list[str] | |
| `lora_alpha` | `--lora_alpha` | 210 | YES | int/None | |
| `lora_dropout` | `--lora_dropout` | 211 | YES | float | |
| `learning_rate` | `--learning_rate` | 212 | YES | float | |
| `weight_decay` | `--weight_decay` | 213 | YES | float | |
| `max_grad_norm` | `--max_grad_norm` | 214 | YES | float | |
| `warmup_ratio` | `--warmup_ratio` | 215 | YES | float | |
| `gradient_accumulation_steps` | `--gradient_accumulation_steps` | 216 | YES | int | |
| `max_seq_len` | `--max_seq_len` | 217 | YES | int | |
| `sft_mode` | `--sft_mode` | 218 | YES | bool | BooleanOptionalAction |
| `train_split` | `--train_split` | 223 | YES | float | |
| `val_split` | `--val_split` | 224 | YES | float | |
| `split_seed` | `--split_seed` | 225 | YES | int | |
| `min_conditioning_length` | `--min_conditioning_length` | 226 | YES | int | |
| `max_conditioning_length` | `--max_conditioning_length` | 228 | YES | int | |
| `max_answer_tokens` | `--max_answer_tokens` | 230 | YES | int | |
| `filter_by_tokens` | `--filter_by_tokens` | 233 | YES | int/None | |
| `filter_answers_by_tokens` | `--filter_answers_by_tokens` | 234 | YES | int | |
| `batch_size` | `--batch_size` | 237 | YES | int/None | |
| `override` | `--override` | 240 | YES | list[str] | |

**FAIR-01 compliance:** All FAIR-01 fields are present and correct:
- `split_seed=42`: YES
- `train_split=0.7`: YES
- `val_split=0.15`: YES
- `min_conditioning_length=4096`: YES
- `max_conditioning_length=6500`: YES
- `max_answer_tokens=64`: YES
- `cache_size=1024`: YES
- `learning_rate=5e-5`: YES (matches M1)
- `sft_mode=true`: YES

**Issues found:**
1. **`max_seq_len: 3500`** — lower than M1's 7000. With prompts of 4096-6500 tokens, the SFT dataset truncates sequences to 3500 tokens. Since `label_start` (prompt length) is ~4096-6500, and max_seq_len is 3500, the answer portion may be cut short or entirely missing. **This is a potential bug** — investigate whether `SFTDataset` handles the case where `label_start > max_seq_len`.

    **Code check (datasets.py:217-218):** `if len(full_ids) > max_seq_len: full_ids = full_ids[:max_seq_len]`. If the prompt is 5000 tokens and max_seq_len=3500, the full sequence is truncated to 3500 tokens, but label_start remains 5000. The collate function (`pad_collate_fn`, line 344-346) sets `labels[i, ls:] = input_ids[i, ls:]` where ls=5000 but the tensor has length 3500. This means NO labels are set (all -100), and the loss is zero. **This would silently skip all training for long prompts.**

    **SEVERITY: CRITICAL.** If M4 is run with `max_seq_len=3500`, the LoRA stages will learn nothing from prompts longer than 3500 tokens (which is the majority, since `min_conditioning_length=4096`). **Must fix to `max_seq_len=7000` to match M1.**

2. **`lora_dropout: 0.0`** — differs from both M1 (0.1) and M3 (0.05). The spec says M4 should match M1's hyperparameters. **Should be 0.1 to match M1.**

---

### `eval_default.yaml` / `eval_main_table.yaml`

Target script: `scripts/run_eval.py`

All keys are read. Types are correct. FAIR-01 fields are properly set in `eval_main_table.yaml`.

**Issues found:**
1. `run_eval.py` does NOT have a `--lora_checkpoint` flag. LoRA checkpoints can only be loaded via the Hydra config's `adapter_path` field or via `eval_namm_splits.py`. The experiment spec's eval commands reference `--es_checkpoint` for LoRA checkpoints, which treats them as ES delta files — this works only if the LoRA checkpoint was saved in the ES-compatible format.

---

## Deprecated Configs (Non-FAIR-01)

| Config | Status | Safe to Use? |
|--------|--------|-------------|
| `lora_default.yaml` | DEPRECATED header | NO — uses `train_split=0.8`, `sft_mode=false`, non-5t `run_config` |
| `lora_m1_only.yaml` | DEPRECATED header | NO — uses non-5t `run_config` |
| `es_m1_only.yaml` | DEPRECATED header | NO — ES path, not LoRA |
| `es_default.yaml` | DEPRECATED header | NO — ES path |
| `lora_rh_m1_instruct.yaml` | DEPRECATED header | NO — 6-task subset, non-5t `run_config` |
| `lora_rh_m4_instruct.yaml` | DEPRECATED header | NO — 3-task subset, non-5t `run_config` |

These are kept for reproducibility of historical runs and should NOT be used for new FAIR-01 experiments.

---

## Dead Keys

| Config | Key | Status |
|--------|-----|--------|
| `lora_rh_m1_instruct.yaml` | `checkpoint_every: 0` | DEAD — no argparse param `checkpoint_every` in `run_lora.py`. Ignored by `set_defaults`. |
| `lora_rh_m4_instruct.yaml` | `checkpoint_every: 0` | DEAD — same as above. |
| `lora_default.yaml` | `checkpoint_every: 0` | DEAD — same as above. |

---

## Critical Findings Summary

1. **`joint_lora_m4_5t.yaml: max_seq_len=3500` is CRITICALLY BROKEN** for the FAIR-01 dataset (prompts > 4096 tokens). Must be 7000. All LoRA training in joint stages would produce zero loss for the majority of samples.

2. **`joint_lora_m4_5t.yaml: lora_dropout=0.0`** does not match M1 (`0.1`). Should be `0.1` if the intent is to match M1.

3. **`joint_default.yaml: learning_rate=2e-4`** does not match M1 (`5e-5`). The corrected `joint_lora_m4_5t.yaml` uses `5e-5`. Always use `joint_lora_m4_5t.yaml` for M4.

4. **M1_recency eval used wrong eviction mode** — ran without `--use_classic_recency`, falling back to random NAMM init params instead of recency.
