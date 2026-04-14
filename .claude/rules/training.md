---
description: Rules for adapter / joint training scripts and their YAML presets
paths:
  - scripts/run_lora.py
  - scripts/run_joint.py
  - scripts/run_es.py
  - scripts/joint_metrics.py
  - scripts/configs/lora_*.yaml
  - scripts/configs/joint_*.yaml
  - scripts/configs/es_*.yaml
  - scripts/lora_rh_m1_instruct_5t.yaml
  - scripts/lora_rh_m4_instruct_5t.yaml
  - grad_lora_finetuning/**
  - es_finetuning/**
---

# Training rules (LoRA / ES / joint)

## FAIR-01 fairness — non-negotiable

All four main conditions (M1, M2, M3, M4) MUST share the same data, base model, and eval-time memory budget. If you change one of these in any training script or YAML preset, you MUST change it everywhere.

- You MUST keep the 5-task QA subset: `qasper, 2wikimqa, qasper_e, hotpotqa_e, 2wikimqa_e`. The Hydra task override is `task@_global_=rh_multi_qa_5t`.
- You MUST keep `train_frac=0.7`, `val_frac=0.15`, `split_seed=42` (yields 306 train / 64 val / 70 test — verified from `results/main_table_5t/*/cs1024/results.json`; per-task test: qasper=14, 2wikimqa=12, qasper_e=18, hotpotqa_e=12, 2wikimqa_e=14).
- You MUST keep `min_conditioning_length=4096`, `max_conditioning_length=6500`, `max_answer_tokens=64`.
- You MUST keep all conditions starting from raw `meta-llama/Llama-3.2-1B-Instruct` weights with no pretrained adapters.
- You MUST evaluate every condition with `cache_size=1024` and greedy decoding (`temperature=0.0`).

## Condition map

The codebase predates the M-numbering. The historical method names do not match the spec one-to-one:

| Spec | Method (in YAML / WandB) | Script | Config |
|---|---|---|---|
| M1 LoRA-only | `rh_m1_lora_instruct_5t` | `run_lora.py` | `scripts/configs/lora_rh_m1_instruct_5t.yaml` |
| M3 LoRA + frozen NAMM | `rh_m4_frozen_5t` | `run_lora.py` | `scripts/configs/lora_rh_m4_instruct_5t.yaml` |
| M4 joint LoRA + NAMM | `joint_lora` | `run_joint.py --adapter_type lora` | `scripts/configs/joint_lora_m4_5t.yaml` |

You MUST NOT rename `rh_m4_frozen_5t` → `m3_*` or otherwise "fix" the historical naming. WandB run history, GCS checkpoint paths, and `experiments/experiment_N/m1_lora_only/...` directories all depend on the existing strings.

## LoRA hyperparameters

- M1: `learning_rate=5e-5`, `num_epochs=150`, `batch_size=1`, `gradient_accumulation_steps=16` (effective batch = 16), `lora_dropout=0.1`. Default rank `r=8`, `alpha=16`. Sweep variants `r=4 alpha=8` and `r=16 alpha=32` are M1-r4 / M1-r16; you MUST keep `alpha = 2 * rank` when adjusting rank. `batch_size=1` (rather than the spec's older `4`) is used everywhere so M1, M3, M4 have identical per-step processing.
- M3: MUST match M1 exactly except `namm_active=true` and `cache_size=1024`. That means `learning_rate=5e-5`, `lora_dropout=0.1`, `batch_size=1`, `gradient_accumulation_steps=16` (effective batch = 16). Any deviation is a confound — the M1-vs-M3 comparison is the paper's headline result. Historical M3 runs used `learning_rate=1e-4`, `lora_dropout=0.05`; those are now labelled "M3-tuned" and must be re-run (see `docs/m3_rerun_plan.md`).
- M4: must call `run_joint.py --config scripts/configs/joint_lora_m4_5t.yaml --adapter_type lora --num_outer_loops 3 --namm_iterations_per_stage 67 --lora_epochs_per_stage 50`. Totals (201 NAMM gens, 150 LoRA epochs) MUST match M1+M2 budgets — that is what makes the comparison fair. LoRA hyperparameters inside each stage MUST match M1 (`learning_rate=5e-5`, `lora_dropout=0.1`, `lora_batch_size=1`, `gradient_accumulation_steps=16`). The 3-loop schedule supersedes the earlier 2-loop design (see `docs/m4_joint_training_analysis.md`).
- M1, M3, M4 MUST use `early_stopping_patience=20` (or `lora_early_stopping_patience=20` for joint). The prior value of 5 was too aggressive for the 150-epoch schedule.
- M1, M3, M4 MUST use `eval_interval=14` (or `lora_eval_interval=14` for joint) — enables best-of-N checkpoint selection at a practical wall-clock cadence.
- You MUST keep `lora_target_modules=[q_proj, v_proj]` for all M-conditions.
- You MUST keep `sft_mode=true` (chat-template formatted prompt, answer-only loss).

## NAMM activation in training scripts

- M1 MUST run with `namm_active=false` (full KV cache during training).
- M3 MUST run with `namm_active=true` AND `--namm_checkpoint <path-to-M2-checkpoint>`. Failing to pass the checkpoint silently falls back to a randomly initialised NAMM and invalidates the run.
- M4 (joint) MUST start cold — `namm_checkpoint=null`. Warm-starting joint training from a pretrained NAMM is a different experiment and MUST NOT be conflated with M4 in the results table.

## Things you MUST NOT do

- You MUST NOT mock the LLM, tokenizer, or KV cache in training tests. Behaviour under real attention matrices is what matters; mock-only tests have masked real bugs in this codebase before.
- You MUST NOT bypass `transformers==4.41.2` pin. The `DynamicCache` interface used by `grad_lora_finetuning/trainer.py` and `namm/llms/` changed in 4.45+.
- You MUST NOT add `try/except` around the NAMM forward call to "make training robust". A NAMM crash is a real bug; swallowing it produces a silently-disabled-eviction run that looks like M1 in the results.
- You MUST NOT change `seed=42` for LoRA runs or `seed=1337` for NAMM runs without flagging it explicitly to the user — paired seeds across conditions are what make the F1 deltas comparable.
- You MUST NOT add a "resume from checkpoint" flag that loads optimizer state across condition boundaries. Each M-condition is a fresh run.

## Output paths

Training runs MUST write under `experiments/experiment_N/<condition>/<run_name>/` with `config.json`, `results.json`, and `checkpoints/` subdirectories. `joint_*` runs additionally produce `namm/latest.pt` and `adapter/stage_K/` subdirectories. Stage indices are 0-based: with `--num_outer_loops 3`, the final adapter stage is `stage_2`.
