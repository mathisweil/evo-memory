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
- You MUST keep `train_frac=0.7`, `val_frac=0.15`, `split_seed=42` (yields 306 train / 64 val / 69 test).
- You MUST keep `min_conditioning_length=4096`, `max_conditioning_length=6500`, `max_answer_tokens=64`.
- You MUST keep all conditions starting from raw `meta-llama/Llama-3.2-1B-Instruct` weights with no pretrained adapters.
- You MUST evaluate every condition with `cache_size=1024` and greedy decoding (`temperature=0.0`).

## Condition map

The codebase predates the M-numbering. The historical method names do not match the spec one-to-one:

| Spec | Method (in YAML / WandB) | Script | Config |
|---|---|---|---|
| M1 LoRA-only | `rh_m1_lora_instruct_5t` | `run_lora.py` | `scripts/lora_rh_m1_instruct_5t.yaml` |
| M3 LoRA + frozen NAMM | `rh_m4_frozen_5t` | `run_lora.py` | `scripts/lora_rh_m4_instruct_5t.yaml` |
| M4 joint LoRA + NAMM | `joint_lora` | `run_joint.py --adapter_type lora` | `scripts/configs/joint_default.yaml` |

You MUST NOT rename `rh_m4_frozen_5t` → `m3_*` or otherwise "fix" the historical naming. WandB run history, GCS checkpoint paths, and `experiments/experiment_N/m1_lora_only/...` directories all depend on the existing strings.

## LoRA hyperparameters

- M1: `learning_rate=5e-5`, `num_epochs=150`, `batch_size=4`, `gradient_accumulation_steps=4` (effective batch = 16). Default rank `r=8`, `alpha=16`. Sweep variants `r=4 alpha=8` and `r=16 alpha=32` are M1-r4 / M1-r16; you MUST keep `alpha = 2 * rank` when adjusting rank.
- M3: `learning_rate=1e-4`, `num_epochs=150`, `batch_size=1`, `gradient_accumulation_steps=16` (effective batch = 16). The smaller per-step batch is intentional — NAMM eviction state is per-sequence and the larger batch OOMs.
- M4: must call `run_joint.py --adapter_type lora --num_outer_loops 2 --namm_iterations_per_stage 100 --lora_epochs_per_stage 75`. Totals (200 NAMM gens, 150 LoRA epochs) MUST match M1+M2 budgets — that is what makes the comparison fair.
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

Training runs MUST write under `experiments/experiment_N/<condition>/<run_name>/` with `config.json`, `results.json`, and `checkpoints/` subdirectories. `joint_*` runs additionally produce `namm/latest.pt` and `adapter/stage_K/` subdirectories. Stage indices are 0-based: with `--num_outer_loops 2`, the final adapter stage is `stage_1`.
