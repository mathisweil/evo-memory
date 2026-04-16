# Investigate why HEAD cannot reproduce commit `650684f` (green run)

## Context

A prior run on commit `650684f5e582fde79a45ae298ac2d8ca14b6f5f8` achieved ~45-50 val
`avg_f1` on the LongBench tasks (qasper, hotpotqa, 2wikimqa, etc.). On HEAD, even
with a new `warm_buffers=True` flag that re-enables the cross-document NAMM
component buffer leakage present in `650684f`, reproduction tops out at ~22 val
`avg_f1` and ~30 train `avg_f1`. The gap appears on **training metrics too**, so
this is not an eval-only or overfitting issue — the model is not learning as
well.

**Green run command (commit `650684f`):**
```
scripts/run_lora.py \
  --config scripts/lora_rh_m4_instruct_5t.yaml \
  --run_name rh_m4_5t_cs1024_maskfix \
  --wandb_project memory_evolution_hf \
  --wandb_run_name rh_m4_5t_cs1024_maskfix \
  --no-gcs \
  --eval_interval 2 \
  --cache_size 1024 \
  --namm_checkpoint eval_results/namm_cs1024_maskfix/ckpt.pt
```

**Failing reproduction command (HEAD, with warm_buffers):**
```
scripts/run_lora.py \
  --config scripts/configs/m3_warm_lora_5t.yaml \
  --run_name m3_warm_r8_cs1024_lr1-4_dr005_ep150 \
  --lora_dropout 0.05 \
  --learning_rate 1e-4 \
  --num_epochs 150 \
  --early_stopping_patience 40 \
  --namm_checkpoint artifacts/NAMM_best_1024/ckpt.pt \
  --cache_size 1024
```

The **exact same NAMM checkpoint** was used for both runs (the file at the two
paths is byte-identical — confirmed with sha256).

The buffer-reset difference (`trainer.py:473-476`) has already been neutralised
by `warm_buffers=True`. Something else is responsible for the gap.

## Your task

Conduct a deep, exhaustive diff between `650684f` and HEAD. Do **not** restrict
yourself to `trainer.py` — the buffer question is already closed. Find
everything else that could plausibly affect learning dynamics or final metrics.

### 1. Hyperparameter / config diff (do this first — cheapest)

- `git show 650684f:scripts/lora_rh_m4_instruct_5t.yaml` and compare every key
  against `scripts/configs/m3_warm_lora_5t.yaml` **plus** the CLI overrides on
  the failing command. Produce a side-by-side table.
- Pay special attention to: `learning_rate`, `num_epochs`, `batch_size`,
  `gradient_accumulation_steps`, `warmup_ratio`, `max_grad_norm`,
  `weight_decay`, `lora_rank`, `lora_alpha`, `lora_dropout`, `max_seq_len`,
  `task_names`, `namm_active`, `eval_interval`, any task-sampling weights,
  any scheduler settings, and any field present in one config but absent in
  the other.
- Note that the green YAML name contains `rh_m4_instruct` — check whether this
  was an **instruct-tuned base model** config and whether HEAD's config points
  to a different base checkpoint. If base model differs, that alone can
  explain everything.

### 2. Full code diff across the commits

```
git diff --stat 650684f..HEAD -- '*.py' '*.yaml' '*.json'
```

Then for every changed file, decide whether the change could affect training
dynamics. Focus areas in priority order:

**NAMM / memory policy**
- `namm/` subtree in full. Any change to `memory_policy.py`, component
  implementations (the ones with `ema_output_buffer`, `stored_keys_buffer`,
  `prev_attn_buffer`), `initialize_buffers`, `set_params_batch_idxs`,
  eviction logic, or the two-phase forward integration.
- How the NAMM checkpoint is loaded in `run_lora.py`. Look for changes to
  the load path, `strict` flags, parameter-slot indexing, or post-load
  buffer initialization. The fact that HEAD added
  `set_params_batch_idxs([0])` inside the train loop suggests the NAMM load
  semantics may have shifted.

**Model / LoRA**
- LoRA adapter code: insertion points, target modules, scaling (`alpha/r`),
  dropout behaviour, merging logic.
- Any changes to the base model wrapper, especially anything touching
  `output_attentions`, `use_cache`, `apply_memory_policy`,
  `memory_policy_fixed_delay`, gradient checkpointing, or hidden-state
  outputs.

**Data pipeline**
- Dataset loader, tokenizer, `pad_collate_fn`, task sampler, shuffle seed
  handling. The warm-buffers regime makes **document order** part of the
  effective model, so any change to sampling / shuffling changes what the
  network sees even with identical buffer leakage.
- Check whether split seeds, task weights, or max-sample caps changed.
- Check whether the `labels` masking convention (what gets `-100`) changed
  — the two-phase forward derives `context_end` from the first non-`-100`
  label, so any upstream change to label construction shifts the phase
  split.

**Training loop outside `_train_step`**
- Optimizer / scheduler construction, warmup, gradient clipping, EMA code,
  checkpoint save/load, early stopping.
- The `skip_baseline_eval` flag, `gc.collect()` / `torch.inference_mode()`
  changes, CPU-side EMA snapshot — unlikely to cause this gap but note if
  any of them touch the training graph.

**Evaluation**
- If eval code changed, the val numbers may be measured differently even if
  the model is identical. Check whether `evaluator.py`, metric computation,
  or the eval-time buffer-reset logic changed. Report whether eval-time
  buffer handling matches between commits.

### 3. NAMM checkpoint sanity

```
sha256sum eval_results/namm_cs1024_maskfix/ckpt.pt artifacts/NAMM_best_1024/ckpt.pt
```

If they differ, stop and report — the premise is wrong. If they match, load
the checkpoint on HEAD and confirm all keys map cleanly to the current NAMM
module (no missing/unexpected keys, no shape mismatches, no silently-skipped
tensors). Report any `load_state_dict` warnings.

### 4. Produce a ranked hypothesis list

For each concrete difference you find, assign a likelihood (high / medium /
low) that it explains the ~25-point avg_f1 gap, with a one-sentence
justification. Order the final list by likelihood. Flag any change that would
affect training dynamics even under `warm_buffers=True`.

### 5. Propose the cheapest verification experiment

Given the ranked list, suggest the smallest single change (ideally a one-line
revert or a CLI override) that would test the top hypothesis. Provide the
exact command to run.

## Deliverables

1. **Config diff table** (green YAML + green CLI vs HEAD YAML + HEAD CLI).
2. **Categorised code-change summary** per focus area above, with file paths
   and line ranges for each change that could plausibly matter.
3. **NAMM checkpoint verification** result.
4. **Ranked hypothesis list** with likelihoods.
5. **One recommended verification command**.

## Constraints

- Do not modify any code in this task. This is analysis only.
- Do not assume `trainer.py` buffer reset is the cause — it has been ruled out.
- If you find a change whose intent is unclear, quote the commit message /
  surrounding code rather than guessing.
- Be exhaustive on the NAMM and data-pipeline paths specifically; those are
  the two areas most likely to contain the culprit given the symptom pattern
  (train metrics also degraded, warm_buffers already enabled).
