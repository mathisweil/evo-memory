---
description: Rules for evaluation, reporting, and figure generation
paths:
  - scripts/run_eval.py
  - scripts/eval_namm_splits.py
  - scripts/generate_report.py
  - scripts/generate_paper_figures.py
  - scripts/plot_main_table.py
  - scripts/configs/eval_default.yaml
  - config/run/full_cache_baseline_llama32_1b.yaml
  - config/run/recency_baseline_llama32_1b.yaml
  - config/config_run_eval.yaml
---

# Eval rules

## FAIR-01 at eval time

Every entry in the main results table MUST be produced with:

- `cache_size=1024` — except B0, which uses `null` (no limit) by definition.
- Greedy decoding (`temperature=0.0`).
- The 5-task QA subset via `--override "task@_global_=rh_multi_qa_5t"` (or the equivalent in the config).
- `train_frac=0.7`, `val_frac=0.15`, `split_seed=42` (so the test split is the same 69 prompts every time).

Final F1 numbers MUST come from the **test** split, not val. Val F1 drives early stopping; reporting it as the headline is data leakage.

## Baseline configs

| Run | `--run_config` | NAMM checkpoint | Cache |
|---|---|---|---|
| B0 base, full cache | `full_cache_baseline_llama32_1b` | none | null |
| B1 base + recency | `recency_baseline_llama32_1b` | none | 1024 |

You MUST NOT invent a new "no eviction" run config. Use `full_cache_baseline_llama32_1b`. Likewise B1 MUST use `recency_baseline_llama32_1b` rather than a hand-rolled recency policy — the existing config matches the recency variant cited in the discussion.

## A4 modularity ablation (M4 checkpoint, NAMM on/off)

A4 is two evals over the same M4 checkpoint, differing only in whether `--namm_checkpoint` is passed.

- M4 stages are 0-indexed. With `--num_outer_loops 2`, the final adapter checkpoint lives at `joint_lora/m4_joint_lora/adapter/stage_1/`, NOT `stage_2/`.
- The NAMM checkpoint to pair with it is `joint_lora/m4_joint_lora/namm/latest.pt` — `latest.pt` is overwritten after every NAMM stage and always reflects the most recent stage.
- The "NAMM off" arm MUST omit `--namm_checkpoint` entirely. Passing it with `--cache_size 99999` does NOT disable the policy; it still scores and reorders tokens.

## Output schema (do not break)

`run_eval.py` MUST produce a `results.json` matching:

```json
{ "f1": 0.0, "exact_match": 0.0, "num_samples": 69, "cache_size": 1024, "method": "<method-id>" }
```

Joint runs produce a list of these (one per outer loop). `generate_report.py`, `generate_paper_figures.py`, and the analysis notebooks all index by these keys — adding, renaming, or nesting them silently breaks downstream plots and tables.

## Things you MUST NOT do

- You MUST NOT change the F1 normalisation in `namm/evaluation/` (lowercase, strip articles, strip punctuation, whitespace-collapse). It matches LongBench (Bai et al., ACL 2024) and the M2/M3 GCS checkpoints were selected against this exact metric.
- You MUST NOT add a `--num_samples` cap to baseline or main-table runs to "save time". 69 test prompts is already small; subsampling makes the F1 noise floor exceed the deltas you are trying to measure. `--num_samples` is for smoke tests only.
- You MUST NOT regenerate paper figures into a new directory. Overwrite the existing figure paths so the LaTeX `\includegraphics` calls keep resolving.
- You MUST NOT print the eval log in a format that the existing `generate_report.py` parser does not recognise. If you change the log format, update the parser in the same edit.
