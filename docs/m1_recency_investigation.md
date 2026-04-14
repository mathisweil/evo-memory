# M1 + Recency Eviction All-Zeros Investigation

**Date:** 2026-04-14

The `results/main_table_5t/M1_recency/cs{1024,2048}/` evaluations report
F1 = 0.00 for every task. This document traces the cause, shows it is a
**command-line misuse, not a model-level failure**, and gives the corrected
command.

---

## What the runs actually did

The command recorded in `results/main_table_5t/M1_recency/cs1024/command.sh`:

```bash
python scripts/eval_namm_splits.py \
    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
    --filter_by_length 8192 --cache_size 1024 --batch_size 8 \
    --splits test extended_test --run_label ext \
    --output_dir eval_results/lora_m1_recency_cs1024_5t
```

This invocation passes `--lora_checkpoint` but NOT `--namm_checkpoint` and NOT
`--use_classic_recency`. Trace through `scripts/eval_namm_splits.py`:

1. **Line 262-265**: `make_eval_model(cfg=cfg)` is called with the default
   Hydra `run_config` (`namm_bam_i1_llama32_1b_5t`), which instantiates the
   full NAMM policy stack (BAM scoring network, DeepMP classifier).
2. **Line 321-327** — the critical branch. With no NAMM checkpoint and no
   `--use_classic_recency`, the script takes the `else` branch:

   ```python
   init_param = memory_policy.get_init_param_values()
   params = init_param.unsqueeze(0).to(device)
   memory_model.set_memory_params(params)
   print("  No checkpoint — using init params "
         "(recency eviction baseline)")
   eval_mode = "recency_baseline"
   ```

   This sets NAMM to its **untrained init parameters** with
   `scoring_initializer=0` (the default in the `namm_bam_i1_llama32_1b_5t`
   preset). Despite the log string calling it a "recency eviction baseline",
   this is **not** StreamingLLM recency — it is NAMM with random-init
   scoring.
3. **Line 338-382**: the LoRA adapter is applied and loaded. This is
   correct — the LoRA weights *are* active.

## Why F1 = 0 specifically

The eviction stats recorded in `results.json` confirm the cache is at budget
(`avg_final_cache_size: 1024.0`, `budget_utilization_pct: 91.56%`). So tokens
are being evicted. But *which* tokens?

Per `.claude/rules/namm.md`:

> With the default `scoring_initializer=0`, the CMA-ES mean starts at the
> eviction boundary (score=0). The first perturbation pushes every token below
> zero and the policy collapses to evict-everything before learning anything.

Here CMA-ES has not run at all — we are at the init point with no training.
With `scoring_initializer=0`, scores cluster around zero. The top-k selection
over a cache-size-1024 budget degenerates to approximately arbitrary token
selection, heavily biased by tie-breaking behaviour in the scoring path. The
LoRA-adapted model, which was trained expecting coherent context, receives
effectively-scrambled attention and emits text that has zero word overlap
with the ground-truth answers — hence F1 = 0.00 on every prompt.

The `extended_test` split shows the same 0.00 pattern, which confirms it is
the policy, not a data-specific artefact.

## Is this a code bug?

**No.** `eval_namm_splits.py` behaves exactly as intended: when the user does
not specify what eviction policy they want, it runs NAMM with init params.
The misleading part is the log string and the `eval_mode = "recency_baseline"`
label, which imply the user got StreamingLLM-style recency when they did not.

## What the run SHOULD have done

There are two correct ways to get "M1 LoRA + recency eviction at cs=1024":

### Option A — `--use_classic_recency` via `eval_namm_splits.py`

```bash
python scripts/eval_namm_splits.py \
    --lora_checkpoint results/rh_m1_lora_instruct_5t/42/best_ckpt.pt \
    --use_classic_recency \
    --cache_size 1024 --batch_size 8 \
    --splits test \
    --output_dir eval_results/lora_m1_recency_cs1024_classic
```

`--use_classic_recency` (line 276-287) swaps the whole policy for
`namm.policy.base.Recency`, which is the StreamingLLM last-N policy.

### Option B — `--run_config recency_baseline_llama32_1b` via `run_eval.py`

`run_eval.py` accepts a separate recency-baseline Hydra preset that uses
`policy@_global_: none` (see `config/run/recency_baseline_llama32_1b.yaml`).
This is how the B1 baseline was done. To combine with an LoRA checkpoint:

```bash
python scripts/run_eval.py \
    --config scripts/configs/eval_main_table.yaml \
    --run_config recency_baseline_llama32_1b \
    --es_checkpoint <path>/best_ckpt.pt \
    --cache_size 1024 \
    --output_dir experiments/experiment_N/a_recency/m1_recency_cs1024
```

Both options should produce non-zero F1. Option A is the simpler fix for
re-running the failed M1_recency/cs1024 and cs2048 rows.

## Is there a useful finding in this result for the paper?

A nuanced one: **the failure shows that "NAMM with zero-init scoring" is
catastrophically worse than a true recency baseline for M1-adapted models**.
This is weak evidence that NAMM is not a "regularisation trick" — a randomly-
scoring evictor is not a useful baseline, as one might naively hope.

But it is NOT evidence that "M1 + recency eviction is broken". That claim
requires Option A or B to be run. Mark the M1_recency row as "needs rerun,
tracked in `docs/m1_recency_investigation.md`" in the results table until a
proper rerun is done.

## Recommended action

Add an M1_recency rerun to `scripts/run_all_experiments.sh` using Option A
(simpler, no Hydra config change needed). Until then, **omit the M1_recency
row from the paper's main results table** — the zero F1s are an artefact of
the command, not a result.

## Secondary finding: the misleading log in `eval_namm_splits.py`

At `scripts/eval_namm_splits.py:321-327`, the print statement says
`"No checkpoint — using init params (recency eviction baseline)"` and sets
`eval_mode = "recency_baseline"`. This is the same label as the correct
recency baseline and is easy to misread. A future patch should either
(a) rename the mode to `"namm_init_baseline"` or similar, or (b) raise an
error when `--lora_checkpoint` is passed with no `--namm_checkpoint` and no
`--use_classic_recency`, since that combination is a likely mistake. This
is out of scope for the current config review but worth an issue.
