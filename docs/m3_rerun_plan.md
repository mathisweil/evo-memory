# M3 Rerun Plan (D1)

**Date:** 2026-04-14

The M3 runs already in WandB used `learning_rate=1e-4` and `lora_dropout=0.05`,
while M1 used `learning_rate=5e-5` and `lora_dropout=0.1`. Those two
differences confound the M1-vs-M3 comparison that the paper hinges on. After
the Part C1 fix, `m3_lora_frozen_namm_5t.yaml` matches M1 exactly except for
`namm_active: true` and `cache_size: 1024`. M3 must be re-run with the new
config.

## The corrected M3 hyperparameters

From `scripts/configs/m3_lora_frozen_namm_5t.yaml`:

```yaml
learning_rate: 5e-5        # MATCHES M1 (was 1e-4)
lora_dropout: 0.1          # MATCHES M1 (was 0.05)
lora_rank: 8
lora_alpha: 16
batch_size: 1
gradient_accumulation_steps: 16
max_seq_len: 7000
num_epochs: 150
eval_interval: 14
batch_size_eval: 2
early_stopping_patience: 20
namm_active: true
cache_size: 1024
split_seed: 42
run_config: namm_bam_i1_llama32_1b_5t
```

## Command

```bash
venv/bin/python scripts/run_lora.py \
    --config scripts/configs/m3_lora_frozen_namm_5t.yaml \
    --run_name m3_cs1024_matched \
    --namm_checkpoint experiments/experiment_N/m2_namm/<best-m2>/namm/latest.pt \
    --wandb_group_name m3_matched
```

Repeat with `--cache_size 2048` → `--run_name m3_cs2048_matched` and
`--cache_size 3072` → `--run_name m3_cs3072_matched` for the cache sweep, each
with `--wandb_group_name m3_matched`.

## WandB naming

The _matched_ suffix distinguishes the new runs from the tuned-hyperparameter
runs. Existing runs `ovosogkj` (cs1024), `m4knrhmr` (cs2048), `4sgkswa6`
(cs3072) stay under their original names. The new runs use the pattern
`m3_cs{1024,2048,3072}_matched` and group `m3_matched`.

## What to do with the old M3 results

The pre-fix runs (lr=1e-4, dropout=0.05) are kept as a **secondary data point**
labelled "M3-tuned":

- Primary results table: "M3 (matched)" uses the new matched runs. Headline
  M1-vs-M3 comparison uses these.
- Ablation / discussion: "M3-tuned" row cites the old numbers (cs1024: 32.28,
  cs2048: 31.06) as evidence that separately tuning M3 does not close the gap.
- `experiment_specification.md` and the results table MUST label the two sets
  distinctly. The raw WandB run IDs stay in the spec for traceability.

## Expected outcomes

Three possible readings after the rerun:

1. **M3-matched > M1** → NAMM eviction is genuinely helpful even at matched
   hyperparameters. Clean positive result.
2. **M3-matched ≈ M3-tuned > M1** → the original story holds; the tuning was
   not load-bearing.
3. **M3-matched < M1** and **M3-tuned > M1** → the old M3 win came from
   separately tuning the LR, not from NAMM. Paper narrative would have to
   pivot.

Each reading is interesting on its own — the rerun is worth doing regardless
of which outcome appears.

## Compute cost

M1 took ~12 hours on an RTX 3090 Ti (28 epochs, batch_size=1). With NAMM
active, M3 per-step time is higher (~1.4x M1). At `early_stopping_patience=20`
on a 150-epoch schedule, expect ~15–18 hours per cache-size point, so ~50 hours
for the three-point sweep. Budget a GPU-day per cache size.

## Blocking issues before running

- None on the config side after Part C1.
- The M2 NAMM checkpoint referenced by `--namm_checkpoint` must be the
  FAIR-01-compliant one trained against the 5-task subset with `cache_size=1024`.
  Using a different M2 checkpoint silently changes the independent variable.
