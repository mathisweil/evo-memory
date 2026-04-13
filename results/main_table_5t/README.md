# Main results table — 5-task QA eval (test + extended_test)

Eval runs against the 5-task LongBench QA subset used throughout this fork (`qasper`, `2wikimqa`, `qasper_e`, `hotpotqa_e`, `2wikimqa_e`).

Naming follows the milestones in `experiment_specification.md` (B0 baseline, B1 recency, M1 LoRA, M2 standalone NAMM, M4 LoRA + frozen NAMM, A4 ablation removing NAMM from M4).

All runs use:

- `train_frac=0.7`, `val_frac=0.15`, `split_seed=42`
- `min_conditioning_length=4096`, `max_conditioning_length=6500`, `max_answer_tokens=64`
- `extended_test`: filtered to length window `(6500, 8192]` (held-out long examples)
- Greedy decoding (`do_sample=False`)

Set sizes (post-filter): **70 test** + **224 extended_test** examples across 5 tasks.

## Layout

```
results/main_table_5t/
├── README.md                  ← this file
├── all_results.json           ← aggregated F1 across all conditions
│
├── B0/                        ← plain Llama, full cache
├── B1/cs{1024,2048}/          ← recency eviction baselines
├── M1/                        ← LoRA only (no NAMM)
├── M2/cs{1024,2048}/          ← standalone NAMM eviction
├── M4/cs{1024,2048}/          ← LoRA on frozen NAMM
└── A4/cs{1024,2048}_no_namm/  ← M4 LoRA, NAMM disabled (cs=8192)
```

Each leaf contains:
- `results.json` — raw eval output (full per-prompt F1 + per-task aggregate)
- `eviction_traces.npz` — symlink to per-prompt eviction signals (NAMM/recency only)
- `command.sh` — exact command used to produce the run
- `README.md` — condition description + parsed scores table

## Status

| Condition | Status |
|-----------|--------|
| `B0` — B0 — plain Llama, full KV cache | ✅ done |
| `B1/cs1024` — B1 — recency eviction, cache_size=1024 | ✅ done |
| `B1/cs2048` — B1 — recency eviction, cache_size=2048 | ✅ done |
| `M1` — M1 — LoRA only (no NAMM) | ✅ done |
| `M2/cs1024` — M2 — standalone NAMM, cache_size=1024 | ✅ done |
| `M2/cs2048` — M2 — standalone NAMM, cache_size=2048 (friend's checkpoint) | ✅ done |
| `M4/cs1024` — M4 — LoRA on frozen NAMM, cache_size=1024 | ✅ done |
| `M4/cs2048` — M4 — LoRA on frozen NAMM, cache_size=2048 | ✅ done |
| `Trunc/plain_1024` — Plain Llama, input truncated to last 1024 tokens | ✅ done |
| `Trunc/plain_2048` — Plain Llama, input truncated to last 2048 tokens | ✅ done |
| `Trunc/lora_m1_1024` — M1 LoRA, input truncated to last 1024 tokens | ✅ done |
| `Trunc/lora_m1_2048` — M1 LoRA, input truncated to last 2048 tokens | ✅ done |
| `M1_recency/cs1024` — M1 LoRA + recency eviction, cache_size=1024 | ✅ done |
| `M1_recency/cs2048` — M1 LoRA + recency eviction, cache_size=2048 | ⏳ pending |
| `M1_under_NAMM/cs1024` — M1 LoRA (no NAMM training) + NAMM eviction cs1024 | ✅ done |
| `M1_under_NAMM/cs2048` — M1 LoRA (no NAMM training) + NAMM eviction cs2048 | ✅ done |
| `A4/cs1024_no_namm` — A4 — M4 (cs1024) LoRA, NAMM disabled (full cache) | ✅ done |
| `A4/cs2048_no_namm` — A4 — M4 (cs2048) LoRA, NAMM disabled (full cache) | ✅ done |

Re-run `python scripts/organize_eval_results.py` to refresh after more jobs finish — the script is idempotent.

## Summary — micro and macro mean F1 across 5 tasks

**Micro** = prompt-count-weighted mean (matches LoRA training `val_lb_avg_f1`). **Macro** = unweighted mean over the 5 tasks (each task = 1/5). Plots in `plots/` use the micro average as the headline metric.

| Condition | test (micro) | test (macro) | extended_test (micro) | extended_test (macro) |
|-----------|-------------:|-------------:|----------------------:|----------------------:|
| `B0` | 22.41 | 24.09 | 22.30 | 23.69 |
| `B1/cs1024` | 11.33 | 11.52 | 6.93 | 7.99 |
| `B1/cs2048` | 11.10 | 11.23 | 9.30 | 10.39 |
| `M1` | 31.14 | 30.26 | 31.84 | 33.42 |
| `M2/cs1024` | 10.83 | 11.15 | 18.79 | 18.70 |
| `M2/cs2048` | 15.27 | 15.91 | 19.19 | 19.50 |
| `M4/cs1024` | 23.52 | 24.30 | 25.40 | 26.23 |
| `M4/cs2048` | 31.41 | 31.43 | 23.40 | 23.51 |
| `Trunc/plain_1024` | 18.21 | 18.44 | 17.83 | 18.78 |
| `Trunc/plain_2048` | 18.26 | 18.76 | 19.35 | 20.27 |
| `Trunc/lora_m1_1024` | 26.90 | 27.08 | 24.24 | 25.71 |
| `Trunc/lora_m1_2048` | 28.87 | 28.91 | 27.67 | 29.34 |
| `M1_recency/cs1024` | 0.00 | 0.00 | 0.00 | 0.00 |
| `M1_recency/cs2048` | — | — | — | — |
| `M1_under_NAMM/cs1024` | 26.97 | 27.20 | 21.83 | 22.54 |
| `M1_under_NAMM/cs2048` | 31.71 | 31.86 | 24.96 | 25.55 |
| `A4/cs1024_no_namm` | 28.82 | 28.69 | 25.62 | 26.40 |
| `A4/cs2048_no_namm` | 38.98 | 38.83 | 30.61 | 31.18 |

## Per-task F1 (test split)

| Condition | qasper | 2wikimqa | qasper_e | hotpotqa_e | 2wikimqa_e | macro | micro |
|---|---|---|---|---|---|---|---|
| `B0` | 25.85 | 26.52 | 6.06 | 44.56 | 17.46 | 24.09 | **22.41** |
| `B1/cs1024` | 22.77 | 10.42 | 7.26 | 10.60 | 6.55 | 11.52 | **11.33** |
| `B1/cs2048` | 23.46 | 7.63 | 6.14 | 9.26 | 9.68 | 11.23 | **11.10** |
| `M1` | 45.03 | 10.00 | 35.62 | 30.51 | 30.16 | 30.26 | **31.14** |
| `M2/cs1024` | 7.67 | 19.23 | 9.03 | 9.96 | 9.87 | 11.15 | **10.83** |
| `M2/cs2048` | 19.81 | 25.00 | 7.30 | 12.28 | 15.18 | 15.91 | **15.27** |
| `M4/cs1024` | 7.82 | 44.23 | 21.91 | 26.79 | 20.75 | 24.30 | **23.52** |
| `M4/cs2048` | 35.74 | 25.00 | 29.86 | 35.12 | 31.41 | 31.43 | **31.41** |
| `Trunc/plain_1024` | 29.80 | 26.52 | 13.99 | 9.38 | 12.50 | 18.44 | **18.21** |
| `Trunc/plain_2048` | 24.81 | 25.00 | 12.42 | 17.28 | 14.29 | 18.76 | **18.26** |
| `Trunc/lora_m1_1024` | 26.35 | 26.52 | 27.20 | 33.89 | 21.43 | 27.08 | **26.90** |
| `Trunc/lora_m1_2048` | 31.56 | 27.56 | 30.04 | 33.95 | 21.43 | 28.91 | **28.87** |
| `M1_recency/cs1024` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |
| `M1_recency/cs2048` | — | — | — | — | — | — | — |
| `M1_under_NAMM/cs1024` | 19.42 | 27.56 | 27.48 | 35.33 | 26.19 | 27.20 | **26.97** |
| `M1_under_NAMM/cs2048` | 35.34 | 27.56 | 28.58 | 35.19 | 32.65 | 31.86 | **31.71** |
| `A4/cs1024_no_namm` | 46.19 | 25.00 | 28.12 | 26.67 | 17.46 | 28.69 | **28.82** |
| `A4/cs2048_no_namm` | 50.22 | 33.33 | 39.16 | 39.68 | 31.75 | 38.83 | **38.98** |

## Per-task F1 (extended_test split)

| Condition | qasper | 2wikimqa | qasper_e | hotpotqa_e | 2wikimqa_e | macro | micro |
|---|---|---|---|---|---|---|---|
| `B0` | 18.34 | 17.86 | 13.11 | 45.88 | 23.27 | 23.69 | **22.30** |
| `B1/cs1024` | 12.23 | 4.42 | 4.80 | 14.24 | 4.27 | 7.99 | **6.93** |
| `B1/cs2048` | 13.79 | 4.22 | 7.36 | 18.24 | 8.35 | 10.39 | **9.30** |
| `M1` | 35.92 | 23.59 | 27.81 | 47.83 | 31.93 | 33.42 | **31.84** |
| `M2/cs1024` | 12.70 | 19.01 | 13.38 | 26.36 | 22.05 | 18.70 | **18.79** |
| `M2/cs2048` | 20.67 | 21.46 | 10.62 | 24.71 | 20.06 | 19.50 | **19.19** |
| `M4/cs1024` | 24.46 | 21.97 | 26.75 | 34.83 | 23.11 | 26.23 | **25.40** |
| `M4/cs2048` | 21.42 | 22.67 | 22.52 | 26.99 | 23.93 | 23.51 | **23.40** |
| `Trunc/plain_1024` | 23.36 | 18.95 | 14.55 | 23.52 | 13.52 | 18.78 | **17.83** |
| `Trunc/plain_2048` | 19.97 | 17.23 | 12.72 | 31.91 | 19.54 | 20.27 | **19.35** |
| `Trunc/lora_m1_1024` | 25.50 | 18.58 | 28.15 | 37.31 | 18.99 | 25.71 | **24.24** |
| `Trunc/lora_m1_2048` | 30.62 | 18.72 | 31.68 | 41.85 | 23.83 | 29.34 | **27.67** |
| `M1_recency/cs1024` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |
| `M1_recency/cs2048` | — | — | — | — | — | — | — |
| `M1_under_NAMM/cs1024` | 17.52 | 19.24 | 22.28 | 33.57 | 20.09 | 22.54 | **21.83** |
| `M1_under_NAMM/cs2048` | 24.26 | 20.73 | 21.52 | 34.32 | 26.92 | 25.55 | **24.96** |
| `A4/cs1024_no_namm` | 30.41 | 23.55 | 21.41 | 31.57 | 25.07 | 26.40 | **25.62** |
| `A4/cs2048_no_namm` | 32.09 | 32.66 | 28.29 | 35.66 | 27.20 | 31.18 | **30.61** |

