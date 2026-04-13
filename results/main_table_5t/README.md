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
| `B1/cs1024` | 12.45 | 12.83 | 7.60 | 8.81 |
| `B1/cs2048` | 13.78 | 14.39 | 10.28 | 11.74 |
| `M1` | 31.14 | 30.26 | 31.84 | 33.42 |
| `M2/cs1024` | 20.30 | 21.12 | 20.65 | 20.75 |
| `M2/cs2048` | 17.40 | 18.30 | 19.01 | 19.29 |
| `M4/cs1024` | 32.28 | 33.27 | 26.92 | 28.03 |
| `M4/cs2048` | 31.06 | 31.05 | 23.15 | 23.36 |
| `Trunc/plain_1024` | 18.21 | 18.44 | 17.83 | 18.78 |
| `Trunc/plain_2048` | 18.26 | 18.76 | 19.35 | 20.27 |
| `Trunc/lora_m1_1024` | 26.90 | 27.08 | 24.24 | 25.71 |
| `Trunc/lora_m1_2048` | 28.87 | 28.91 | 27.67 | 29.34 |
| `M1_recency/cs1024` | 0.00 | 0.00 | 0.00 | 0.00 |
| `M1_recency/cs2048` | — | — | — | — |
| `M1_under_NAMM/cs1024` | 26.97 | 27.20 | 21.83 | 22.54 |
| `M1_under_NAMM/cs2048` | 31.35 | 31.59 | 24.50 | 25.29 |
| `A4/cs1024_no_namm` | 28.82 | 28.69 | 25.62 | 26.40 |
| `A4/cs2048_no_namm` | 33.91 | 34.07 | 25.66 | 25.96 |

## Per-task F1 (test split)

| Condition | qasper | 2wikimqa | qasper_e | hotpotqa_e | 2wikimqa_e | macro | micro |
|---|---|---|---|---|---|---|---|
| `B0` | 25.85 | 26.52 | 6.06 | 44.56 | 17.46 | 24.09 | **22.41** |
| `B1/cs1024` | 22.29 | 10.42 | 7.26 | 17.65 | 6.55 | 12.83 | **12.45** |
| `B1/cs2048` | 23.32 | 7.63 | 6.14 | 25.93 | 8.93 | 14.39 | **13.78** |
| `M1` | 45.03 | 10.00 | 35.62 | 30.51 | 30.16 | 30.26 | **31.14** |
| `M2/cs1024` | 28.30 | 27.56 | 8.09 | 17.50 | 24.16 | 21.12 | **20.30** |
| `M2/cs2048` | 26.79 | 25.00 | 6.06 | 18.45 | 15.18 | 18.30 | **17.40** |
| `M4/cs1024` | 29.30 | 44.23 | 26.56 | 43.45 | 22.79 | 33.27 | **32.28** |
| `M4/cs2048` | 39.68 | 25.00 | 30.47 | 35.51 | 24.60 | 31.05 | **31.06** |
| `Trunc/plain_1024` | 29.80 | 26.52 | 13.99 | 9.38 | 12.50 | 18.44 | **18.21** |
| `Trunc/plain_2048` | 24.81 | 25.00 | 12.42 | 17.28 | 14.29 | 18.76 | **18.26** |
| `Trunc/lora_m1_1024` | 26.35 | 26.52 | 27.20 | 33.89 | 21.43 | 27.08 | **26.90** |
| `Trunc/lora_m1_2048` | 31.56 | 27.56 | 30.04 | 33.95 | 21.43 | 28.91 | **28.87** |
| `M1_recency/cs1024` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |
| `M1_recency/cs2048` | — | — | — | — | — | — | — |
| `M1_under_NAMM/cs1024` | 19.42 | 27.56 | 27.48 | 35.33 | 26.19 | 27.20 | **26.97** |
| `M1_under_NAMM/cs2048` | 35.34 | 27.56 | 27.20 | 35.19 | 32.65 | 31.59 | **31.35** |
| `A4/cs1024_no_namm` | 46.19 | 25.00 | 28.12 | 26.67 | 17.46 | 28.69 | **28.82** |
| `A4/cs2048_no_namm` | 43.56 | 38.89 | 34.63 | 35.80 | 17.46 | 34.07 | **33.91** |

## Per-task F1 (extended_test split)

| Condition | qasper | 2wikimqa | qasper_e | hotpotqa_e | 2wikimqa_e | macro | micro |
|---|---|---|---|---|---|---|---|
| `B0` | 18.34 | 17.86 | 13.11 | 45.88 | 23.27 | 23.69 | **22.30** |
| `B1/cs1024` | 15.36 | 5.35 | 4.80 | 14.24 | 4.27 | 8.81 | **7.60** |
| `B1/cs2048` | 16.84 | 4.40 | 7.66 | 21.47 | 8.32 | 11.74 | **10.28** |
| `M1` | 35.92 | 23.59 | 27.81 | 47.83 | 31.93 | 33.42 | **31.84** |
| `M2/cs1024` | 19.49 | 23.74 | 12.13 | 26.36 | 22.05 | 20.75 | **20.65** |
| `M2/cs2048` | 20.20 | 20.90 | 10.14 | 24.71 | 20.49 | 19.29 | **19.01** |
| `M4/cs1024` | 25.39 | 27.27 | 23.78 | 41.21 | 22.51 | 28.03 | **26.92** |
| `M4/cs2048` | 23.41 | 22.00 | 14.59 | 29.55 | 27.24 | 23.36 | **23.15** |
| `Trunc/plain_1024` | 23.36 | 18.95 | 14.55 | 23.52 | 13.52 | 18.78 | **17.83** |
| `Trunc/plain_2048` | 19.97 | 17.23 | 12.72 | 31.91 | 19.54 | 20.27 | **19.35** |
| `Trunc/lora_m1_1024` | 25.50 | 18.58 | 28.15 | 37.31 | 18.99 | 25.71 | **24.24** |
| `Trunc/lora_m1_2048` | 30.62 | 18.72 | 31.68 | 41.85 | 23.83 | 29.34 | **27.67** |
| `M1_recency/cs1024` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |
| `M1_recency/cs2048` | — | — | — | — | — | — | — |
| `M1_under_NAMM/cs1024` | 17.52 | 19.24 | 22.28 | 33.57 | 20.09 | 22.54 | **21.83** |
| `M1_under_NAMM/cs2048` | 23.74 | 20.10 | 21.60 | 35.75 | 25.28 | 25.29 | **24.50** |
| `A4/cs1024_no_namm` | 30.41 | 23.55 | 21.41 | 31.57 | 25.07 | 26.40 | **25.62** |
| `A4/cs2048_no_namm` | 24.54 | 31.15 | 22.88 | 29.92 | 21.31 | 25.96 | **25.66** |

