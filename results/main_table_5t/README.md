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
| `A4/cs1024_no_namm` — A4 — M4 (cs1024) LoRA, NAMM disabled (full cache) | ✅ done |
| `A4/cs2048_no_namm` — A4 — M4 (cs2048) LoRA, NAMM disabled (full cache) | ✅ done |

Re-run `python scripts/organize_eval_results.py` to refresh after more jobs finish — the script is idempotent.

## Summary — micro and macro mean F1 across 5 tasks

**Micro** = prompt-count-weighted mean (matches LoRA training `val_lb_avg_f1`). **Macro** = unweighted mean over the 5 tasks (each task = 1/5). Plots in `plots/` use the micro average as the headline metric.

| Condition | test (micro) | test (macro) | extended_test (micro) | extended_test (macro) |
|-----------|-------------:|-------------:|----------------------:|----------------------:|
| `B0` | 19.28 | 20.02 | 21.09 | 21.72 |
| `B1/cs1024` | 11.96 | 11.62 | 8.70 | 8.84 |
| `B1/cs2048` | 13.26 | 13.10 | 10.84 | 11.22 |
| `M1` | 31.14 | 30.26 | 31.84 | 33.42 |
| `M2/cs1024` | 16.84 | 17.49 | 16.50 | 17.30 |
| `M2/cs2048` | 15.06 | 16.13 | 16.94 | 17.50 |
| `M4/cs1024` | 25.87 | 25.42 | 23.25 | 24.12 |
| `M4/cs2048` | 19.77 | 20.44 | 21.19 | 21.53 |
| `Trunc/plain_1024` | 14.40 | 14.73 | 15.56 | 15.85 |
| `Trunc/plain_2048` | 12.13 | 12.64 | 18.76 | 18.63 |
| `Trunc/lora_m1_1024` | 22.64 | 21.83 | 21.58 | 22.26 |
| `Trunc/lora_m1_2048` | 27.99 | 27.53 | 27.61 | 27.87 |
| `M1_recency/cs1024` | 0.00 | 0.00 | 0.00 | 0.00 |
| `M1_recency/cs2048` | — | — | — | — |
| `A4/cs1024_no_namm` | 30.58 | 30.61 | 27.25 | 28.37 |
| `A4/cs2048_no_namm` | 21.81 | 22.59 | 24.44 | 24.28 |

## Per-task F1 (test split)

| Condition | qasper | 2wikimqa | qasper_e | hotpotqa_e | 2wikimqa_e | macro | micro |
|---|---|---|---|---|---|---|---|
| `B0` | 9.19 | 20.71 | 11.61 | 28.31 | 30.27 | 20.02 | **19.28** |
| `B1/cs1024` | 15.70 | 7.94 | 15.16 | 10.15 | 9.12 | 11.62 | **11.96** |
| `B1/cs2048` | 16.09 | 7.80 | 14.41 | 15.55 | 11.66 | 13.10 | **13.26** |
| `M1` | 45.03 | 10.00 | 35.62 | 30.51 | 30.16 | 30.26 | **31.14** |
| `M2/cs1024` | 10.39 | 27.56 | 11.02 | 17.46 | 21.03 | 17.49 | **16.84** |
| `M2/cs2048` | 9.65 | 25.00 | 6.61 | 25.79 | 13.61 | 16.13 | **15.06** |
| `M4/cs1024` | 19.55 | 27.56 | 34.83 | 26.39 | 18.76 | 25.42 | **25.87** |
| `M4/cs2048` | 13.88 | 24.21 | 13.85 | 26.79 | 23.47 | 20.44 | **19.77** |
| `Trunc/plain_1024` | 10.68 | 27.96 | 12.38 | 8.33 | 14.29 | 14.73 | **14.40** |
| `Trunc/plain_2048` | 8.08 | 29.76 | 10.13 | 8.33 | 6.91 | 12.64 | **12.13** |
| `Trunc/lora_m1_1024` | 24.35 | 26.52 | 32.23 | 9.38 | 16.67 | 21.83 | **22.64** |
| `Trunc/lora_m1_2048` | 39.09 | 26.67 | 29.73 | 16.67 | 25.51 | 27.53 | **27.99** |
| `M1_recency/cs1024` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |
| `M1_recency/cs2048` | — | — | — | — | — | — | — |
| `A4/cs1024_no_namm` | 36.50 | 24.60 | 33.23 | 43.06 | 15.67 | 30.61 | **30.58** |
| `A4/cs2048_no_namm` | 17.32 | 24.60 | 15.08 | 32.94 | 23.02 | 22.59 | **21.81** |

## Per-task F1 (extended_test split)

| Condition | qasper | 2wikimqa | qasper_e | hotpotqa_e | 2wikimqa_e | macro | micro |
|---|---|---|---|---|---|---|---|
| `B0` | 11.54 | 23.10 | 12.60 | 40.50 | 20.84 | 21.72 | **21.09** |
| `B1/cs1024` | 13.61 | 7.37 | 10.02 | 5.05 | 8.17 | 8.84 | **8.70** |
| `B1/cs2048` | 12.93 | 7.65 | 11.60 | 13.06 | 10.87 | 11.22 | **10.84** |
| `M1` | 35.92 | 23.59 | 27.81 | 47.83 | 31.93 | 33.42 | **31.84** |
| `M2/cs1024` | 11.53 | 15.71 | 11.37 | 32.17 | 15.72 | 17.30 | **16.50** |
| `M2/cs2048` | 11.18 | 14.96 | 8.65 | 32.78 | 19.92 | 17.50 | **16.94** |
| `M4/cs1024` | 13.54 | 21.29 | 24.66 | 41.15 | 19.93 | 24.12 | **23.25** |
| `M4/cs2048` | 11.22 | 28.31 | 8.77 | 38.86 | 20.49 | 21.53 | **21.19** |
| `Trunc/plain_1024` | 10.80 | 17.43 | 13.00 | 23.89 | 14.11 | 15.85 | **15.56** |
| `Trunc/plain_2048` | 10.15 | 26.12 | 11.71 | 27.53 | 17.62 | 18.63 | **18.76** |
| `Trunc/lora_m1_1024` | 19.51 | 18.58 | 27.32 | 28.36 | 17.53 | 22.26 | **21.58** |
| `Trunc/lora_m1_2048` | 25.68 | 29.14 | 27.32 | 31.94 | 25.28 | 27.87 | **27.61** |
| `M1_recency/cs1024` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |
| `M1_recency/cs2048` | — | — | — | — | — | — | — |
| `A4/cs1024_no_namm` | 28.84 | 24.66 | 26.48 | 38.03 | 23.82 | 28.37 | **27.25** |
| `A4/cs2048_no_namm` | 15.47 | 30.28 | 12.40 | 36.08 | 27.15 | 24.28 | **24.44** |

