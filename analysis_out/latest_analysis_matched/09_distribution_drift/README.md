# 09 — Distribution Drift

## What this does

Measures how much the output distribution p(x_{t+1} | x_{1:t}) changes when
NAMM eviction is added or removed, token-by-token during greedy generation.

### Two pairs compared

| Pair | Condition A | Condition B | Question |
|------|-------------|-------------|----------|
| 1 | M1 full cache | M1 + post-hoc NAMM | How much does adding NAMM disrupt a model that never trained with eviction? |
| 2 | M4 + NAMM (in-loop) | A4 full cache (M4 weights, no NAMM) | How much does removing NAMM disrupt a model that trained under eviction? |

### Metrics per generated token

- **JS divergence**: Jensen-Shannon divergence of the full softmax distribution
- **Top-k overlap** (k=5, k=10): fraction of highest-probability token IDs shared
- **Token match rate**: fraction of greedy-decoded tokens that are identical

### Interpretation

If M4↔A4 (pair 2) has:
- Lower F1 drop AND lower JS drift → **robustness** (M4 learned to be NAMM-invariant)
- Lower F1 drop BUT similar/higher JS drift → **specialization** (the distribution
  changes, but the changed behavior is better suited to the task)

## Command

```bash
python analysis_out/latest_analysis_matched/09_distribution_drift/analyze_distribution_drift.py \
    --m1_lora_checkpoint experiment_artifacts/gcs/final_cs1024/m1_lora_matched.pt \
    --m4_lora_checkpoint experiment_artifacts/gcs/final_cs1024/m4_lora_namm.pt \
    --namm_checkpoint experiment_artifacts/gcs/final_cs1024/namm_cs1024_maskfix.pt \
    --cache_size 1024 --split test --prompts_per_task 2 \
    --eval_json_dir analysis_out/latest_analysis_matched/00_eval_results \
    --output_dir analysis_out/latest_analysis_matched/09_distribution_drift
```

## Requires GPU (~6 min on 3090 Ti, 4 conditions × 10 prompts)

## Outputs

- `drift_results.json` — per-prompt, per-step JS divergence + top-k overlap + token match
- `js_divergence_by_pair.png` — mean JS divergence per generation step (both pairs)
- `topk5_overlap_by_pair.png` — mean top-5 overlap per generation step
- `topk10_overlap_by_pair.png` — mean top-10 overlap per generation step
- `aggregate_summary.png` — bar charts: mean JS, token match rate, top-5 overlap
- `drift_vs_f1.png` — scatter: per-prompt mean JS vs F1 change (if eval JSONs provided)
