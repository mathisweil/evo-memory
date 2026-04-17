# 09 — Distribution Drift

## What this does

Measures how much the output distribution p(x_{t+1} | x_{1:t}) changes when
NAMM eviction is toggled on or off, token-by-token during greedy generation.

### Two pairs compared

| Pair | Condition A | Condition B | Question |
|------|-------------|-------------|----------|
| 1 | M1 full cache | M1 + post-hoc NAMM | How much does adding NAMM disrupt a model that never trained with eviction? |
| 2 | M4 + NAMM (in-loop) | A4 full cache (M4 weights, no NAMM) | How much does removing NAMM disrupt a model that trained under eviction? |

### Metric

Mean per-prompt JS divergence between the two conditions' next-token
distributions, averaged across generation steps.

## Post-hoc filter

Drops prompts where any condition's greedy generation exceeds
`max(5, 2 × shortest_gold_answer_token_count)` — these are repetition /
hallucination loops rather than real answers. Retention on the test split
is 19/70 prompts (qasper_e loses all 18).

## Run

Generate drift data (GPU, ~40 min on 3090 Ti for 70 test prompts):

```bash
python analyze_distribution_drift.py \
    --m1_lora_checkpoint experiment_artifacts/gcs/final_cs1024/m1_lora_matched.pt \
    --m4_lora_checkpoint experiment_artifacts/gcs/final_cs1024/m4_lora_namm.pt \
    --namm_checkpoint    experiment_artifacts/gcs/final_cs1024/namm_cs1024_maskfix.pt \
    --cache_size 1024 --split test \
    --output_dir analysis_out/latest_analysis_matched/09_distribution_drift
```

Filter and produce the JS bar plot (CPU only, seconds):

```bash
python filter_and_replot.py --split test
```

## Outputs

- `drift_results.json` — per-prompt, per-step JS divergence + top-k + token match
- `drift_results_filtered.json` — same after the length-filter
- `js_divergence_bar.png` — mean JS divergence per pair (filtered n)
