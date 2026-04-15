# 03 — Deep Score & Attention Analysis v2

## What this does

Runs both M1-matched and M4 under the **same frozen NAMM policy** (cs1024 maskfix
checkpoint) on all 70 test prompts. Captures per-layer:
- Raw NAMM token scores (the scoring network's output, before top-k selection)
- Which tokens survived eviction (retained_idxs)
- Absolute positions of surviving tokens (cache_position_ids)
- Full attention weight matrices

Then compares the two models on multiple dimensions.

## Important: per-head vs union

Llama 3.2-1B has **8 KV heads** per layer. NAMM eviction is **per-head**: each head
independently selects its top `cache_size=1024` tokens. Different heads can keep
different tokens.

- **Per-head metrics**: compare head h of M1 vs head h of M4 directly. Each keeps
  exactly 1024 tokens. Per-head Jaccard is the clean "same budget, same head" comparison.
- **Union metrics**: take the union of all 8 heads' kept positions per model. This gives
  "all positions accessible to at least one head" — can be up to 8×1024 positions
  (in practice ~2000 due to overlap between heads).

## Metrics computed

### A. Kept-set decomposition
- `n_both_kept` (union): positions kept by both M1 and M4 (across any head)
- `n_m1_only` / `n_m4_only`: positions kept by one model but not the other
- `jaccard` (union): Jaccard similarity of the union sets
- `per_head_jaccard`: Jaccard between M1 head h and M4 head h (8 values per layer)
- `per_head_n_kept_m1/m4`: tokens kept by each head (should be ≈1024)

### B. NAMM score analysis
NAMM scores are the scoring network's output *before* top-k selection. Higher score = more
likely to be kept. Averaged across heads for position-level analysis.
- `m1/m4_score_gap`: mean(kept scores) − mean(evicted scores). Larger = cleaner separation.
- `m1/m4_kept_score_std`: standard deviation of scores among kept tokens. Lower = NAMM is
  more confident about its selections.
- `m1/m4_all_score_std`: std of ALL scores (kept + evicted). Context for the above.
- `m1/m4_topk_mean` / `bottomk_mean`: mean score of the top-1024 vs bottom scores.

### C. Score rank correlation
- `spearman_r`: Spearman rank correlation of token scores between M1 and M4. High (>0.8) =
  both models produce similar token importance rankings. Low = they disagree on what matters.
  Only computed when both score vectors have the same length (same cache state).

### D. Per-token attention on shared kept tokens
For positions kept by BOTH models: extracts each model's attention weight on that position
(last query token → shared kept position), normalizes over the shared set, then compares:
- `shared_attn_mad_mean`: mean absolute difference of normalized attention. 0 = identical.
- `shared_attn_corr_mean`: Pearson correlation. 1 = same attention pattern on shared tokens.

### E. Positional analysis
For M1-only and M4-only tokens: what part of the prompt are they from?
- `first_third` / `middle_third` / `last_third`: count of tokens from each region.

### F. Score histograms
Overlaid histograms of kept (green) vs evicted (red) NAMM scores at selected layers (0, 4, 8, 12, 15).

## Command
```bash
python scripts/analyze_attention_divergence_v2.py \
    --m1_lora_checkpoint checkpoints_backup/lora_m1_lr1e4_matched/best_ckpt.pt \
    --m4_lora_checkpoint checkpoints_backup/lora_m4_cs1024_maskfix/best_ckpt_step260_val52.06.pt \
    --namm_checkpoint eval_results/namm_cs1024_maskfix/ckpt.pt \
    --cache_size 1024 --split test \
    --output_dir analysis_out/latest_analysis_matched/03_attention_score_analysis_v2
```

## Requires GPU (~30 min on 3090 Ti, 2×70 forward passes)

## Outputs
- `kept_decomposition.png` — stacked bar (both/M1-only/M4-only) + union Jaccard per layer
- `score_gap_concentration.png` — score gap, score std, top-k vs bottom scores
- `rank_corr_attention.png` — Spearman correlation, per-token attention MAD and correlation
- `positional_analysis.png` — where in the prompt are M1-only vs M4-only tokens
- `score_histograms_v2.png` — kept vs evicted score distributions at selected layers
- `scores_by_category.png` — kept vs evicted mean scores per model
- `per_head_analysis.png` — per-head Jaccard vs union Jaccard, per-head kept counts, head spread
- `deep_analysis.json` — full per-prompt per-layer metrics
