# 06 — Eviction Mask Drift

## What this does

Tests whether LoRA fine-tuning changes NAMM's eviction decisions even with frozen NAMM
weights. Since NAMM scores tokens using the LLM's **attention activations**, changing the
LLM parameters (via LoRA) changes the attention patterns, which changes the scores, which
changes which tokens get evicted.

Runs THREE conditions under the same frozen NAMM:
1. **Base** (LoRA weights zeroed) — what the pre-trained model evicts
2. **M1-matched** — what the full-cache-trained LoRA evicts
3. **M4** — what the NAMM-in-loop-trained LoRA evicts

Compares eviction decisions (union over heads) via Jaccard per layer.

## Metrics computed per layer, per pair

- **Jaccard similarity**: overlap of kept-token sets (union over heads).
  1.0 = identical eviction decisions. 0.5 = half the tokens differ.
- **Drift fraction**: `(|A_only| + |B_only|) / |A ∪ B|` — fraction of tokens that
  changed eviction status between the two conditions.

## Interpretation
- High Base→M1 Jaccard (>0.9) + low Base→M4 Jaccard (<0.8) → M4's LoRA actively
  reshaped attention patterns that NAMM uses for scoring. M1's LoRA barely affected them.
- Low Jaccard across all pairs → ANY LoRA fine-tuning substantially changes eviction
  decisions. This validates the coupled optimization concern from the paper: fine-tuning
  the LLM and freezing the eviction policy is not truly "freezing" the effective eviction
  behavior.

## Command
```bash
python scripts/analyze_eviction_mask_drift.py \
    --m1_lora_checkpoint checkpoints_backup/lora_m1_lr1e4_matched/best_ckpt.pt \
    --m4_lora_checkpoint checkpoints_backup/lora_m4_cs1024_maskfix/best_ckpt_step260_val52.06.pt \
    --namm_checkpoint eval_results/namm_cs1024_maskfix/ckpt.pt \
    --cache_size 1024 --split test \
    --output_dir analysis_out/latest_analysis_matched/06_eviction_mask_drift
```

## Requires GPU (~30 min, 3 conditions × 70 prompts)
