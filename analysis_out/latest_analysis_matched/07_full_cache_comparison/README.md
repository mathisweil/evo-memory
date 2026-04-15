# 07 — Full Cache Comparison (M1 vs M1+NAMM vs A4)

## What this does

Compares three conditions on the same 70 test prompts:
1. **M1-matched, full cache** — LoRA trained without NAMM, evaluated without NAMM
2. **M1-matched, NAMM cs1024** — same LoRA, evaluated with NAMM eviction
3. **A4 = M4 weights, full cache** — LoRA trained WITH NAMM, evaluated WITHOUT NAMM

This isolates the effect of training under eviction on the model's behavior
when eviction is NOT active at eval time.

## Key question
Did training under eviction (M4/A4) make the model more robust, or did it learn
a fundamentally different strategy that happens to work better?

## Metrics computed per layer

### Hidden states (all 3 pairs)
- L2 distance and cosine similarity of last-token hidden states
- Answers: how different are the internal representations?

### Attention (M1 full vs A4 full only — both under full cache)
- **Entropy**: is A4 more focused? Lower entropy = fewer tokens getting attention mass.
- **Pearson correlation**: do they attend to the same positions?
- **Mean absolute difference**: magnitude of attention pattern divergence.
- **Positional mass**: fraction of attention on first/middle/last third of prompt.

## Command
```bash
python scripts/analyze_full_cache_comparison.py \
    --m1_lora_checkpoint checkpoints_backup/lora_m1_lr1e4_matched/best_ckpt.pt \
    --m4_lora_checkpoint checkpoints_backup/lora_m4_cs1024_maskfix/best_ckpt_step260_val52.06.pt \
    --namm_checkpoint eval_results/namm_cs1024_maskfix/ckpt.pt \
    --cache_size 1024 --split test \
    --output_dir analysis_out/latest_analysis_matched/07_full_cache_comparison
```

## Requires GPU (~25 min, 3 conditions × 70 prompts)

## Outputs
- `hidden_states.png` — L2 + cosine between all 3 pairs per layer
- `attention_analysis.png` — entropy, correlation, MAD (M1 vs A4 under full cache)
- `positional_attention.png` — attention mass on first/middle/last third per condition
- `l2_ratio.png` — per layer: is LoRA adaptation or NAMM eviction a bigger change?
- `full_cache_comparison.json` — aggregate metrics
