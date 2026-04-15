# 01 — LoRA Weight Divergence

## What this does

Loads two LoRA checkpoints (M1-matched and M4), extracts the effective weight update
`delta_W = B @ A` for each LoRA adapter (q_proj and v_proj at each of the 16 layers),
and compares them.

## Metrics computed per layer, per module (q_proj / v_proj)

- **Frobenius distance** `||dW_M1 - dW_M4||_F`: how different are the learned updates?
  Large distance = the two models adapted the same layer in very different ways.
- **Cosine similarity** `cos(dW_M1, dW_M4)`: do the updates point in the same direction in
  weight space? 1 = same direction (just different magnitude), 0 = orthogonal (completely
  different adaptations), negative = opposing.
- **Norm ratio** `||dW_M4|| / ||dW_M1||`: did M4 learn larger or smaller updates than M1?
  Ratio > 1 means M4 has larger updates at that layer.

## Interpretation
- Cosine near 0 everywhere → the two models learned fundamentally different adaptations,
  not variations of the same thing.
- Norm ratio >> 1 → M4 trained harder (larger weight changes). Since both used the same lr,
  this means the M4 loss landscape (with NAMM eviction) drove stronger gradient signals.

## Command
```bash
python scripts/analyze_lora_divergence.py \
    --ckpt_a checkpoints_backup/lora_m1_lr1e4_matched/best_ckpt.pt \
    --ckpt_b checkpoints_backup/lora_m4_cs1024_maskfix/best_ckpt_step260_val52.06.pt \
    --label_a "M1-matched (lr=1e-4)" \
    --label_b "M4 (NAMM in-loop)" \
    --output_dir analysis_out/latest_analysis_matched/01_lora_weight_divergence
```

## No GPU needed — just loads .pt files (~2 seconds)
