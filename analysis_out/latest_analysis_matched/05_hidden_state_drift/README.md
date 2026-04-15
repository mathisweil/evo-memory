# 05 — Hidden State Drift

## What this does

For each model (M1-matched, M4), runs the SAME 70 test prompts under TWO conditions:
1. Full cache (`apply_memory_policy=False`) — no eviction
2. NAMM active (`apply_memory_policy=True`) — with cs1024 eviction

Captures the **last token's hidden state** at every layer (embedding output through
all 16 transformer layers + LM head = 17 states). Computes per-layer drift between
the two conditions.

## Metrics computed per layer

- **L2 distance** `||h_full - h_namm||_2`: magnitude of the hidden state change caused by
  eviction. Larger = eviction perturbs internal representations more at this layer.
- **Cosine similarity** `cos(h_full, h_namm)`: direction agreement. 1 = same direction
  (eviction only scales the representation), < 1 = eviction rotates it.
- **Relative drift** `||h_full - h_namm|| / ||h_full||`: drift as a fraction of the
  hidden state magnitude. Controls for layers with larger activations.

## Interpretation
- If M4 has LESS drift than M1 → training under eviction made representations **robust**
  to pruning (the model learned cache-invariant features).
- If M4 has MORE drift → training under eviction **specialized** the representations
  for the evicted cache (they work well with NAMM but diverge more when full cache is restored).

## Command
```bash
python scripts/analyze_hidden_state_drift.py \
    --m1_lora_checkpoint checkpoints_backup/lora_m1_lr1e4_matched/best_ckpt.pt \
    --m4_lora_checkpoint checkpoints_backup/lora_m4_cs1024_maskfix/best_ckpt_step260_val52.06.pt \
    --namm_checkpoint eval_results/namm_cs1024_maskfix/ckpt.pt \
    --cache_size 1024 --split test \
    --output_dir analysis_out/latest_analysis_matched/05_hidden_state_drift
```

## Requires GPU (~20 min, 4 forward passes per prompt: 2 models × 2 conditions)
