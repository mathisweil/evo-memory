# Analysis Results — Matched Hyperparameters (April 15, 2026)

All analyses at cs1024 comparing **M1-matched** (LoRA lr=1e-4, dropout=0.05, full cache) vs **M4** (LoRA+NAMM in-loop, same lr/dropout). Hyperparameters are now identical — the only experimental variable is whether NAMM was active during training.

## M1-matched checkpoint
- Path: `checkpoints_backup/lora_m1_lr1e4_matched/best_ckpt.pt`
- Training: `lr=1e-4, dropout=0.05, rank=8, [q_proj,v_proj], 150 epochs`
- Best step: 64, val F1: 38.35
- Note: lr=1e-4 is TOO HIGH for full-cache LoRA — M1-matched only reaches 13.8 test F1 (vs 28.0 for old M1 at lr=5e-5). This itself is a finding: NAMM eviction regularizes the higher lr.

## F1 Results (cs1024, matched hyperparams)

| Condition | Test F1 | Extended F1 |
|---|:-:|:-:|
| M1-matched (recency, no NAMM) | 13.8 | 10.2 |
| M1-matched under NAMM (post-hoc) | 19.0 | 22.0 |
| M4 LoRA+NAMM (NAMM in-loop) | 33.5 | 25.8 |
| **M4 − M1-matched under NAMM** | **+14.5** | **+3.8** |

The adaptation gain under matched hyperparams is **+14.5 F1** (vs +6.2 with old mismatched M1). The effect of training under eviction is even larger than previously estimated.

## 01 — LoRA Weight Divergence
- M4 has **4–10× larger** update norms than M1-matched (vs 1.1-1.8× with old M1)
- Cosine similarity still near zero — orthogonal subspaces
- M1-matched learned very little (small norms → low F1), M4 learned a lot (large norms → high F1)
- Interpretation: lr=1e-4 is too aggressive for full-cache; NAMM eviction constrains the optimization landscape enough to make it productive

## 03 — Deep Score & Attention Analysis
- Kept-token Jaccard: 0.75–0.99, similar to previous (L0 near-identical, deeper layers diverge)
- Score gap comparable between models
- Attention on shared tokens still near-identical (corr >0.96)
- M4 retains more unique tokens at most layers

## 05 — Hidden State Drift
- M4 has **23.5% MORE drift** than M1-matched (up from 8.6% with old M1)
- Early layers: M4 drifts ~2× more
- L5 and L10: M4 actually drifts LESS (arrows ←)
- Stronger evidence for specialization over robustness

## 06 — Eviction Mask Drift
- **Base→M1-matched: Jaccard=0.951** (only 4.9% drift) — M1-matched barely changes attention patterns
- **Base→M4: Jaccard=0.805** (19.5% drift) — M4 actively reshapes them
- Dramatic contrast: M1-matched didn't learn enough to change eviction decisions (consistent with its low F1), while M4 learned enough to change ~20% of eviction decisions AND achieve high F1

## 07 — Full Cache Comparison (M1 vs M1+NAMM vs A4)
**Script**: `scripts/analyze_full_cache_comparison.py`
**Question**: What did training under eviction change about how the model processes full-cache input?

**Findings**:
- M1→A4 hidden state distance (7.7) is **2× larger** than M1→M1+NAMM (4.0) — changing LoRA weights matters more than applying eviction
- Attention entropy is near-identical (M1: 2.46, A4: 2.43) — A4 is NOT more "focused"
- Attention correlation = 0.91 — similar but meaningfully different patterns
- **Positional attention shift**: A4 attends **more to the first third** of the prompt (0.526 vs 0.484) and **less to the last third** (0.432 vs 0.482). Training under eviction taught the model to extract information from early context more aggressively, and this persists under full cache.

## Not included in this folder
- **Generation entropy**: OOM on 24GB GPUs, pending fix
- **Cost analysis / cs2048**: not requested for matched analysis
