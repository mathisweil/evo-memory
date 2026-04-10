# Analysis 6 — Token Importance Alignment (NAMM Scores vs Attention)

> **TL;DR:** NAMM token importance scores are **negatively correlated** with the LLM's attention weights (mean Spearman rho = -0.14), meaning NAMM systematically assigns low scores to tokens the model attends to. M1 and M3 show nearly identical alignment (rho = -0.137 vs -0.136), indicating M3 fine-tuning does **not** reshape attention to agree with NAMM's eviction decisions. Eviction regret is low overall (~7% of attention mass on evicted tokens) but peaks in early layers 0-3 where attention is broad. The negative correlation suggests NAMM operates on a complementary signal to attention — it uses spectrogram-based embeddings (temporal patterns in KV cache) that capture a different notion of token importance than instantaneous attention.

## Methodology

### Setup

- **Checkpoints:** M1 (LoRA full-context, best val F1 45.48) and M3 cs1024 (LoRA + frozen NAMM, best val F1 45.59), both using the same NAMM checkpoint from M2 cs1024 training.
- **Test data:** 3 samples per task from the 5-task test set (15 samples total).
- **Cache size:** 1024 tokens.

### Score Extraction

For each test sample, the script performs a two-pass approach:

1. **Pass 1 (full context):** Forward pass with `apply_memory_policy=False` and `output_attentions=True`. This yields full-context attention weights and the KV cache before any eviction.

2. **Pass 2 (NAMM analyze):** Calls `memory_policy.update_cache(analyze=True)` on the full KV cache. This invokes the NAMM scoring pipeline:
   - Token embeddings are computed via the BAM attention spectrogram (STFT of attention patterns)
   - The MLP scoring network assigns per-token importance scores (shape: `[batch, n_heads, n_tokens]`)
   - The selection criteria determines which tokens to retain/evict
   - All scores and indices are returned in analysis dicts

### Metrics

1. **Spearman rank correlation (per layer):** For each layer, tokens are ranked by NAMM score and by mean attention received (averaged over heads and query positions). Spearman rho measures rank agreement. Positive rho = NAMM retains tokens the model attends to; negative rho = NAMM retains tokens the model does *not* attend to.

2. **Eviction regret (per layer):** For each token NAMM would evict, the total and mean attention it would have received from all query positions. High regret = NAMM removes tokens the model still needs.

3. **Alignment shift (M1 vs M3, per task):** Mean Spearman correlation compared between conditions.

## Findings

### Score-Attention Correlation

| Metric | M1 | M3 cs1024 |
|--------|-----|-----------|
| Mean Spearman rho (all layers) | -0.137 | -0.136 |
| Std Spearman rho | 0.070 | 0.071 |

See `score_attention_correlation.png`. The correlation is **negative at every layer**, with the strongest anti-correlation in early layers (rho ~ -0.25 at layers 2-3) and weaker anti-correlation in later layers (rho ~ -0.10 at layers 8-15). M1 and M3 curves are virtually indistinguishable.

Layer-by-layer pattern:
- **Layer 0:** Near-zero correlation (rho ~ -0.02) — NAMM and attention are orthogonal
- **Layers 1-4:** Strong negative correlation (rho ~ -0.20 to -0.25) — NAMM actively retains tokens the model does not attend to
- **Layers 5-6:** Brief positive spike — partial alignment
- **Layers 7-15:** Moderate negative correlation (rho ~ -0.10 to -0.15) — persistent anti-alignment

### Eviction Regret

See `eviction_regret.png`.

| Metric | M1 | M3 cs1024 |
|--------|-----|-----------|
| Mean total regret (all layers) | 0.070 | 0.067 |

Total regret peaks in early layers (0-3) where attention is broadest and more tokens carry non-trivial attention mass. Later layers have lower regret as attention becomes more focused on fewer tokens. M3 shows marginally lower regret than M1 across most layers.

Per-token regret shows a spike at layer 6, suggesting this layer has a few heavily-attended tokens that NAMM would evict.

### Alignment Shift (M1 vs M3)

See `alignment_shift.png`. Per-task comparison shows:

- All tasks have negative alignment (rho < 0 everywhere)
- M1 and M3 are nearly identical across all 5 tasks
- HotpotQA-E shows the least negative alignment (rho ~ -0.05) — consistent with Report 1's finding that eviction helps this task
- Qasper and 2WikiMQA show more negative alignment (rho ~ -0.12 to -0.15)

## Interpretation

### Why is the correlation negative?

This is the most surprising finding. The original hypothesis predicted moderate *positive* correlation, since NAMM should evict tokens the model doesn't need. Instead, NAMM's scores are anti-correlated with attention. Several explanations:

1. **NAMM uses a different importance signal.** NAMM's BAM scoring network uses attention *spectrograms* (STFT of attention patterns over the KV cache) as input, not raw attention weights. The spectrogram captures temporal patterns — how attention to a token changes across layers — which is a fundamentally different signal from instantaneous attention magnitude at any single layer. NAMM may learn that tokens with *stable, low-variation* attention patterns are safe to keep (they serve a consistent role), while tokens with *high, spiky* attention are transient and can be evicted after the relevant layers have processed them.

2. **Attention ≠ information necessity.** High attention does not mean a token is needed for future processing. Attention sinks (BOS tokens) receive massive attention but carry little information. NAMM may learn to evict high-attention low-information tokens (sinks in later layers) while retaining low-attention high-information tokens (rare content tokens needed for answer generation).

3. **NAMM optimises task performance, not attention preservation.** NAMM was trained via CMA-ES to maximise F1, not to preserve attention patterns. The evolution discovered that the best eviction strategy is not "keep what the model attends to" but rather "keep what the model will need later," which may include currently-unattended tokens.

### Why doesn't M3 increase alignment?

The hypothesis predicted M3 would show higher NAMM-attention alignment after fine-tuning with eviction active. Instead, M1 and M3 are nearly identical. This means:

1. **M3's adaptation does not reshape attention to match NAMM.** Instead of learning to attend to tokens NAMM retains (which would increase alignment), M3 learns a *different* strategy — distributing attention more broadly (Report 5, +4% entropy) to extract information from all tokens before eviction happens. This is pre-emptive information gathering, not post-hoc alignment.

2. **The LoRA adaptation and the NAMM policy remain independent systems.** M3's LoRA modifies attention patterns (Report 5) but not in a direction that aligns with NAMM's scoring. The two systems cooperate by operating on complementary signals rather than converging on the same importance ranking.

### Connection to other reports

| Report | Finding | Implication for Report 6 |
|--------|---------|--------------------------|
| 1 (Sensitivity) | M3 matches M1 on aggregate F1 | The negative alignment doesn't hurt — NAMM's anti-correlated eviction strategy works |
| 4 (LoRA weights) | Orthogonal subspaces, M3 norms larger | M3's LoRA doesn't optimise toward NAMM alignment |
| 5 (Attention) | M3 has higher entropy, lower sinks | M3 broadens attention pre-emptively rather than aligning with NAMM |

### Implications for the paper

1. **NAMM operates on a complementary signal to attention.** This is a novel finding — learned eviction policies don't simply preserve high-attention tokens. The spectrogram-based scoring captures a different notion of importance (temporal attention patterns across layers) that is anti-correlated with instantaneous attention. This distinction is important for understanding why learned eviction outperforms attention-based heuristics (H2O, ScissorHands).

2. **Fine-tuning adapts by hedging, not aligning.** M3 doesn't learn to agree with NAMM; it learns to be robust to NAMM's decisions regardless of what they are. This "pre-emptive hedging" strategy (Report 5) is complementary to NAMM's eviction rather than redundant with it.

3. **Low eviction regret validates NAMM's quality.** Despite the negative correlation with attention, only ~7% of attention mass falls on evicted tokens, and M3 further reduces this to ~6.7%. NAMM selectively evicts tokens with low absolute attention mass even though their relative NAMM ranking is inverted.

## Figures

- `score_attention_correlation.png` — Spearman rho between NAMM scores and mean attention received, per layer, for M1 and M3
- `eviction_regret.png` — Total and per-token attention mass on evicted tokens, per layer
- `alignment_shift.png` — Per-task comparison of mean alignment (Spearman rho) between M1 and M3

## References

- Kim et al. (2022). "Learned Token Pruning for Transformers." *KDD 2022*.
- Munkhdalai et al. / Sakana AI (2024). "Neural Attention Memory Models."
- Xiao et al. (2024). "Efficient Streaming Language Models with Attention Sinks." *ICLR 2024*.
