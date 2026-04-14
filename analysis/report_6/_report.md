# Analysis 6 -- Token Importance Alignment (NAMM Scores vs Attention)

> Checkpoints: M2 NAMM from WandB `z5bo4n8k`, M3 LoRA from `h0bzg6on` step 260.

> **TL;DR:** NAMM scores are weakly **negatively** correlated with
> attention (Spearman rho ≈ -0.15).  NAMM tends to retain tokens the
> model does *not* heavily attend to.  All three conditions (M1, M2, M3)
> show the same pattern — the LoRA does not change the NAMM-attention
> relationship.

---

## Methodology

All three conditions run through the NAMM chunked pipeline
(`apply_memory_policy=True`, 256-token chunks) on full-length prompts
(4096-6500 tokens).  Per-token NAMM scores and retention decisions are
captured via a monkey-patch on `select_new_tokens`.  Per-token attention
received is accumulated across chunks via forward hooks on `self_attn`.

- **Samples:** 365 (73 per task, balanced across all 5 tasks)
- **Conditions:** M1 (LoRA full-context weights), M2 (no LoRA),
  M3 (LoRA eviction-aware weights).  All use the same NAMM policy.
- **Cache size:** 1024 tokens

> **Note on M1:** M1 normally operates without NAMM (full context, no
> eviction).  Here we run M1's weights through the NAMM pipeline to
> measure how NAMM would score tokens given M1's attention patterns.
> This is hypothetical for M1 — it never sees NAMM in practice.  M2 and
> M3 are in their actual operating regimes.

---

## Findings

### Score-Attention Correlation

| Condition          | Mean Spearman rho | Std   |
| ------------------ | ----------------: | ----: |
| M1 (full context)  |            -0.115 | 0.227 |
| M2 (NAMM, no LoRA) |            -0.151 | 0.235 |
| M3 (LoRA + NAMM)   |            -0.168 | 0.237 |

NAMM scores are weakly **negatively** correlated with attention — tokens
that receive more attention tend to get *lower* NAMM retention scores.
This is counterintuitive but consistent across all conditions and all
365 samples.

The per-layer pattern (see `score_attention_correlation.png`) shows
positive correlation in layer 0 (~+0.35) that drops to negative in
layers 2-6 (~-0.2 to -0.3), then fluctuates around -0.1 to -0.4 in
deeper layers.

### M1 ≈ M2 ≈ M3

All three conditions show the same pattern.  The LoRA (M1 or M3)
does not reshape attention-NAMM alignment.  The correlation is a
property of the NAMM scoring network interacting with the base model's
attention, not of the LoRA adaptation.

### Eviction Regret

| Condition          | Mean total regret | Mean per-token regret |
| ------------------ | ----------------: | --------------------: |
| M1 (full context)  |             0.280 |               0.00108 |
| M2 (NAMM, no LoRA) |             0.289 |               0.00112 |
| M3 (LoRA + NAMM)   |             0.291 |               0.00112 |

Regret measures the attention mass on evicted tokens.  ~28-29% of
total attention lands on tokens that NAMM evicts.  This is consistent
with the negative correlation: NAMM evicts tokens the model attends to.

---

## Discussion

### Why is the correlation negative?

NAMM's scoring network uses attention spectrograms (STFT of attention
patterns across chunks) and positional embeddings, not raw attention
magnitude.  The negative correlation suggests NAMM's learned scoring
assigns high importance to tokens that are *not* heavily attended in the
current chunk — possibly tokens that carry information the model hasn't
yet integrated, or positionally important tokens (e.g., section
boundaries) that don't receive direct attention.

This finding differs from the old 15-sample analysis (which reported
rho = +0.135 on 1024-token truncated prompts).  The discrepancy is
likely due to sequence length: at 1024 tokens NAMM barely evicts,
while at 4096-6500 tokens NAMM aggressively evicts ~96% of tokens.
The scoring behaviour at different compression levels may be
qualitatively different.

### LoRA does not change NAMM-attention alignment

M1, M2, and M3 all show the same correlation, confirming Report 5's
finding that the LoRA does not change attention patterns.  M3's
performance advantage (val F1 52.06 vs M2's 14.90) comes from
value-space extraction, not from reshaping attention toward or away
from NAMM's preferences.

### Connection to Other Reports

| Report        | Finding                            | Implication                                   |
| ------------- | ---------------------------------- | --------------------------------------------- |
| 5 (Attention) | M2 ≈ M3 entropy under eviction     | Consistent: LoRA doesn't change attention     |
| 3 (Retention) | ~4-6% retention                    | High regret expected at extreme eviction      |
| 4 (LoRA wts)  | Orthogonal subspaces, 1.42x norms  | Large weight diffs but same attention pattern |

---

## Figures

| Plot                                                                           | Description                                    |
| ------------------------------------------------------------------------------ | ---------------------------------------------- |
| [`score_attention_correlation.png`](plots/score_attention_correlation.png)      | Spearman rho per layer (M1 vs M2 vs M3)        |
| [`eviction_regret.png`](plots/eviction_regret.png)                             | Attention mass on evicted tokens per layer      |
| [`alignment_shift.png`](plots/alignment_shift.png)                             | Per-task alignment comparison (M1 vs M2 vs M3)  |
