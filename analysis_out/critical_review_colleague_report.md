# Critical Review: Colleague's Summary Report

**Date:** 2026-04-13
**Context:** Review in light of the attention mask bug discovered during this session.

---

## The Attention Mask Bug (discovered 2026-04-13)

During NAMM's split processing, the `attention_mask` tracks cumulative prompt length while the KV cache is compacted to ~cache_size after eviction. The causal mask slice `[:,:,:,-kv_len:]` grabs the wrong region of the oversized mask, causing softmax to produce **perfectly uniform attention** (entropy = log(N), std = 0) after ~9 eviction steps during prefill. This bug exists in the original NAMM codebase from the first commit.

**Impact:** All prefill hidden states after chunk ~9 are computed under uniform attention. Generation-time attention is unaffected (mask rebuilt fresh by `prepare_inputs_for_generation`).

**Fix:** Committed as `e3655c3`. When mask-cache mismatch exceeds q_len, build a fresh causal mask sized to actual kv_len instead of slicing from the oversized mask.

**Consequence for the report:** All NAMM eval numbers (M2, M3, B1, M1_under_NAMM) were computed with broken prefill attention. The NAMM policy was also *trained* under this broken attention (CMA-ES fitness eval uses the same forward pass). With the fix applied but without retraining NAMM, M3/cs1024 dropped from 32.28 to 23.52 — the policy's eviction decisions are wrong with correct attention.

---

## What Holds

### Report 4 — LoRA Weight Comparison
**Status: VALID.** Weight-space analysis (cosine overlap, norm ratios, subspace orthogonality) is independent of the attention mask bug. The LoRA weights are what they are.

### Report 5 — Attention Entropy
**Status: PARTIALLY VALID.** The measurements were on full-context inputs (no eviction at inference), so the mask bug doesn't affect them directly. However, the "pre-emptive hedging" interpretation needs reframing: M3 was *trained* with broken prefill attention, so the broader attention may be an adaptation to the uniform-attention training regime rather than a genuine eviction-robust strategy. After NAMM retraining with the fix, the attention patterns may differ.

### Report 7 — CKA Representation Similarity
**Status: VALID.** CKA measures representation geometry, independent of the mask bug.

### Report 6 — NAMM Scores vs Attention Correlation
**Status: VALID but interpretation changes.** The negative correlation (rho = -0.14) is real, but NAMM's scoring was operating in a regime where prefill attention was uniform for most of processing. The "complementary signal" interpretation may not hold once attention is fixed and NAMM is retrained — the scoring network may learn to align more with attention when attention is actually informative.

### A4 Ablation Results
**Status: VALID at inference** (A4 doesn't use NAMM at inference). But the LoRA weights were trained under broken NAMM attention, so "eviction as regulariser" should be reframed as "training under constrained/uniform attention regime as regulariser." A4/cs2048 = 33.91 > M1 = 31.14 stands, but the mechanism may differ from what's described.

### Truncation Baselines
**Status: VALID.** Truncation evals don't go through NAMM's split processing (we deactivated NAMM entirely for truncation runs), so they're unaffected by the mask bug.

---

## What Doesn't Hold

### Headline: M3/cs1024 (32.28) > M1 (31.14)
**Status: CONTAMINATED.** This was computed with broken prefill attention. With the mask fix (but old NAMM policy), M3/cs1024 = 23.52. The NAMM policy needs retraining with correct attention before this claim can be made. The claim *may* still hold after retraining — but we don't know yet.

### Claim 3: Eviction Hierarchy (NAMM > Truncation > Recency)
**Status: NEEDS RE-EVALUATION for base model.** With the mask fix, M2/cs1024 dropped from 20.30 to 10.83 — now worse than truncation (18.21). The hierarchy may reverse or may be restored after NAMM retraining. The *with-LoRA* hierarchy (M3 > Trunc+LoRA) also needs retraining to verify.

### Report 8 — Probing for Evicted Content
**Status: NEEDS RE-RUNNING.** Probe accuracy patterns were measured under broken attention. The information loss patterns in layers 7-14 may change substantially with correct attention.

### Report 9 — Gradient Flow
**Status: NEEDS RE-RUNNING.** The 865% loss increase and uncorrelated gradients partly reflect the uniform attention regime, not just eviction. The gradient flow under correct attention will be different.

### "Pre-emptive Hedging" Hypothesis
**Status: NEEDS REFRAMING.** The evidence (broader attention, reduced sink reliance) was measured on models trained under broken attention. The hedging may be a response to uniform attention during training rather than a deliberate eviction-robust strategy. After retraining, M3 may adopt a different strategy entirely.

---

## What's Missing from the Report

### 1. The Attention Mask Bug as a Finding
This should be a prominent part of the paper. It exists in the original NAMM codebase and potentially affects all published NAMM results. The discovery that attention collapses to perfectly uniform after ~9 eviction steps is novel — to our knowledge, no prior work has reported this.

### 2. Ghost Information Analysis
We measured KV vector cosine similarity between NAMM-evicted and full-context conditions. Key finding: negative cosines (keys point in opposite directions) across layers 2-15. However, this was measured under broken attention, so some of the "ghost contamination" may actually be "uniform attention contamination." Needs re-running after NAMM retraining.

### 3. Generation-Time Attention Is Correct
The mask bug only affects prefill, not generation. During token-by-token generation, attention is sharp and selective (entropy ~2.5, max weight ~0.9). This explains why models produce reasonable outputs despite broken prefill: generation reads from degraded cache entries but with proper selective attention.

### 4. Recency Baseline Explanation
The report says M1_recency/cs1024 = 0.0 is a "pipeline failure." The actual cause: the recency baseline runs the full NAMM deep-policy stack with zero weights (scoring_initializer=0), producing degenerate eviction based on STFT architectural biases that favor early tokens (attention sinks), not a last-N strategy. It's a design issue, not a pipeline failure.

### 5. F1 vs Prompt Length Analysis
We showed M4/cs1024 F1 collapses from 32.4 (in-distribution 4096-5500 tokens) to 5.7 (far-OOD 7500-8200 tokens). Truncation stays at 16.7. This is currently attributed to OOD eviction decisions, but with the mask bug, the collapse may be caused by increasing mask misalignment on longer prompts (more chunks = more accumulated mismatch).

---

## Recommendations

1. **Retrain NAMM with the mask fix** — ESSENTIAL before publishing. Commands prepared, ~10-12h on 4090.
2. **Re-run all NAMM evals** after retraining to get clean numbers.
3. **Re-run Reports 8 and 9** after retraining — gradient flow and probing results will change.
4. **Frame the mask bug as a finding** in the paper: "We discovered that the original NAMM implementation suffers from attention collapse during split-processed prefill due to a causal mask misalignment."
5. **Keep pre-fix and post-fix results side by side** — showing the NAMM policy is coupled to its training attention regime is itself a finding about distribution shift.
6. **The A4 story holds** regardless — A4/cs2048 > M1 doesn't depend on NAMM at inference. But reframe the mechanism.
7. **After retraining:** if M3-maskfix > M1, the claims are stronger than before (clean attention, proper eviction). If M3-maskfix < M1, the paper's contribution shifts to the bug discovery + analysis of how training under broken attention accidentally produces good models.
