# Section C — Eviction-Mask Drift Report

This document is the *interpretation companion* to the metrics in
`eval_results/section_c_metrics.json` and the figures in
`figures/section_c/`. It does not pick a verdict until the numbers are in
— instead, each figure below is paired with **decision templates** that
map concrete metric ranges to the corresponding thesis-text sentence.

**Question.** Does the frozen NAMM, when dropped on top of different
fine-tuned LLMs, keep the same KV tokens? Concretely: for B0 (base),
M1 (base+LoRA trained without NAMM) and M4 (base+LoRA trained jointly
with NAMM), how much does the 1024-token eviction mask change at the
end of prefill on the same 70-prompt FAIR-01 test split?

All runs share: `cache_size=1024`, `temperature=0.0`,
`task@_global_=rh_multi_qa_5t`, `train_frac=0.7`, `val_frac=0.15`,
`split_seed=42`, `+protected_tail_n=5`, and the same
`--namm_checkpoint`.

---

## Figure C1 — Pairwise mask IoU (headline)

`figures/section_c/C1_mask_overlap.pdf`

A 3×3 heatmap of mean Intersection-over-Union between the sets of
retained prompt positions, averaged over prompts, heads and layers. The
diagonal is 1.0 by construction. The off-diagonal cells quantify
cross-condition drift. The plot title reports two reference baselines:

- **Cross-prompt B0 baseline.** Mean IoU between two *different*
  prompts' masks under B0 — i.e. how far apart NAMM's masks are when
  nothing except the prompt text changes. A condition-pair IoU *below*
  this is meaningful drift; *above* it means the substrate change moves
  the mask less than just feeding a new prompt does.
- **Random baseline.** Closed-form `k / (2P − k)` for two
  uniformly-drawn top-k masks at the same retention rate. This is the
  floor for any non-trivial policy.

**Decision template (fill in after looking at C1).**

| Observation | Text to use in §6.x |
|---|---|
| `IoU(B0, M4) ≥ cross-prompt B0` | NAMM retains substantially more in common between B0 and M4 than between two different prompts. The eviction policy is effectively substrate-invariant on this split, so the joint-training concern (that M4 learns a substrate-specific mask that "expects" LoRA activations) is empirically unfounded. |
| `IoU(B0, M4) ≈ cross-prompt B0` | The LoRA adapter induces about as much mask drift as a different prompt does. Mask drift and prompt variability are comparable-magnitude phenomena; neither dominates. |
| `IoU(B0, M4) < cross-prompt B0` | NAMM re-ranks enough tokens under M4 that B0's and M4's masks look further apart than two independent prompts under B0. Joint training materially reshaped the policy in a substrate-specific way. |
| `IoU(B0, M4) ≈ random` | The two masks barely overlap. This would contradict the paper's §5.4 "NAMM is nearly attention-orthogonal" finding and should trigger a re-check of the NAMM checkpoint / LoRA checkpoint pairing. |
| `IoU(B0, M1) > IoU(B0, M4)` | M1-LoRA (trained without NAMM in the loop) perturbs NAMM's masks *less* than M4-LoRA (trained jointly with NAMM). This would be the evidence that joint training shapes the LLM toward NAMM's masking, not the other way around. |
| `IoU(B0, M1) < IoU(B0, M4)` | M1 drifts more than M4. This is the expected direction if M4's joint training pulled the LLM's representations back into alignment with NAMM's scoring head. |

---

## Figure C2 — Retention by layer (union over heads)

`figures/section_c/C2_retention_by_layer.pdf`

For each condition, per-layer mean fraction of *unique* prompt positions
retained by at least one head at the final eviction step. Note:
per-head retention is 1024/P by construction at `cache_size=1024` (top-k
binds) so it would be flat — we plot the **union over heads**, which
varies because different heads evict different tokens.

**Decision template.**

| Observation | Interpretation |
|---|---|
| All three curves lie on top of each other | Heads' evictions disagree about the same tokens regardless of the underlying LLM. The per-layer head diversity is a property of NAMM + prompt, not of the substrate. |
| M4 curve sits **below** B0/M1 | M4 heads agree more with one another (smaller union) — joint training pushed heads toward redundant eviction. |
| M4 curve sits **above** B0/M1 | M4 heads diverge more (larger union) — each head keeps a distinct slice. This would be mildly surprising given the shared scoring head. |
| Late layers (13–15) drop visibly | Late-layer retention shrinks because heads all agree on a small "answer-relevant" subset. If this only happens under M4, the LoRA has made late-layer representations more mask-agreeable. |

---

## Figure C3 — Score distribution shift (appendix)

`figures/section_c/C3_score_distributions.pdf`

KDE / histogram of the final NAMM token scores pooled across layers and
heads, one curve per condition, with a vertical dashed line at the
eviction threshold `s=0`. Captioned with pairwise Kolmogorov-Smirnov
distances.

**Decision template.**

| Observation | Interpretation |
|---|---|
| `KS ≤ 0.05` on every pair | Score distributions are statistically indistinguishable at this sample size. What little drift exists in C1 comes from *re-ordering* of tokens near the k-th quantile, not from shifting the overall score landscape. |
| `KS(B0, M4) > 0.1` | The distribution of NAMM scores genuinely shifts under M4. A rightward shift would mean more tokens pass the implicit threshold; a leftward shift the opposite. Cross-reference with per-layer retention (C2) — if C2 doesn't change, the score shift is a monotone re-scaling that leaves the top-k set near-invariant. |
| KDE shows a clear bimodal-vs-unimodal pattern switch | NAMM developed (or lost) a "keep / evict" bimodality on one substrate. This is worth a direct column in §6. |

---

## Figure C4 — IoU by layer (appendix)

`figures/section_c/C4_iou_by_layer.pdf`

Per-layer IoU for the three cross-condition pairs. Lets us see whether
drift is concentrated in specific layers (e.g. matching the §5.3 per-layer
hidden-state drift, which localises to the last ~3 layers).

**Decision template.**

| Observation | Interpretation |
|---|---|
| Drift is uniform across layers | NAMM's policy drifts globally under LoRA. Consistent with "LoRA rotates the whole residual stream a little". |
| Drift concentrates in layers 13–15 | Matches the §5.3 per-layer hidden-state-shift profile. Mask drift is driven by the same late-layer perturbation that LoRA is known to cause — not a new phenomenon. |
| Drift concentrates in early layers (0–3) | Surprising, because §5.3 finds early-layer hidden states are nearly untouched. Would warrant re-checking that the NAMM checkpoint really is identical across conditions. |
| `IoU(B0, M4)` dips only at the last layer | Only the final layer's mask is sensitive. This is consistent with LoRA's documented "output-layer adaptation" story (LoRA mostly changes what the final layer emits). |

---

## Cross-check: Spearman ρ(NAMM scores, LLM attention) vs paper §5.4

`eval_results/section_c_metrics.json::spearman_namm_vs_attn`

Paper §5.4 reports `ρ_M1 = −0.115` and `ρ_M4 = −0.168`, measured at the
final step over the same 70-prompt test split (our assumption). The
analyzer reproduces these from the dumps and emits a WARN line if
`|our − paper| > 0.03`.

- **If reproduction is within ±0.03**: treat §5.4 as independently
  confirmed. Cite C1/C2 as complementary "mask-level" corroboration of
  the "NAMM is attention-orthogonal" claim.
- **If reproduction exceeds ±0.03**: do not cite §5.4 numbers as
  reproduced. The discrepancy is likely because §5.4 used a different
  prompt subset (val + test rather than test alone, or a pre-chat-template
  tokenisation). Report both numbers and explain the split difference.
- **If the sign flips**: investigate before reporting anything. A sign
  flip would mean the attention capture is misaligned with the scoring
  slots, or the NAMM checkpoint swap is somehow wrong.

---

## Caveats that must appear in the caption of any published figure

1. **Protected tail.** `+protected_tail_n=5` forces the last 5 prompt
   positions (the chat-template assistant header) to be retained at
   every step. Those positions are excluded from IoU / retention / KS
   computations in the analyzer. Without that exclusion, every pair of
   conditions would enjoy +5/1024 ≈ 0.5% of trivial agreement.
2. **Per-head IoU uses each head's own retained prompt positions.** The
   dump records `final_retained_positions` by gathering each head's
   `position_ids` through that head's `retained_idxs` at the final step
   — so head divergence after multi-step gather is preserved. IoU is
   then computed per (layer, head) and averaged; this is stricter than
   "union over heads" IoU and is the right granularity for the drift
   question.
3. **M1 with NAMM is an off-distribution setting for M1.** M1 was
   trained without NAMM in the loop, so running it with cache=1024
   eviction produces the lower F1 cited in the paper. For Section C we
   nonetheless evaluate M1 with NAMM active — the comparison only makes
   sense when all three conditions share the same eviction pipeline.
4. **Top-k retention is a hard budget.** At `cache_size=1024` and
   `prompt_length > 1024`, per-(layer, head) retention count is exactly
   1024. Any "retention rate" reported per layer is therefore a union
   across heads, not a per-head quantity — this is why C2's y-axis can
   exceed `1024 / prompt_length` and why flat per-head retention isn't
   plotted.
5. **Extended-test is an optional appendix.** The 154-prompt
   `extended_test` split adds volume but is *not* what §6.x's headline
   numbers are computed on. Use it only as a robustness appendix.

---

## Numeric summary template (to fill in after running the analyzer)

Replace with values from `eval_results/section_c_metrics.json` when the
full run completes. Retain the wording so the report text is stable
across re-runs.

```
Pairwise IoU (mean over prompts × heads × layers):
    B0 ↔ M1 = {pairwise_iou.B0__M1.mean:.3f}
    B0 ↔ M4 = {pairwise_iou.B0__M4.mean:.3f}
    M1 ↔ M4 = {pairwise_iou.M1__M4.mean:.3f}

References:
    cross-prompt IoU under B0 = {reference_cross_prompt_iou_B0.mean:.3f}
    analytic random floor     = {reference_random_baseline_iou:.3f}

Per-layer retention (union over heads, final step):
    B0: mean {per_layer_retention.B0.per_layer_mean[0]:.3f} .. {per_layer_retention.B0.per_layer_mean[15]:.3f}
    M1: mean {per_layer_retention.M1.per_layer_mean[0]:.3f} .. {per_layer_retention.M1.per_layer_mean[15]:.3f}
    M4: mean {per_layer_retention.M4.per_layer_mean[0]:.3f} .. {per_layer_retention.M4.per_layer_mean[15]:.3f}

Spearman ρ(NAMM scores, LLM attention):
    paper §5.4: M1 = -0.115, M4 = -0.168
    this run  : M1 = {spearman_namm_vs_attn.M1.mean:+.3f}, M4 = {spearman_namm_vs_attn.M4.mean:+.3f}
```
