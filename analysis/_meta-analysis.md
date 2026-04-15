# Independent Critique of the Analysis Work

> Last updated: 2026-04-15.
> Status: All reports (0--9) rewritten with maskfix data. M3 maskfix
> training step 260 of 425 (run killed). Maskfix test-set evals now
> available for all conditions; M3 > M1 confirmed on test.

> **Naming warning:** Reports use the M0--M3 convention from the
> experiment specification. Results directories and some older code
> use "M4" for what is actually M3 (LoRA + frozen NAMM). Real M4
> (joint co-training of LoRA and NAMM) has not been run.

---

## 1. Strengths of the Analysis Suite

The nine-report structure is a genuine strength. Starting from dataset
characterisation (R0) through performance (R1--R2), internal mechanics
(R3--R5), model-policy interaction (R6), representational similarity
(R7), and gradient flow (R9), each report addresses a
distinct question, and the findings compose into a coherent narrative.
Specific strengths:

1. **Hypothesis-driven framing.** Report 0 makes falsifiable predictions;
   Report 1 tests them and honestly acknowledges where they fail. This
   drives the remaining investigation in a principled way.

2. **Converging evidence from independent measurements.** Weight-space
   analysis (R4: orthogonal subspaces), attention patterns (R5: M2 ≈ M3),
   token alignment (R6: weak positive correlation), and CKA (R7: high
   similarity, divergence at layer 9) all independently support the claim
   that M3 learns a qualitatively different but representationally mild
   adaptation.

3. **Reproducibility infrastructure.** WandB run IDs are documented in
   every report. Each report has a self-contained `generate_plots.py`
   that pulls data from WandB or checkpoint files. Plot filenames are
   in `plots/` subdirectories within each report folder.

4. **Honest about limitations.** Reports consistently flag the M3
   checkpoint maturity issue, small GPU-analysis sample sizes, and the
   difference between validation and test metrics.

5. **The maskfix rewrite was done thoroughly.** All nine reports were
   updated with corrected-mask data. Buggy numbers are preserved as
   historical context sections rather than silently replaced, making
   the analysis transparent about its own evolution.

---

## 2. Methodological Concerns

### 2.1 The M3 maskfix checkpoint is immature

The maskfix M3 run (`h0bzg6on`) has completed only step 260 of an
425 steps before kill. Every maskfix number in every
report is an interim result from a partially-trained model. This is
the single most important caveat in the analysis suite.

The consequences are not merely that "numbers might change slightly."
The reports themselves show that M3 was still improving at step 260
(Report 2: 90% of the improvement range reached at step 222, but best
val F1 still rising). A model at 61% through its (incomplete) training may have a
qualitatively different internal structure from the converged model --
the adaptation might reorganise as training progresses. Specific risks:

- **R1 headline numbers will shift.** The M3 val F1 of 52.06 is the
  best so far, not the final value. If training plateaus or degrades,
  the 14.5% advantage over M1 could narrow.
- **R4 LoRA norm ratios are snapshot-dependent.** Adapter norms
  typically grow during training. The current 1.42x ratio may increase.
- **R7 divergence locus may move.** The layer-9 CKA minimum could
  shift as later layers receive more gradient updates.
- **R3 retention patterns are from the M2 NAMM, not M3.** The NAMM
  is frozen, so retention patterns themselves are stable -- but the
  LoRA checkpoint against which they are analysed is not.

**Comparison with buggy runs is confounded by training stage.** The
buggy M3 checkpoint ran for ~600 steps (near convergence), while the
maskfix checkpoint is at step 260. Differences between buggy and
maskfix results conflate two effects: (a) the attention mask fix and
(b) checkpoint maturity. Reports generally acknowledge this, but the
confound makes it impossible to attribute any specific change cleanly
to the mask fix alone.

### 2.2 Large validation-to-test gaps

Maskfix test-set evaluations now exist for all conditions:

| Condition            | val F1 | test F1 | ext F1 | val-to-test gap |
| -------------------- | -----: | ------: | -----: | --------------: |
| B0 plain             |     -- |   22.41 |  22.42 |              -- |
| M2 NAMM cs1024       |  14.90 |   19.27 |  18.70 |           +4.37 |
| M1 LoRA (full cache) |  45.48 |   27.97 |  25.75 |          -17.51 |
| M3 LoRA +NAMM cs1024 |  52.06 |   33.51 |  25.84 |          -18.55 |
| A4 LoRA (no NAMM)    |     -- |   36.07 |  24.91 |              -- |

The headline finding **M3 > M1 holds on test** (33.51 vs 27.97,
+5.54 F1), partially resolving the prior concern that the comparison
rested entirely on validation data. However, two issues remain:

1. **Substantial val-to-test drops.** M1 falls 17.5 points and M3
   falls 18.6 points from validation to test. This is consistent with
   checkpoint selection bias: the validation set is small (n=64
   samples), and peak validation F1 was used to select checkpoints.
   The val metric effectively overfits to the validation distribution.
   M2's anomalous *increase* from val (14.90) to test (19.27) may
   reflect a different metric computation during training versus
   standalone evaluation, or simply that M2's poor val performance
   had nowhere to go but up.

2. **The ext split shows further degradation for fine-tuned models.**
   M3 ext F1 (25.84) is close to M1 ext F1 (25.75), erasing most of
   M3's test-set advantage. A4 shows the starkest pattern: 36.07 test
   vs 24.91 ext (-11.16 points), suggesting that fine-tuned models
   overfit to the test distribution as well, or that the ext split
   contains harder examples. This warrants investigation.

The test results confirm that M3 > M1 generalises beyond the
validation set, but the magnitude of the advantage is much smaller
than the validation numbers suggest (5.54 vs 6.58 F1 points), and the
ext split narrows it further.

### 2.3 Small and truncated GPU analysis samples

Reports 4--9 use between 10 and 60 samples, truncated to 1024 tokens.
The training data consists of 4096--6500 token sequences. Truncating to
1024 tokens puts both models in an out-of-distribution regime where
neither model's learned adaptation is fully exercised.

| Report | Samples | Tokens | Measurement               |
| ------ | ------: | -----: | ------------------------- |
| 4      |      10 |  1,024 | LoRA weight norms, overlap|
| 5      |      50 |  4,096+ | Attention entropy         |
| 6      |      15 |  1,024 | NAMM-attention correlation|
| 7      |      10 |  1,024 | CKA similarity            |
| 8      |      -- |     -- | Abandoned (flawed labels) |
| 9      |      40 |  1,024 | Gradient flow             |

None of these reports include error bars, bootstrap confidence
intervals, or significance tests (except Report 3, which uses
correlations from WandB logs across many steps). At n=10--15, the
standard error on mean entropy or CKA is large enough that the reported
differences could plausibly be noise. Report 5 now uses 50 samples at
full length (4096+ tokens) with NAMM eviction active, which is a
substantial improvement, but still lacks formal statistical tests.

### 2.4 Report 8 abandoned

Report 8 (probing) has been dropped. The probe labels were constructed
by string-matching gold answers against the input context to identify
"answer token" positions, then checking whether those positions survived
NAMM eviction. This suffers from the same fundamental flaw as the
relevant-tokens analysis dropped from Report 0: string matching does not
give ground truth for which tokens are needed to answer the question.
Answer information often appears in paraphrased or indirect form. The
probe results were inconclusive (both M1 and M3 at the majority-class
baseline), likely reflecting noisy labels rather than anything about
information retention.

### 2.5 Recovery ratio sensitivity to small denominators

Report 1 computes recovery ratios as (M3 - M2) / (M1 - M2). When the
M1-M2 gap is small for a particular task, the recovery ratio becomes
extremely sensitive to noise. HotpotQA-E's recovery ratio of 200% has
a denominator of (44.00 - 14.00) = 30 points, which is substantial --
but for tasks where M1 and M2 are closer, the ratio would be unstable.
The mean recovery ratio of 121.53% is dominated by tasks with large
M3 gains. This metric is useful as a summary, but should not be
over-interpreted at the per-task level.

---

## 3. What Changed with Maskfix and What Is Robust

The attention mask bug caused the model to attend uniformly rather than
using the correct causal/position-aware mask. Fixing this changed some
findings substantially while leaving others qualitatively intact.

### 3.1 Changes that altered conclusions

**Report 6 (Token Alignment): Methodology corrected.** The old analysis
(15 samples, 1024-token truncation, two-pass analyze) reported
rho = +0.135.  With proper chunked processing on full-length prompts
(n=365), NAMM scores are weakly **anti-correlated** with attention
(rho ≈ -0.15).  NAMM retains tokens the model does NOT heavily attend
to — similar to the buggy-era finding (rho = -0.137), though for
different reasons.  All three conditions (M1, M2, M3) show the same
pattern.

**Report 7 (CKA): Layer shift from 3 to 9.** The buggy CKA minimum
was at layer 3 (CKA 0.979); with maskfix it is at layer 9 (CKA 0.990).
The buggy mask forced early aggressive corrections, pushing divergence
to layer 3. With correct masking, the adaptation defers to a later
semantic-integration layer. This also changes the cross-report
convergence story: the old summary identified layer 3 as a critical
adaptation point across multiple reports; that convergence no longer
exists.

**Report 9 (Gradient Flow): Retention shifted dramatically.** The
buggy NAMM retained ~20.1% of tokens; the corrected NAMM retains
~4%. With balanced sampling (n=255), eviction increases loss by 79%
(not 641% as the old 40-sample analysis reported) and gradient cosine
similarity is 0.21 (weakly aligned, not ~0.01). The old numbers were
biased by uneven task sampling toward harder examples.

### 3.2 Findings that are robust to the mask fix

**Report 4 (LoRA Weights): Orthogonal subspaces persist.** The buggy
norm ratio was 1.93x; maskfix gives 1.42x -- quantitatively smaller,
but the qualitative finding (M3 norms > M1, near-orthogonal subspaces)
holds. The subspace overlap changed modestly (0.18 to 0.21 for q_proj).

**Report 5 (Attention Entropy): Methodology corrected.** The old
analysis (both models on full context, no eviction) was flawed --
it measured a hypothetical, not actual operating regimes. The corrected
analysis runs M1 on full context, M2 and M3 with NAMM eviction active.
Result: M2 and M3 have nearly identical entropy (-1.4% difference),
meaning the LoRA does not change attention patterns. M3's performance
advantage comes from value-space extraction, not attention routing.
The old "hedging" narrative is not supported.

**Report 3 (Retention Patterns): Layers 8--9 most aggressive in both
regimes.** The NAMM's per-layer eviction structure is stable because
the NAMM itself (M2) was trained independently.

---

## 4. Open Methodological Issues

### 4.1 M3's mechanism of action is unclear

The corrected Report 5 shows M2 ≈ M3 in attention entropy, ruling out
the "pre-emptive hedging" narrative. M3's LoRA produces dramatically
different weights (Report 4: 1.42x norms, orthogonal subspaces) and
much higher F1 (52.06 vs M2's 14.90), but the attention distributions
are indistinguishable. This points to value-space (v_proj) extraction
as the mechanism, but this has not been directly tested. A targeted
ablation -- applying only q_proj or only v_proj LoRA -- would clarify
which projection drives M3's advantage.

### 4.2 Random subspace baseline still missing for Report 4

Report 4 claims near-orthogonal overlap (~0.21 for q_proj). For rank-8
subspaces in dimension 2048, the expected overlap between random
subspaces is approximately sqrt(8/2048) ~ 0.063. The observed 0.21 is
~3.3x the random baseline -- low, but meaningfully above chance. This
baseline should be computed explicitly rather than left to the reader.

### 4.3 Two GPU analysis code paths were never reconciled

Reports 5 and 7 had two implementations: `run_gpu_analyses.py` (bare
merged model, entropy averaged over all query positions) and per-report
`generate_plots.py` (NAMM memory model, entropy at last query position
only). These produce different measurements. The maskfix rewrite
presumably used one path consistently, but the code-level ambiguity was
never explicitly resolved in the reports.

### 4.4 No significance testing

Across nine quantitative reports, no p-values, bootstrap confidence
intervals, or permutation tests appear (except Report 3's retention
correlations). The headline finding "M3 exceeds M1" is a difference of 6.58 F1 points
on validation (52.06 vs 45.48) and 5.54 points on test (33.51 vs 27.97).
Both gaps are likely significant, but "likely" is not a substitute for a
test, especially given the small validation set (n=64).
Sample sizes have improved (n=365 for reports 5-7, n=255 for report 9)
but formal statistical tests are still absent.

---

## 5. What Should Be Done Next

In priority order:

1. **Complete M3 maskfix training.** The current checkpoint is at step 260 of 425 (run killed).
   All maskfix analyses are interim. Training must finish before any
   result can be considered final.

2. **Rerun reports 1--9 with the final M3 checkpoint.** The current
   analyses are snapshots at step 260. Once training completes (or
   early-stops by a proper criterion), the full analysis suite should
   be rerun to verify that the qualitative findings hold at convergence.

3. **Add bootstrap confidence intervals** to at least the R1 per-task
   F1 comparison (M3 vs M1) and the R5 entropy comparisons. This does
   not require new GPU runs -- it requires resampling existing
   per-sample or per-prompt metrics. With test-set results now
   available, confidence intervals on the test F1 gap (M3 - M1 = 5.54)
   are particularly important.

4. **Investigate the ext-split degradation.** M3's advantage over M1
   nearly vanishes on the ext split (25.84 vs 25.75). Understanding
   whether this reflects harder examples, distribution shift, or
   overfitting to the test distribution is important for the
   generalisability claim.

5. **Increase GPU analysis sample sizes** if computationally feasible.
   Moving from 10--15 to 30--50 samples would substantially tighten
   standard errors. If full-length (4096+) inputs are too expensive,
   a truncation to 2048 would be less distributional shift than 1024.

6. **Compute the random subspace overlap baseline** for Report 4.

7. **Revisit information retention (Report 8).** The probe approach was
   abandoned due to unreliable labels. A reformulation that probes for
   the answer itself (not specific token positions) could address this.

8. ~~**Run maskfix test-set evaluations.**~~ Done. Test-set results
   confirm M3 > M1 (+5.54 F1). See Section 2.2 for details.

---

## 6. Summary Assessment

The analysis suite is well-structured, transparent, and covers an
unusually broad range of mechanistic questions for an MSc thesis. The
core finding -- that eviction-aware training produces a qualitatively
different LoRA adaptation (orthogonal subspaces, unchanged attention
patterns, high CKA) that surpasses full-context fine-tuning -- is
supported by
converging evidence from multiple independent measurements and survived
a major methodological correction (the attention mask fix).

However, the work is currently in an interim state. The M3 maskfix
checkpoint is less than half-trained, sample sizes in GPU analyses are
small, and no statistical tests accompany any finding. Test-set
evaluations now confirm M3 > M1 (33.51 vs 27.97), but with large
val-to-test drops (~18 points for M3) and near-parity on the ext split
(25.84 vs 25.75). The Report 6 sign flip demonstrates how a single bug
can invert a qualitative conclusion -- a reminder that the remaining
findings, while consistent, are not immune to revision once training
completes and proper evaluation is done.

The most robust findings are:
- Orthogonal LoRA subspaces (R4): qualitatively stable, quantitatively
  shifted
- M2 ≈ M3 attention entropy (R5): LoRA does not change attention
  patterns; M3's advantage is in value extraction
- Moderate gradient distortion under eviction (R9): +79% loss, cos 0.21

The least reliable findings are:
- Absolute F1 numbers (R1--R2): interim checkpoint, large val-to-test
  gaps (~18 points), ext split nearly erases M3's advantage
- Probe results (R8): abandoned due to flawed labels
- Layer-specific convergence narratives: the layer-3 story collapsed
  with maskfix; the layer-9 story may shift with further training

The path forward is clear: complete M3 training, rerun the analyses,
add statistical tests, and investigate the ext-split degradation. Test-set
evaluations are now done and confirm the core M3 > M1 finding, but the
large val-to-test gaps and ext-split convergence temper the strength of
the claim. The infrastructure is in place to address the remaining items
efficiently.
