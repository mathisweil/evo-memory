# Independent Critique of the Analysis Work

> Last updated: 2026-04-14.
> Status: All reports (0--9) rewritten with maskfix data. M3 maskfix
> training ~43% complete (step 260 of ~608). No maskfix test-set evals.

> **Naming warning:** Reports use the M0--M3 convention from the
> experiment specification. Results directories and some older code
> use "M4" for what is actually M3 (LoRA + frozen NAMM). Real M4
> (joint co-training of LoRA and NAMM) has not been run.

---

## 1. Strengths of the Analysis Suite

The nine-report structure is a genuine strength. Starting from dataset
characterisation (R0) through performance (R1--R2), internal mechanics
(R3--R5), model-policy interaction (R6), representational similarity
(R7), probing (R8), and gradient flow (R9), each report addresses a
distinct question, and the findings compose into a coherent narrative.
Specific strengths:

1. **Hypothesis-driven framing.** Report 0 makes falsifiable predictions;
   Report 1 tests them and honestly acknowledges where they fail. This
   drives the remaining investigation in a principled way.

2. **Converging evidence from independent measurements.** Weight-space
   analysis (R4: orthogonal subspaces), attention patterns (R5: hedging),
   token alignment (R6: weak positive correlation), and CKA (R7: high
   similarity, divergence at layer 10) all independently support the claim
   that M3 learns a qualitatively different but representationally mild
   adaptation.

3. **Reproducibility infrastructure.** WandB run IDs are documented in
   every report. Each report has a self-contained `generate_plots.py`
   that pulls data from WandB or checkpoint files. Plot filenames are
   tagged `_maskfix` to distinguish from buggy-era outputs.

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
estimated 608 steps (~43% of training). Every maskfix number in every
report is an interim result from a partially-trained model. This is
the single most important caveat in the analysis suite.

The consequences are not merely that "numbers might change slightly."
The reports themselves show that M3 was still improving at step 260
(Report 2: 90% of the improvement range reached at step 222, but best
val F1 still rising). A model at 43% through training may have a
qualitatively different internal structure from the converged model --
the adaptation might reorganise as training progresses. Specific risks:

- **R1 headline numbers will shift.** The M3 val F1 of 52.06 is the
  best so far, not the final value. If training plateaus or degrades,
  the 14.5% advantage over M1 could narrow.
- **R4 LoRA norm ratios are snapshot-dependent.** Adapter norms
  typically grow during training. The current 1.42x ratio may increase.
- **R7 divergence locus may move.** The layer-10 CKA minimum could
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

### 2.2 No maskfix test-set evaluations exist

All reported F1 numbers are **validation** metrics. The existing
test-set evaluations (in `results/main_table_5t/all_results.json`)
were run on the **buggy** checkpoints. No maskfix conditions have been
evaluated on the held-out test split.

This matters for two reasons:

1. **Selection bias.** Reporting the best validation F1 as the primary
   metric inflates results because the checkpoint was implicitly
   selected to maximise it. The M3 val F1 of 52.06 at step 260 is the
   peak of a noisy curve; the true expected performance on unseen data
   is lower.
2. **Generalisation is untested.** The buggy test-set evals showed that
   validation-to-test gaps can be substantial and condition-dependent
   (e.g., the buggy A4/cs2048 shifted by +12 points after the eval
   fix). Without maskfix test-set numbers, the headline claim that "M3
   exceeds M1" rests entirely on validation data.

Until maskfix test-set evaluations are run, the analysis is preliminary.

### 2.3 Small and truncated GPU analysis samples

Reports 4--9 use between 10 and 60 samples, truncated to 1024 tokens.
The training data consists of 4096--6500 token sequences. Truncating to
1024 tokens puts both models in an out-of-distribution regime where
neither model's learned adaptation is fully exercised.

| Report | Samples | Tokens | Measurement               |
| ------ | ------: | -----: | ------------------------- |
| 4      |      10 |  1,024 | LoRA weight norms, overlap|
| 5      |      10 |  1,024 | Attention entropy, sinks  |
| 6      |      15 |  1,024 | NAMM-attention correlation|
| 7      |      10 |  1,024 | CKA similarity            |
| 8      |      40 |  1,024 | Linear probes             |
| 9      |      40 |  1,024 | Gradient flow             |

None of these reports include error bars, bootstrap confidence
intervals, or significance tests (except Report 3, which uses
correlations from WandB logs across many steps). At n=10--15, the
standard error on mean entropy or CKA is large enough that the reported
differences could plausibly be noise. The entropy shift of +5.0% and
sink shift of -2.7% (Report 5) are presented as definitive findings, but
no statistical test accompanies them.

### 2.4 Report 8 probe task is poorly calibrated

Report 8 (probing) is the weakest in the suite and the report
acknowledges this directly. The binary label (was any answer token
evicted?) produces a majority-class baseline of 0.600. Both M1 (0.599)
and M3 (0.513) probe accuracies are at or below this baseline. The
probe has no discriminative power and the results are uninformative.

The probe design is flawed at a more fundamental level: mean-pooled
hidden states over the full sequence are a blunt instrument for
detecting whether specific answer tokens were in the evicted set. A
probe that examines token-level representations at the positions
surrounding the answer region, or that uses a more informative label
(e.g., fraction of answer tokens evicted), might have more power.

The report correctly labels its results as "inconclusive." The concern
is that the inconclusive report still appears in the analysis chain and
is cited in cross-report connection tables, which risks giving it more
weight than its null result deserves.

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

**Report 6 (Token Alignment): Sign flip.** Under the bug, the
NAMM-attention Spearman correlation was rho = -0.137, interpreted as
"complementary signals" -- NAMM retains tokens the model does NOT attend
to. With maskfix, rho = +0.135, meaning NAMM retains tokens the model
DOES attend to. This is the opposite interpretation: the corrected
result says NAMM is doing what one would expect (retaining important
tokens), not pursuing a complementary strategy. The entire "NAMM and
LoRA operate as complementary systems" narrative from the buggy analysis
was wrong. Reports that cited this finding need to be read with this
reversal in mind.

**Report 7 (CKA): Layer shift from 3 to 10.** The buggy CKA minimum
was at layer 3 (CKA 0.979); with maskfix it is at layer 10 (CKA 0.990).
The buggy mask forced early aggressive corrections, pushing divergence
to layer 3. With correct masking, the adaptation defers to a later
semantic-integration layer. This also changes the cross-report
convergence story: the old summary identified layer 3 as a critical
adaptation point across multiple reports; that convergence no longer
exists.

**Report 9 (Gradient Flow): Retention shifted dramatically.** The
buggy NAMM retained ~20.1% of tokens; the corrected NAMM retains
~3.8%. The absolute loss gap is nearly identical (7.68 vs 7.80), but
the corrected NAMM is 5x more aggressive and 19% more efficient per
percentage point of eviction. The gradient distortion finding
(cosine sim ~0.01) is consistent across both regimes.

### 3.2 Findings that are robust to the mask fix

**Report 4 (LoRA Weights): Orthogonal subspaces persist.** The buggy
norm ratio was 1.93x; maskfix gives 1.42x -- quantitatively smaller,
but the qualitative finding (M3 norms > M1, near-orthogonal subspaces)
holds. The subspace overlap changed modestly (0.18 to 0.21 for q_proj).

**Report 5 (Attention Entropy): Hedging pattern persists.** Both
buggy and maskfix M3 show higher entropy and lower sink fractions
relative to M1, with similar magnitudes (+5.0% entropy, -2.7% sinks).
This is arguably the most robust finding in the suite -- the hedging
pattern survived a fundamental change to the attention mechanism.

**Report 3 (Retention Patterns): Layers 8--9 most aggressive in both
regimes.** The NAMM's per-layer eviction structure is stable because
the NAMM itself (M2) was trained independently.

---

## 4. Open Methodological Issues

### 4.1 "Pre-emptive hedging" remains an untested narrative

The summary interprets M3's broader attention as "pre-emptive hedging"
-- the model distributes attention to be robust to arbitrary evictions.
This is post-hoc storytelling. An equally valid interpretation: M3's
LoRA was optimised under a noisy training signal (evicted context varies
per step), and the broader attention is an artefact of optimisation
under noise, analogous to how dropout produces flatter weight
distributions. To test the hedging hypothesis, one would need to show
that M3's performance is more robust to *random* eviction patterns than
M1's -- that the broader attention is specifically functional, not a
side-effect.

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
correlations). The headline finding "M3 (52.06) exceeds M1 (45.48)" is
a difference of 6.58 F1 points on a validation set of ~64 samples. This
is likely significant, but "likely" is not a substitute for a test.
The mechanistic findings (entropy +5.0%, CKA min 0.990, rho +0.135) at
n=10--15 especially need uncertainty quantification.

---

## 5. What Should Be Done Next

In priority order:

1. **Complete M3 maskfix training.** The current checkpoint is at ~43%.
   All maskfix analyses are interim. Training must finish before any
   result can be considered final.

2. **Rerun reports 1--9 with the final M3 checkpoint.** The current
   analyses are snapshots at step 260. Once training completes (or
   early-stops by a proper criterion), the full analysis suite should
   be rerun to verify that the qualitative findings hold at convergence.

3. **Run maskfix test-set evaluations.** All headline F1 numbers
   currently rest on validation data. Test-set evaluation on the held-out
   split is necessary before any claim about M3 vs M1 performance can
   be made with confidence.

4. **Add bootstrap confidence intervals** to at least the R1 per-task
   F1 comparison (M3 vs M1) and the R5 entropy/sink comparisons. This
   does not require new GPU runs -- it requires resampling existing
   per-sample or per-prompt metrics.

5. **Increase GPU analysis sample sizes** if computationally feasible.
   Moving from 10--15 to 30--50 samples would substantially tighten
   standard errors. If full-length (4096+) inputs are too expensive,
   a truncation to 2048 would be less distributional shift than 1024.

6. **Compute the random subspace overlap baseline** for Report 4.

7. **Redesign the probe task** for Report 8 (or drop it). The current
   binary classification with a 0.600 majority baseline has no power.
   A regression probe on the fraction of answer tokens evicted, or a
   per-position probe, would be more informative.

8. **Run the A4 ablation with maskfix checkpoints.** The existing A4
   results (M3 checkpoint evaluated without NAMM at inference) are from
   buggy checkpoints. This ablation directly tests whether M3's hedging
   is functional under full context.

---

## 6. Summary Assessment

The analysis suite is well-structured, transparent, and covers an
unusually broad range of mechanistic questions for an MSc thesis. The
core finding -- that eviction-aware training produces a qualitatively
different LoRA adaptation (orthogonal subspaces, broader attention, high
CKA) that surpasses full-context fine-tuning -- is supported by
converging evidence from multiple independent measurements and survived
a major methodological correction (the attention mask fix).

However, the work is currently in an interim state. The M3 maskfix
checkpoint is less than half-trained, no test-set evaluations exist for
corrected models, sample sizes in GPU analyses are small, and no
statistical tests accompany any finding. The Report 6 sign flip
demonstrates how a single bug can invert a qualitative conclusion -- a
reminder that the remaining findings, while consistent, are not immune
to revision once training completes and proper evaluation is done.

The most robust findings are:
- M3 attention hedging (R5): replicated across buggy and maskfix regimes
- Orthogonal LoRA subspaces (R4): qualitatively stable, quantitatively
  shifted
- Gradient distortion under eviction (R9): fundamental to the regime

The least reliable findings are:
- Absolute F1 numbers (R1--R2): interim checkpoint, validation only
- Probe results (R8): null result from a poorly calibrated task
- Layer-specific convergence narratives: the layer-3 story collapsed
  with maskfix; the layer-10 story may shift with further training

The path forward is clear: complete M3 training, rerun the analyses,
run test-set evaluations, and add statistical tests. The infrastructure
is in place to do all of this efficiently.
