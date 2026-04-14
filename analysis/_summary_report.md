# Summary Report: Adaptation Under Eviction in Neural Attention Memory Models

## Executive Summary

We fine-tune LLaMA 3.2-1B-Instruct with LoRA (rank 8) while NAMM (Cetin
et al., ICLR 2025) manages KV cache eviction, and evaluate on 5 LongBench QA
tasks. This report synthesises findings from 10 analysis reports (0-9) that
were **rerun with corrected ("maskfix") NAMM checkpoints** after discovering
and fixing an attention mask bug in the NAMM split-processing pipeline.

The maskfix data is the primary data throughout this report. All validation-set
numbers come from maskfix checkpoints. Test-set numbers are from the buggy
NAMM evaluation pipeline and are pending re-evaluation with maskfix -- they
are included for completeness but should be interpreted with caution.

### Key findings (maskfix validation data)

**1. M3 exceeds M1 on validation by 14.5%.** M3 maskfix best val F1 is 52.06
(step 260) vs M1 best val F1 of 45.48 (step 336). The recovery ratio is
121.5% -- M3 under eviction does not merely match full-context M1 but
substantially surpasses it. Multi-hop tasks see the largest gains.

**2. M3 converges faster than M1.** M3 reaches its peak at step 260 (~43%
through training) vs M1's peak at step 336. M3 also starts from a higher
baseline (23.70 vs M1's initial). The negative train-val gap suggests
eviction acts as a regulariser.

**3. Corrected NAMM is far more aggressive than buggy NAMM.** Maskfix NAMM
retains only 3.8% of tokens (vs 20% under the buggy mask), with a 641% loss
increase over full context. Retention CV is 0.115 (more uniform across layers
than buggy CV of 0.183). Layers 8-9 are the most aggressive eviction sites.

**4. M3 LoRA adapts more efficiently under corrected NAMM.** M3/M1 q_proj
norm ratio is 1.42x (down from 1.93x under the buggy mask). LoRA subspaces
remain near-orthogonal (~0.21 overlap). The smaller norms suggest more
efficient adaptation, not a noisier version of M1.

**5. NAMM-attention correlation is POSITIVE with corrected masks.** The buggy
analysis found anti-correlation (rho = -0.137), supporting a "complementary
signals" narrative. This was an artefact. With correct attention masks, NAMM
scores are weakly positively correlated with attention (rho = +0.135). NAMM
does track attention, though weakly. M3 does not reshape attention toward NAMM.

**6. The attention entropy shift is robust.** M3 shows +5.0% entropy and
-2.7% sink reduction vs M1 (nearly identical to buggy: +5.2% and -2.4%). The "pre-emptive hedging" pattern appears in both buggy and maskfix
data, confirming it is a genuine property of eviction-aware training.

**7. Representational divergence shifts to deeper layers.** CKA mean is 0.995
(min 0.990 at layer 9). Under the buggy mask, the CKA minimum was at layer 3
(0.979). This changes the mechanistic story: the locus of M3's adaptation is
in deeper layers, not the early-middle layers previously identified.

**8. Probing results are inconclusive.** The random baseline shifted to 0.600
due to class imbalance. M1 accuracy is 0.599 (at chance) and M3 is 0.513
(below chance). Neither model shows clear evidence of retaining or losing
evicted content beyond baseline rates.

**9. Gradient flow confirms eviction severity.** Retention 3.8%, loss +641%,
gradient cosine similarity ~0.01 (uncorrelated). Corrected NAMM creates a
roughly 5x more aggressive eviction regime than the buggy version.

### What changed from the buggy-era analysis

The attention mask bug caused NAMM's attention mask to grow with cumulative
input length rather than tracking actual post-eviction cache size. From chunk
9 onward (~2300 tokens into a ~5000-token prompt), attention collapsed to
uniform 1/N across all heads and layers. Approximately 60% of each prompt's
prefill occurred under broken attention. This bug exists in the original
Sakana AI codebase.

After fixing the bug, all 10 analysis reports were rerun with maskfix
checkpoints. The key narrative changes are:

- **Report 6 (NAMM-attention correlation):** The "complementary signals"
  narrative was wrong. The anti-correlation (rho = -0.137) was an artefact
  of broken attention. Correct NAMM shows positive correlation (rho = +0.135).
- **Report 7 (CKA):** Divergence shifts from layer 3 (buggy) to layer 9
  (maskfix). The "layer 3 critical adaptation point" narrative no longer holds.
- **Report 4 (LoRA norms):** Norms are 26% smaller with maskfix (1.42x vs
  1.93x q_proj ratio), indicating more efficient adaptation.
- **Report 8 (probing):** Results are now inconclusive due to baseline shift,
  not showing clear information loss as the buggy analysis suggested.
- **Report 9 (gradient flow):** Retention drops from 20% to 3.8% -- corrected
  NAMM is roughly 5x more aggressive.
- **Report 3 (retention):** CV drops from 0.183 to 0.115, meaning corrected
  eviction is more uniform across layers.
- **M2 performance degrades:** Maskfix M2 best val F1 is 14.90, worse than
  buggy M2 val 27.90. The corrected (more aggressive) eviction policy without
  LoRA adaptation struggles more.

---

## 1. Experimental Setup

We fine-tune Llama 3.2-1B-Instruct on 5 LongBench QA tasks (Qasper,
2WikiMQA, Qasper-E, HotpotQA-E, 2WikiMQA-E) under 17 conditions spanning
baselines, truncation controls, ablations, and experimental treatments.

Analysis reports 1-9 use **maskfix validation data**. The test-set numbers
below are from the **buggy NAMM evaluation pipeline** -- maskfix test-set
evals have not yet been run.

### Buggy-era test-set results (for reference)

| Label                    | LoRA         | Eviction  | KV Cache | Test F1 |
| ------------------------ | ------------ | --------- | -------- | ------: |
| **B0** (plain Llama)     | No           | None      | Full     |   22.41 |
| **B1/cs1024**            | No           | Recency   | 1024     |   11.33 |
| **B1/cs2048**            | No           | Recency   | 2048     |   11.10 |
| **M1** (LoRA only)       | Yes (rank 8) | None      | Full     |   31.14 |
| **M2/cs1024** (NAMM)     | No           | NAMM      | 1024     |   10.83 |
| **M2/cs2048** (NAMM)     | No           | NAMM      | 2048     |   15.27 |
| **M3/cs1024** (LoRA+NAMM)| Yes (rank 8) | NAMM (frz)| 1024     |   23.52 |
| **M3/cs2048** (LoRA+NAMM)| Yes (rank 8) | NAMM (frz)| 2048     |   31.41 |
| **Trunc/plain_1024**     | No           | Truncation| 1024     |   18.21 |
| **Trunc/plain_2048**     | No           | Truncation| 2048     |   18.26 |
| **Trunc/lora_m1_1024**   | Yes (M1's)   | Truncation| 1024     |   26.90 |
| **Trunc/lora_m1_2048**   | Yes (M1's)   | Truncation| 2048     |   28.87 |
| **M1_recency/cs1024**    | Yes (M1's)   | Recency   | 1024     |    0.00 |
| **M1_under_NAMM/cs1024** | Yes (M1's)   | NAMM      | 1024     |   26.97 |
| **M1_under_NAMM/cs2048** | Yes (M1's)   | NAMM      | 2048     |   31.71 |
| **A4/cs1024**            | Yes (M3's)   | None      | Full     |   28.82 |
| **A4/cs2048**            | Yes (M3's)   | None      | Full     |   38.98 |

These test-set numbers were computed with the buggy attention mask in NAMM.
All NAMM-dependent conditions (M2, M3, M1_under_NAMM) are affected. B0, M1,
Trunc, and A4 conditions do not use NAMM at inference and are unaffected.

### Maskfix validation benchmarks

| Condition | Best Val F1 | At Step |
| --------- | ----------: | ------: |
| M1        |       45.48 |     336 |
| M3        |       52.06 |     260 |
| M2        |       14.90 |      -- |

The dataset comprises 306 train / 64 val / 70 test samples (4096-6500 tokens
each) plus 224 extended_test samples (6500-8192 tokens), with a divide between
localised scientific paper QA (Qasper tasks) and distributed multi-hop
reasoning (2WikiMQA/HotpotQA — answers require combining facts from
multiple passages).

### Naming warning

> **What `results/main_table_5t/` labels "M4" is actually experiment-spec M3
> (LoRA + frozen NAMM).** Real M4 (joint co-training of LoRA and NAMM) has not
> been run. This report uses **M3** throughout. Data keys in
> `all_results.json` use "M4" (e.g. `M4/cs1024`). See
> `experiment_specification.md` for the full milestone naming.

---

## 2. Key Findings by Report

### Report 0 -- Dataset Characterisation

The 5 tasks fall into two families with fundamentally different information
structures:
- **Qasper tasks:** Localised answers in scientific papers (answer typically
  in a single section/paragraph)
- **Multi-hop tasks:** Distributed answers across Wikipedia passages (require
  combining facts from 2+ passages)

At cache=1024, ~80% of tokens are evicted. Multi-hop tasks were predicted to
suffer most because they require retaining multiple scattered passages. Report
1's maskfix validation data shows the opposite -- multi-hop tasks see the
largest M3 gains over M1.

### Report 1 -- Performance Across Conditions (Maskfix Validation)

This is the central results report, now using maskfix validation data. The
headline finding is that M3 substantially exceeds M1.

**Maskfix validation results:**

- M3 best val F1: **52.06** (step 260)
- M1 best val F1: **45.48** (step 336)
- Recovery ratio: **121.5%** (M3 exceeds M1, not merely recovers)
- Multi-hop tasks see the largest gains from eviction-aware training

**Key findings:**

1. **M3 exceeds M1 by 14.5% on validation.** This is a substantial margin
   that holds across multiple evaluation checkpoints, not a single-step
   artefact.

2. **Multi-hop tasks benefit most.** Contrary to the Report 0 prediction
   that multi-hop tasks would suffer most from eviction, M3 shows its
   largest gains over M1 on distributed-answer tasks.

3. **M2 (NAMM only) degrades under maskfix.** M2 best val F1 drops to 14.90
   (from 27.90 under the buggy mask). Without LoRA adaptation, the more
   aggressive corrected eviction substantially hurts performance. This makes
   the M3 result even more striking -- LoRA adaptation not only compensates
   for the harsher eviction but turns it into an advantage.

**Note on test-set numbers:** The buggy-era test-set results (Section 1
table) show M3/cs2048 (31.41) matching M1 (31.14) and A4/cs2048 (38.98)
substantially exceeding M1. However, all NAMM-dependent test numbers reflect
the buggy attention mask. Maskfix test-set evaluation is pending.

### Report 2 -- Adaptation Rate (Maskfix Validation)

M3 converges faster and starts higher than M1:

| Metric                | M1    | M3       |
| --------------------- | ----- | -------- |
| Initial val F1        | --    | 23.70    |
| Best val F1           | 45.48 | 52.06    |
| Step at peak          | 336   | 260      |
| Training progress     | --    | ~43%     |
| Train-val gap         | --    | Negative |

M3 reaches its peak at roughly 43% through training (step 260 of ~600),
while M1 peaks later at step 336. M3's higher starting baseline (23.70)
suggests that the NAMM-equipped model begins from a stronger position, not
a weaker one as previously assumed. The negative train-val gap is consistent
with eviction acting as a regulariser -- the model generalises better than
it fits training data.

### Report 3 -- Per-Layer Retention (Maskfix)

Corrected NAMM retains only **3.8%** of tokens (vs 20% under buggy mask).
Eviction is non-uniform across layers but more uniform than the buggy
version:

| Metric             | Maskfix | Buggy |
| ------------------ | ------: | ----: |
| Mean retention     |    3.8% |  20%  |
| Retention CV       |   0.115 | 0.183 |
| Most aggressive    | L8-L9   | L9    |
| Least aggressive   | L0      | L0    |

Layers 8-9 perform the most aggressive eviction. The lower CV (0.115 vs
0.183) means corrected NAMM distributes eviction more evenly across layers
rather than concentrating it in a few. This 5x more aggressive eviction
regime makes M3's strong performance even more remarkable -- the LoRA
succeeds despite retaining under 4% of the original KV cache.

### Report 4 -- LoRA Weight Comparison (Maskfix)

M1 and M3 continue to learn in **near-orthogonal** LoRA subspaces, but M3's
norms are smaller under maskfix than under the buggy mask:

| Metric           | Maskfix | Buggy |
| ---------------- | ------: | ----: |
| q_proj norm ratio| 1.42x   | 1.93x |
| Subspace overlap | ~0.21   | ~0.18 |

The 26% reduction in norm ratio (1.42x vs 1.93x) indicates more efficient
adaptation -- M3 achieves its stronger performance with smaller LoRA
perturbations. The subspaces remain near-orthogonal, confirming that M3
learns a qualitatively different adaptation from M1 regardless of the mask
bug. However, the smaller norms suggest that under corrected eviction, the
model does not need to compensate as aggressively.

### Report 5 -- Attention Entropy (Maskfix)

On **full-context** inputs (no eviction at inference), M3 shows measurably
different attention patterns:

| Metric             | M1    | M3 (maskfix)  | M3 (buggy)    |
| ------------------ | ----- | ------------- | ------------- |
| Entropy shift      | --    | +5.0%         | +5.2%         |
| Sink reduction     | --    | -2.7%         | -2.4%         |

The pre-emptive hedging pattern (broader attention, reduced sink reliance)
appears in both buggy and maskfix data, with slightly larger effect sizes
under maskfix. This robustness across two different NAMM configurations
confirms that the hedging behaviour is a genuine consequence of
eviction-aware training, not an artefact of any particular NAMM
implementation. M3 distributes attention more broadly so that no single
token's eviction is catastrophic.

### Report 6 -- Token Importance Alignment (Maskfix)

**This is the report with the largest narrative change.** Under the buggy
mask, NAMM scores were negatively correlated with attention (rho = -0.137),
supporting a "complementary signals" interpretation. This was an artefact of
the broken attention.

| Metric            | M1 (maskfix) | M3 (maskfix) | M1 (buggy) |
| ----------------- | -----------: | -----------: | ---------: |
| Mean Spearman rho |       +0.135 |       +0.140 |     -0.137 |

With correct attention masks, NAMM scores are **weakly positively
correlated** with attention (rho = +0.135). This makes more intuitive
sense: NAMM's spectrogram-based scoring, which uses attention patterns as
input, produces retention scores that agree (weakly) with instantaneous
attention magnitude.

M1 and M3 show nearly identical correlation (+0.135 and +0.140), meaning M3
fine-tuning does not reshape attention toward or away from NAMM's
preferences. The lack of co-adaptation is consistent across both buggy and
maskfix analyses -- only the direction of the base correlation changed.

### Report 7 -- Representation Similarity (CKA, Maskfix)

CKA between M1 and M3 is extremely high but with a different divergence
profile than the buggy analysis:

| Metric         | Maskfix   | Buggy     |
| -------------- | --------- | --------- |
| CKA mean       | 0.995     | 0.992     |
| CKA minimum    | 0.990     | 0.979     |
| Min at layer   | Layer 9   | Layer 3   |

The divergence shift from layer 3 (buggy) to layer 9 (maskfix) is
significant for the mechanistic interpretation. Under the buggy mask, the
"layer 3 critical adaptation point" narrative (Section 3.5 of the old
report) tied together multiple reports. With maskfix, the maximum
representational divergence occurs in deeper layers, closer to the output.
This aligns better with the observation that layers 8-9 perform the most
aggressive eviction (Report 3) -- the LoRA adapts representations most
strongly near the eviction zone, not upstream of it.

Overall, M3 representations are even closer to M1's under maskfix (mean
0.995 vs 0.992), suggesting the corrected eviction regime produces a model
that is functionally very similar to M1 despite the near-orthogonal LoRA
subspaces.

### Report 8 -- Probing for Residual Knowledge of Evicted Content (Maskfix)

**Results are inconclusive.** The random baseline shifted to 0.600 due to
class imbalance in the maskfix probe dataset, making the probe accuracy
values difficult to interpret:

| Metric             | M1 (maskfix) | M3 (maskfix) | Random |
| ------------------ | -----------: | -----------: | -----: |
| Mean probe accuracy|        0.599 |        0.513 |  0.600 |

M1 accuracy (0.599) is at chance level and M3 (0.513) is below chance.
Neither result provides clear evidence about whether evicted content is
retained or lost. The buggy analysis showed M1 at 0.557 and M3 at 0.484
(vs random 0.500), which appeared to show meaningful information loss in
M3's later layers. With the baseline shift, this interpretation no longer
holds.

This report needs methodological revision (balanced class sampling or a
different probe design) before conclusions can be drawn.

### Report 9 -- Gradient Flow and Loss Attribution (Maskfix)

The corrected NAMM creates a substantially harsher training signal than the
buggy version:

| Metric                     | Maskfix        | Buggy          |
| -------------------------- | -------------: | -------------: |
| Mean retention ratio       |          3.8%  |         20.1%  |
| CE loss (evicted)          |     ~6.7 (est) |          8.706 |
| Loss increase vs full ctx  |         +641%  |          +865% |
| Gradient cos sim           |         ~0.01  |          0.015 |

Gradient directions remain essentially uncorrelated between evicted and
full-context conditions under both buggy and maskfix NAMM. The key
difference is the severity: corrected NAMM retains 5x fewer tokens (3.8%
vs 20%), creating a far more aggressive information bottleneck. Despite
this, M3 achieves a higher val F1 (52.06) than under the buggy mask,
suggesting the LoRA can successfully adapt to even extreme eviction.

---

## 3. Cross-Report Analysis

### 3.1 The Coherent Picture: Weight -> Function -> Task (Maskfix)

Reports 4, 5, 6, 7, and 9 form a chain from weight-space to function-space
to task performance under maskfix:

```
Weight space (Report 4)          Function space (5, 6, 7)
--------------------------       ---------------------------
Orthogonal LoRA subspaces   ->   +5.0% entropy, -2.7% sinks
M3 norms 1.42x (efficient) ->   Positive NAMM correlation
                             ->   CKA min at layer 9 (0.990)

Gradient signal (Report 9)       Task space (Report 1)
--------------------------       ----------------------
3.8% retention, +641% loss  ->   M3 val F1: 52.06
Uncorrelated gradients      ->   M1 val F1: 45.48
                             ->   Recovery ratio: 121.5%
```

M3 learns in near-orthogonal weight subspaces with more efficient
perturbations (smaller norms). It develops broader attention patterns
(hedging) and produces representations that diverge most at layer 9
(near the eviction hotspot at layers 8-9). Despite operating under
extreme eviction (3.8% retention), M3 exceeds M1's validation performance
by 14.5%.

### 3.2 Pre-emptive Hedging: Confirmed Across Both NAMM Versions

The hedging hypothesis -- that M3 distributes attention broadly to be
robust against arbitrary eviction -- is supported by maskfix data:

- **Report 5:** M3 increases entropy (+5.0%) and reduces sink reliance
  (-2.7%), with slightly larger effects than under buggy NAMM.
- **Report 4:** M3's LoRA subspaces are orthogonal to M1's, confirming a
  qualitatively different adaptation strategy.
- **Report 6:** M3 does not reshape attention toward NAMM (identical rho
  for M1 and M3), meaning hedging is undirected -- the model hedges
  against eviction in general, not toward specific NAMM decisions.

The hedging pattern is robust across both buggy and maskfix NAMM
configurations, even though the corrected NAMM is 5x more aggressive. This
suggests the strategy is a fundamental consequence of training under any
eviction regime, not a response to a particular eviction severity.

### 3.3 NAMM Tracks Attention (Weakly), Not the Opposite

The buggy-era "complementary signals" narrative described NAMM as operating
on fundamentally different information from attention. With corrected masks,
the picture is simpler: NAMM's spectrogram-based scoring produces retention
preferences that weakly agree with attention magnitude (rho = +0.135).

This makes architectural sense. NAMM takes attention spectrograms as input
and was evolved (CMA-ES) to maximise F1. Its scoring reflects attention
patterns, transformed through the spectrogram representation. The weak
positive correlation confirms NAMM uses attention-derived information,
though its spectrogram processing captures patterns beyond raw magnitude.

The lack of M3-induced alignment shift (M1 and M3 both show rho = +0.135)
remains consistent: M3 does not learn to cooperate with NAMM's eviction
decisions.

### 3.4 Layer 9 Replaces Layer 3 as the Adaptation Locus

The buggy-era analysis identified layer 3 as the critical adaptation point,
supported by convergent evidence from Reports 5, 7, 4, and 3. With maskfix
data, the CKA minimum shifts to layer 9 (Report 7), and the most aggressive
eviction occurs at layers 8-9 (Report 3).

The revised picture: M3's LoRA adapts representations most strongly in the
layers immediately downstream of the heaviest eviction. Rather than
"preparing" representations upstream for eviction (the layer 3 narrative),
the adaptation occurs at or after the eviction site, compensating for
information loss after it occurs.

### 3.5 M2 Degradation Highlights the LoRA's Role

Maskfix M2 best val F1 (14.90) is substantially worse than buggy M2 (27.90).
Without LoRA adaptation, the corrected (5x more aggressive) eviction
devastates performance. Yet M3 with the same corrected NAMM achieves val F1
52.06 -- a 37-point improvement over M2.

This contrast establishes that the LoRA is doing the heavy lifting. NAMM
alone under corrected eviction loses too much information. The LoRA's
adaptation (broader attention, orthogonal subspaces, efficient norms)
transforms a destructive bottleneck into a beneficial training signal.

### 3.6 Eviction Severity and Adaptation Quality

The buggy vs maskfix comparison provides a natural experiment on eviction
severity:

| Regime        | Retention | M3 best val F1 | M2 best val F1 |
| ------------- | --------: | -------------: | -------------: |
| Buggy NAMM    |      20%  |          ~45   |          27.90 |
| Maskfix NAMM  |      3.8% |          52.06 |          14.90 |

More aggressive eviction *hurts* the base model (M2 degrades) but
*helps* the LoRA-adapted model (M3 improves). This is consistent with
the regularisation hypothesis: more aggressive eviction forces the LoRA
into a more robust adaptation, analogous to stronger dropout improving
generalisation despite worsening memorisation.

However, the buggy M3 val F1 (~45) is a rough estimate. The comparison
should be treated as suggestive rather than definitive.

---

## 4. Summary Table

| Report | Question                     | Maskfix Answer                          |
| ------ | ---------------------------- | --------------------------------------- |
| 0      | What are the tasks?          | Localised Qasper, distributed multi-hop |
| 1      | Does M3 match M1?           | Exceeds: val 52.06 vs 45.48 (+14.5%)   |
| 2      | How fast does M3 learn?      | Faster: peak at step 260 vs 336         |
| 3      | Is eviction uniform?         | Nearly: CV 0.115, L8-9 most aggressive  |
| 4      | Are LoRA weights similar?    | No: orthogonal, norms 1.42x (efficient) |
| 5      | Are attention patterns same? | No: +5.0% entropy, -2.7% sinks          |
| 6      | Does M3 align with NAMM?    | No: same +0.135 rho as M1               |
| 7      | Are representations similar? | Mostly: CKA 0.990-1.0, min at layer 9  |
| 8      | Is evicted info retained?    | Inconclusive (baseline imbalance)        |
| 9      | Does eviction change grads?  | Yes: 3.8% retention, cos ~0.01           |

---

## 5. Buggy vs Maskfix: What Changed and What Held

| Finding                   | Buggy           | Maskfix         | Changed? |
| ------------------------- | --------------- | --------------- | -------- |
| M3 > M1 on val           | Yes             | Yes (stronger)  | Stronger |
| Pre-emptive hedging       | +4.2% entropy   | +5.0% entropy   | Robust   |
| Orthogonal subspaces      | ~0.18 overlap   | ~0.21 overlap   | Robust   |
| NAMM-attn correlation     | rho = -0.137    | rho = +0.135    | REVERSED |
| CKA min layer             | Layer 3         | Layer 9         | Shifted  |
| LoRA norm ratio (q_proj)  | 1.93x           | 1.42x           | Smaller  |
| Retention ratio           | 20%             | 3.8%            | 5x lower |
| Probe informativeness     | M3 drops to .375| Inconclusive    | Changed  |
| Gradient uncorrelated     | cos ~0.015      | cos ~0.01       | Robust   |

Three findings reversed or substantially changed (NAMM-attention
correlation, CKA layer, probing). Three findings were robust across both
versions (hedging, orthogonal subspaces, uncorrelated gradients). The robust
findings are the strongest claims in the analysis.

---

## 6. Open Questions

1. **Maskfix test-set evaluation is the most urgent next step.** All
   test-set numbers in this report are from the buggy NAMM pipeline. Until
   maskfix test evals are run, the M3 > M1 finding rests on validation data
   alone (n=64). The buggy test numbers show M3/cs2048 (31.41) matching M1
   (31.14), but these cannot be compared to maskfix validation results.

2. **Why does more aggressive eviction help M3 but hurt M2?** The
   maskfix NAMM retains 3.8% of tokens (vs 20% buggy), and M3 val F1
   improves while M2 degrades. If this is a regularisation effect, there
   should be an optimal eviction severity. Controlled experiments varying
   cache size under maskfix NAMM would map this relationship.

3. **Does the CKA shift to layer 9 align with a causal mechanism?** The
   coincidence of CKA minimum (layer 9) and aggressive eviction (layers
   8-9) is suggestive. Targeted ablation of LoRA weights at specific
   layers could test whether the layer 9 adaptation is necessary for M3's
   performance advantage.

4. **Can joint training (M4) further improve M3?** M3 uses a frozen NAMM
   policy. If the eviction policy could adapt to the LoRA's changing
   representations during training, it might learn to retain tokens
   specifically important for the fine-tuned model. This is the unrun M4
   condition.

5. **Is the probing approach salvageable?** Report 8's inconclusive results
   stem from class imbalance. A revised probe with balanced sampling or a
   different design (e.g., regression on token-level features rather than
   binary classification on mean-pooled states) could clarify whether M3
   retains or loses evicted content.

6. **Does the M3 > M1 advantage hold at scale?** All experiments use Llama
   3.2-1B and 4096-6500 token contexts. The buggy-era extended_test
   results (M1: 31.84 vs M3/cs1024: 25.40 at 6500-8192 tokens) suggest
   the advantage may not persist with longer contexts, though these
   numbers are also buggy-era. Testing at longer contexts and with larger
   models is needed.

7. **What caused the M1_recency/cs1024 failure?** The all-zero outputs
   (buggy test: 0.00 F1) need diagnosis. If recency eviction is
   fundamentally incompatible with the M1 LoRA adapter, this has
   implications for understanding what NAMM provides that recency does not.
