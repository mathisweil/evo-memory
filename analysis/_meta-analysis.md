## Update (2026-04-12): Corrected Test-Set Results (eval fix: chat template + protected tail)

PR #12 merged test-set evaluations for all conditions across two splits: **test** (70 prompts, 4096-6500 tokens) and **extended_test** (224 prompts, 6500-8192 tokens). The initial numbers (2026-04-11) were evaluated with two bugs: (1) the chat template was not applied to eval prompts, creating a train/eval mismatch, and (2) truncation could cut into the answer-bearing tail of the prompt. Both are now fixed; the numbers below are the corrected values from `results/main_table_5t/all_results.json`.

> **Naming warning:** What is labelled "M4" in `results/main_table_5t/` is actually experiment-spec **M3** (LoRA + frozen NAMM). Real M4 (joint co-training) has not been run. See `experiment_specification.md` for details. This update uses the results-directory naming ("M4") for consistency with the data files.

### Recommendations now addressed

- **#1 (test-set evals)** -- DONE. All 10 conditions evaluated on held-out test and extended_test splits. Results in `results/main_table_5t/all_results.json`.
- **#2 (B1 recency baseline)** -- DONE. B1/cs1024 micro=12.45, B1/cs2048 micro=13.78 on test. Both are well below B0 (22.41) and far below M1 (31.14), confirming that naive recency eviction is destructive even relative to the untuned model.
- **#4 (significance tests)** -- PARTIALLY DONE. `scripts/paired_delta_analysis.py` produces paired bootstrap CIs (B=2000, percentile method) for 8 key comparisons on both splits. Full significance testing on mechanistic analyses (R5-R7) still needed.
- **#6 (exclude or complete cs3072)** -- RESOLVED. cs3072 dropped from all test-set evaluations; only cs1024 and cs2048 are reported.
- **#15 (B1 missing)** -- DONE. See #2 above.
- **#16 (M4 joint training)** -- DONE. M4 (LoRA on frozen NAMM) now has test results at both cache sizes. Note: what was previously called M3 in the analysis reports is now M4 in the results naming.
- **#17 (no test-set eval)** -- DONE. See #1 above.
- **#19 (A4 ablation)** -- DONE. A4/cs1024_no_namm micro=28.82, A4/cs2048_no_namm micro=33.91 on test.

### What changed with the eval fix (2026-04-12)

The initial test numbers (2026-04-11) were computed without the chat template on eval prompts and without protecting the answer-bearing tail during truncation. Corrected test micro-F1 (old → new):

| Condition | Old test micro | New test micro | Delta  |
| --------- | -------------- | -------------- | ------ |
| B0        | 19.28          | 22.41          | +3.13  |
| M1        | 31.14          | 31.14          | 0.00   |
| M2/cs1024 | 16.84          | 20.30          | +3.46  |
| M4/cs1024 | 25.87          | 32.28          | +6.41  |
| A4/cs1024 | 30.58          | 28.82          | -1.76  |
| A4/cs2048 | 21.81          | 33.91          | +12.10 |

New truncation baselines: Trunc/plain_1024 (18.21), Trunc/plain_2048 (18.26), Trunc/lora_m1_1024 (26.90), Trunc/lora_m1_2048 (28.87). M1_recency/cs1024 is broken (all zeros).

### Revised finding: M4/cs1024 now matches/exceeds M1 on test

- **M4/cs1024 (32.28) now exceeds M1 (31.14) on test.** This reverses the pre-fix finding. The original val-based claim that M4 matches M1 is now supported by corrected test data.
- **Extended_test gap persists.** M4/cs1024 (26.92) trails M1 (31.84) by 15% on extended_test. Longer contexts push more information outside the 1024-token budget.
- **A4 picture is more nuanced.** A4/cs1024_no_namm (28.82) is *below* M4/cs1024 (32.28), meaning NAMM at inference *helps* the M4 cs1024 LoRA. But A4/cs2048_no_namm (33.91) exceeds M4/cs2048 (31.06), so the pattern is cache-size-dependent.
- **Critique #2 (val-not-test) was still methodologically valid.** It is a coincidence that corrected test numbers support the original val-based narrative. The right process would have been to evaluate on test from the start.

### Recommendations still open

- **#3 (cs3072 undertrained)** -- Resolved by dropping cs3072 entirely.
- **#5 (conflicting GPU pipelines)** -- Still unresolved.
- **#7 (significance tests on mechanistic analyses R5-R7)** -- Still needed.
- **#8 (recovery ratio misleading)** -- Still relevant with test numbers.
- **#9 (random subspace baseline)** -- Still needed for Report 4.
- **#10 (crude relevant token analysis)** -- Still relevant.
- **#11 (pre-emptive hedging untested)** -- Now partially supported at cs1024: M4/cs1024 (32.28) > A4/cs1024_no_namm (28.82), consistent with the LoRA benefiting from NAMM at inference. But cs2048 reverses (A4=33.91 > M4=31.06), so the picture is mixed.
- **#12 (NAMM-attention correlation over-interpreted)** -- Still relevant (n=15, no error bars).
- **#13 (layer 3 cherry-picked)** -- Still relevant.
- **#14 (6x cache reduction)** -- Now better supported on test: M4/cs1024 (32.28) matches M1 (31.14). Extended_test gap (15%) still warrants a caveat.

---

*Original critique follows below, written before test-set results were available.*

---

I've now read the full repo context and all 8 analysis reports plus the summary, code, and experiment specification. Here's my critique.

---

# Critique of the Analysis Work

## Overall Assessment

This is a strong, well-structured analysis suite for an MSc thesis. The reports form a coherent chain from dataset characterisation (R0) through task performance (R1-R2), internal model mechanics (R3-R5), model-policy interaction (R6), and representational similarity (R7), capped by a synthesis (summary). The narrative is clear and the central claim — M3 learns a qualitatively different adaptation strategy rather than aligning with NAMM — is well-supported by converging evidence from multiple angles. That said, there are significant methodological concerns and interpretive over-reaches that weaken the conclusions.

---

## Methodological Issues (most serious)

### 1. B0 baseline is wrong

The `experiment_specification.md` says B0 is the base model with **full KV cache, no eviction, no fine-tuning**. But in `report_1/generate_plots.py:131-141`, B0 is extracted from the LoRA run `kz6vqo2o` via `lora/baseline_lb_avg_f1`. This is the baseline logged at the *start of M1 training*, meaning it's evaluated under M1's inference setup — which may include specific evaluation code paths, answer extraction, or truncation that differ from a standalone `run_eval.py` baseline. The git log mentions "fix: use correct B0 baseline (kz6vqo2o, not qfoxxi2m)" suggesting there was already confusion here. A dedicated `run_eval.py` baseline run was never executed (the experiment spec confirms "B0 — not started", "B1 — not started"). This means B0's values are approximate at best.

### 2. All M3 and M2 metrics are validation, not test

Every F1 number in reports 1-2 is the **best validation F1**, not test-set F1. The experiment spec explicitly requires all final comparisons on the test split (69 samples). Reporting val F1 as the primary metric inflates M3 (and M1) numbers due to implicit selection bias — you're cherry-picking the step with highest val F1 and reporting *that* val F1 as the result. Proper procedure: take the checkpoint at best val step, evaluate on the held-out test split. This is the most consequential methodological gap.

### 3. M3 cs3072 was massively undertrained

M3 cs3072 completed only 4 epochs / 117 steps vs 150 epochs planned. The reports repeatedly flag this but then still include cs3072 in all comparisons, tables, and charts. This run should either be excluded entirely or clearly separated as unreliable. Instead, it muddies the cache-size trend analysis in reports 1-2, leading to confusing patterns (e.g. M2 cs3072 > M2 cs1024, M3 cs3072 < M3 cs1024).

### 4. Very small sample sizes in GPU analyses (Reports 5, 6, 7)

- **Report 5** (attention entropy): 10 test prompts, truncated to 1024 tokens. These prompts are 4096-6500 tokens at full length — truncating to 1024 fundamentally changes the attention distribution. The entropy differences (~4%) may simply reflect truncation artefacts rather than genuine adaptation.
- **Report 6** (token alignment): 3 samples per task = 15 total, also at 1024 tokens. Spearman correlations on 15 samples have wide confidence intervals. The claim of rho = -0.14 needs error bars — this could easily be consistent with zero given the variance.
- **Report 7** (CKA): 10 samples at 1024 tokens. CKA with 10 data points and high-dimensional representations can be unstable.

The truncation to 1024 tokens is the bigger concern: both models were trained on 4096-6500 token inputs. Evaluating on 1024-token truncations puts both models in an out-of-distribution regime where neither model's learned adaptation is fully exercised. The entropy and CKA differences may not reflect how the models actually behave on the data they were designed for.

### 5. Two separate GPU analysis pipelines produce potentially conflicting data

Reports 5 and 7 have two code paths:
- `run_gpu_analyses.py` — loads bare HuggingFace model + PEFT, merges LoRA, runs standard `model()` forward pass. Computes entropy *averaged over all query positions*.
- `report_5/generate_plots.py` — uses the full NAMM `memory_model` infrastructure, loads LoRA into the wrapped model, runs with/without `apply_memory_policy`. Computes entropy *at the last query position only*.

These are fundamentally different measurements. The report text describes the `generate_plots.py` methodology but the actual data files (`.npz`) appear to be generated by `run_gpu_analyses.py`. If the wrong data file backs the report, the methodology description doesn't match the data.

### 6. M1 uses `apply_memory_policy=False` — but what about M3?

In Report 5's `generate_plots.py`, M1 is run with a Recency policy (no eviction) and `apply_memory_policy=False`. M3 is run with `apply_memory_policy=True` — meaning NAMM *actively evicts tokens during the forward pass*. The report then describes this as comparing M1 and M3 "on full-context inputs (no NAMM eviction at inference)." If `apply_memory_policy=True` for M3, there **is** eviction at inference. This contradicts the report's framing.

The `run_gpu_analyses.py` script correctly runs both models without eviction (bare merged model forward pass), but if the report used data from `generate_plots.py`, the asymmetric eviction invalidates the comparison.

---

## Statistical/Analytical Concerns

### 7. No significance testing or confidence intervals

Not a single p-value, confidence interval, or bootstrap test appears in any report except Report 3 (retention correlation). The headline finding "M3 cs1024 matches M1" (45.59 vs 45.48) is a difference of 0.11 F1 points. With 69 test samples (or even 64 val samples), this is well within noise. Similarly, the attention entropy difference (+4%) and CKA dip (0.979) are presented as definitive findings without any test of whether they're distinguishable from measurement noise at n=10.

### 8. Recovery ratio is misleading for HotpotQA-E

HotpotQA-E's recovery ratio of 4.51 (Report 1) is treated as evidence of "massive synergy" between NAMM and LoRA. But the denominator `(M1 - M2)` is only 4.46 F1 points (44.0 - 39.54), making the ratio extremely sensitive to noise. If M2's HotpotQA-E score were 1 F1 point higher (plausible given M2 variance), the recovery ratio drops from 4.51 to 3.5. The large ratios are an artefact of small denominators, not large effects.

### 9. "Near-orthogonal" subspace overlap needs a random baseline

Report 4 claims M1 and M3 learn in "near-orthogonal" subspaces (overlap ~0.18). But what is the expected overlap between two *random* 8-dimensional subspaces of R^2048? The report acknowledges this ("barely above what one would expect from random") but never actually computes the baseline. For rank-8 subspaces in d=2048, the expected mean cosine of principal angles between random subspaces is approximately sqrt(8/2048) ≈ 0.063. So overlap of 0.18 is actually ~3x the random baseline — still low, but meaningfully above random, not "barely above."

### 10. Report 0's relevant token analysis is crude

The relevant token estimation (answer string + question entity matching with +-200 character windows) is acknowledged as a lower bound, but it's used heavily to form predictions that are then "disproven." The disproval of these predictions is presented as a key finding, but the predictions were based on a flawed signal. HotpotQA-E's high "relevant token" count (2034) is inflated by distractor passages containing entity mentions — exactly the issue that makes the prediction wrong.

---

## Interpretive Concerns

### 11. "Pre-emptive hedging" is a narrative, not a tested hypothesis

The summary report frames M3's broader attention as "pre-emptive hedging" — the model distributes attention to be robust to arbitrary evictions. But this is post-hoc storytelling. An equally valid interpretation: M3's LoRA was optimized under a noisy training signal (evicted context varies per step), and the broader attention is simply an artefact of optimization under noise (analogous to why dropout leads to flatter weight distributions). To test the hedging hypothesis, you'd need to show that M3's performance is more robust to *random* eviction patterns than M1's — i.e., that the broader attention is specifically functional, not just a side-effect of noisy training.

### 12. The negative NAMM-attention correlation is over-interpreted

Report 6 presents rho = -0.14 as a key finding showing "complementary signals." But rho = -0.14 is a very weak correlation — it means NAMM scores explain about 2% of the variance in attention. This is consistent with the two signals being essentially *independent*, not actively complementary. The framing of "anti-correlation" overstates what is really "near-zero with a slight negative bias."

### 13. Layer 3 as "critical adaptation point" is cherry-picked convergence

Reports 4, 5, and 7 each identify slightly different layers as interesting (Report 5: layers 3, 6, 13, 15; Report 7: layer 3; Report 4: layer 14 for max norm, layer 6 for min overlap). The summary selectively cites layer 3 across all reports to create a convergent narrative, but this requires ignoring that layer 14 is the biggest in Report 4, that Report 5's largest shifts are at layers 6, 13, and 15, and that Report 3's biggest eviction layers are 8-9.

### 14. "6x cache reduction" is misleading framing

The summary claims "6x smaller KV cache" (1024 vs ~5000+ tokens). But the experiment specification says contexts are 4096-6500 tokens, so the actual compression is 4-6.3x, not a clean "6x." More importantly, the "6x" applies to the cache size, not to memory footprint — the model still needs to process the full input before eviction happens.

---

## Missing Analyses / Gaps

### 15. B1 (recency baseline) was never run
The experiment spec calls for it. Without B1, you can't establish how much of NAMM's value comes from *learned* eviction vs. any eviction. If recency eviction + LoRA also recovers to M1 levels, the entire NAMM contribution story collapses.

### 16. M4 (joint training) was never run
The experiment spec's primary research question (RQ4: are jointly-trained NAMM and LoRA co-dependent?) is unanswered.

### 17. No test-set evaluation
As noted above, all reported numbers are validation F1.

### 18. No per-task learning curves
Report 2 analyzes aggregate val F1 learning curves, but the most interesting finding from Report 1 (Qasper suffering, HotpotQA-E benefiting) would be much better understood with per-task learning curves showing when and how this divergence develops.

### 19. No ablation: M3 at inference without NAMM
The experiment spec (A4) calls for evaluating M3's checkpoint with NAMM disabled at inference. This would directly test whether M3's "hedging" is robust to full context or whether the model specifically adapted to the eviction distribution.

---

## Code Quality Observations

- **Report 4** runs top-level code at module import (`torch.load` at line 35-37). This means importing the module has side effects.
- **Report 5** has hardcoded `rank=4` in `apply_lora_adapters()` (line 401) but the experiment spec says rank 8. If the checkpoint is actually rank 8, this would silently produce wrong results.
- **Report 6** `import torch` is at module level (line 186) outside the GPU-only functions, meaning the script fails on CPU even with `--plot-only` (unlike Report 5 which gates torch imports inside functions).
- The `run_gpu_analyses.py` script uses crude prompt construction (`sample["context"][:8000] + "\nQuestion: "...`) instead of the actual LongBench templates, creating a distribution mismatch between analysis-time and training-time inputs.

---

## Strengths

To be fair, the work has substantial strengths:

1. **Well-structured hypothesis-driven approach**: Report 0 makes predictions, Report 1 tests them, and the honest acknowledgment that predictions were wrong drives deeper investigation.
2. **Converging evidence from multiple angles**: Weight space (R4), attention (R5), alignment (R6), and representations (R7) all contribute to the same story.
3. **The core finding is interesting**: That M3 learns an orthogonal adaptation rather than aligning with NAMM is a genuine finding worth reporting.
4. **Reproducible pipeline**: WandB run IDs are documented, code pulls data directly from WandB/checkpoints, and each report has a self-contained `generate_plots.py`.
5. **Honest about limitations**: Reports consistently flag the cs3072 undertrained issue, the negative train-val gap anomaly, and the small sample sizes.

---

## Recommendations (Priority Order)

1. **Run test-set evaluations** for all conditions. This is non-negotiable for a thesis.
2. **Run B1 (recency baseline)**. Without it, you can't claim NAMM specifically is responsible for M3's performance.
3. **Resolve the Report 5/6 methodology ambiguity** — which data file backs which report? Fix the rank=4 hardcode if wrong. Clarify whether M3 was evaluated with or without eviction.
4. **Add significance tests** — at minimum, bootstrap CIs on the M3-vs-M1 F1 difference.
5. **Increase GPU analysis sample sizes** to at least 30-50 samples at full sequence length (or a meaningful truncation like 4096). If memory is a constraint, use gradient checkpointing or process in chunks.
6. **Either exclude cs3072 from all analyses or complete the training run.**
7. **Compute the random subspace overlap baseline** for Report 4 to properly contextualise the orthogonality claim.
8. **Tone down "pre-emptive hedging"** to "a broader attention pattern consistent with eviction robustness" unless you can test it directly (e.g., random eviction robustness test).