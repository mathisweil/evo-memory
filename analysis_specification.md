# evo-memory — Analysis Specification

**Core question:** Does fine-tuning with NAMM active (M3) cause the model to adapt to the distribution shift introduced by KV-cache eviction, and if so, how?

We compare three conditions:
- **M1** — LoRA fine-tuned with full context (no eviction)
- **M2** — NAMM eviction policy trained alone (no LoRA, frozen LLM)
- **M3** — LoRA fine-tuned with frozen NAMM active during training

> **Naming note:** What is called M3 in this specification is labelled "M4" in `results/main_table_5t/`. See `experiment_specification.md` for the full naming warning.

---

## Analysis Status

| Section | Topic                         | Status | Report                  |
| ------- | ----------------------------- | ------ | ----------------------- |
| §0      | Dataset characterisation      | DONE   | `report_0/_report.md`   |
| §1      | Per-task sensitivity          | DONE   | `report_1/_report.md`   |
| §2      | Adaptation rate               | DONE   | `report_2/_report.md`   |
| §3      | Per-layer retention           | DONE   | `report_3/_report.md`   |
| §4      | LoRA weight comparison        | DONE   | `report_4/_report.md`   |
| §5      | Attention entropy             | DONE   | `report_5/_report.md`   |
| §6      | Token importance alignment    | DONE   | `report_6/_report.md`   |
| §7      | CKA representation similarity | DONE   | `report_7/_report.md`   |
| §8      | Probing for residual knowledge| DONE   | `report_8/_report.md`   |
| §9      | Gradient flow attribution     | DONE   | `report_9/_report.md`   |

### Additional conditions not in the original spec

The following conditions have been evaluated but were not covered in the original three-condition design:

- **B1** (recency baseline) — now evaluated.
- **A4** (NAMM disabled at eval) — now evaluated on M3 checkpoints (not M4 joint as originally planned).
- **Truncation baselines** (Trunc/plain, Trunc/lora_m1) — new conditions added during evaluation.
- **M1_recency** — attempted but broken (all zeros); not usable.

### Attention mask bug and maskfix reruns

A critical attention mask bug was discovered in the NAMM split-processing loop (see `scripts/diagnose_attention_mask.py`). The attention mask grows with cumulative input length rather than tracking the actual post-eviction KV cache size. From chunk 9 onward (~2300 tokens), attention collapses to uniform 1/N. This bug exists in the original Sakana AI NAMM codebase and affects all published results.

Maskfix reruns retrain M2 and M3 with the corrected attention mask:
- **M2 maskfix cs1024** (`z5bo4n8k`): finished, val F1 14.90 (worse than buggy 27.90)
- **M2 maskfix cs2048** (`jip3a3dm`): running
- **M3 maskfix cs1024** (`h0bzg6on`): running, val F1 52.06 at step 260 (exceeds buggy 45.59 and M1 45.48)

Checkpoints available locally (`experiment_artifacts/gcs/M2_cs1024_maskfix/`, `M3_cs1024_maskfix/`) and backed up to GCS (`gs://statistical-nlp/evo-memory/checkpoints_backup_20260414/`).

Maskfix analyses completed (all §0–§9):
- §0: No rerun needed (dataset characterisation)
- §1–§3: Rewritten from WandB data
- §4, §5, §7: GPU analyses using merged LoRA weights (`generate_data_4_5_7.py`)
- §6, §8, §9: GPU analyses using full NAMM infrastructure (`generate_data_6_8_9.py`)

Each report's `_report.md` now contains the maskfix (corrected) results as the primary data. Previous buggy versions are available in git history.

---

## 0 · Dataset Characterisation and Performance Hypotheses

**Question:** What are the structural differences between the 5 tasks, and what performance patterns should we expect under each condition before looking at results?

**Method:** For each task in the 5-task LongBench QA subset (qasper, 2wikimqa, qasper_e, hotpotqa_e, 2wikimqa_e), characterise:

1. **Task structure:** What kind of reasoning is required — localised passage retrieval, multi-hop reasoning across documents, or comparison-based reasoning?
2. **Answer characteristics:** Short factoid, extractive span, yes/no, or free-form? Average answer length.
3. **Information distribution:** Is the answer-relevant information concentrated in one passage or distributed across multiple passages? This directly predicts eviction sensitivity.
4. **Eviction impact:** At cache_size=1024 with contexts of 4096–6500 tokens, roughly 75–85% of tokens are evicted. Which task types lose critical information?

Form predictions for the relative task ranking under each condition (B0, M1, M2, M3) based purely on task characteristics, before comparing against actual results in §1.

**Hypothesis:** Tasks with localised information (Qasper — answer in a specific paper section) should be less sensitive to eviction than multi-hop tasks (2WikiMQA, HotpotQA — require combining facts from multiple passages). Under M3 fine-tuning, recovery should be easier for localised tasks since the model can learn to attend to the retained passage.

**Data needed:** Dataset metadata, prompt templates, sample counts per split.

**Effort:** Low — dataset inspection only.

**References:**
- Bai et al. (2023). "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding." *ACL 2024*.
- Dasigi et al. (2021). "A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers." *NAACL 2021*. Introduces the Qasper dataset.
- Yang et al. (2018). "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering." *EMNLP 2018*.
- Ho et al. (2020). "Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps." *COLING 2020*. Introduces 2WikiMultihopQA.

---

## 1 · Per-Task Eviction Sensitivity

**Question:** Which tasks are most affected by KV-cache eviction, and does M3 fine-tuning recover the lost performance uniformly or selectively?

**Method:** Compute per-task eviction sensitivity as the relative F1 drop between M1 (full context) and M3 (evicted context):

```
sensitivity(task) = (M1_F1 - M3_F1) / M1_F1
```

A high sensitivity score means the task relies on information that NAMM tends to evict. Cross-reference with task characteristics: average answer position within the context, average context length, and whether the task requires specific detail retrieval (e.g. Qasper scientific QA) vs high-level reasoning (e.g. 2WikiMQA).

**Hypothesis:** Tasks requiring precise detail retrieval from specific locations (Qasper) will show higher eviction sensitivity than tasks solvable from distributed contextual cues (2WikiMQA, HotpotQA). M3 fine-tuning should partially close the gap, but more so for the distributed-cue tasks.

**Data needed:** Per-task best val F1 for M1 and M3 at each cache size (already in wandb).

> **Update:** The report now uses maskfix validation data. Key finding: M3 maskfix (val F1 52.06) substantially exceeds M1 (val F1 45.48), counter to the hypothesis that eviction would uniformly hurt performance. Buggy test-set results (M3: 23.52, M1: 31.14) showed M3 underperforming, but this was confounded by the attention mask bug.

**Effort:** Low — analysis of existing metrics only.

**References:**
- Bai et al. (2023). "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding." *ACL 2024*. Introduces the task suite and characterises task difficulty as a function of context dependency.
- Kamradt (2023). "Needle in a Haystack — Pressure Testing LLMs." Demonstrates that retrieval accuracy depends on the position of target information within long contexts, directly relevant to understanding which tokens NAMM evicts and why some tasks suffer more.

---

## 2 · Adaptation Rate and Learning Efficiency

**Question:** Does M3 learn at a different rate than M1? Does the information bottleneck imposed by eviction slow convergence, or does it act as implicit regularisation?

**Method:** From the validation F1 trajectories of M1 and M3:

1. **Normalised improvement curve:** Plot `(val_F1(step) - baseline_F1) / (best_val_F1 - baseline_F1)` for M1 and M3. This normalises both curves to [0, 1] and isolates the learning dynamics from the final performance level.
2. **Steps to X% performance:** Measure the number of gradient steps for each condition to reach 50%, 75%, and 90% of its own best validation F1.
3. **Overfitting gap:** Plot `train_F1 - val_F1` over training. If M3 shows a consistently smaller gap, eviction acts as a form of regularisation by forcing the model to generalise from partial context.

**Hypothesis:** M3 may converge more slowly initially (due to the harder optimisation landscape with missing tokens) but show less overfitting. If the information bottleneck acts as regularisation analogous to dropout, we may see M3 achieve comparable or better generalisation despite lower training F1.

**Data needed:** Per-step val F1 and train F1 for M1 and M3 (already in wandb).

**Effort:** Low — plotting and simple metric computation from existing data.

**References:**
- Srivastava et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *JMLR 15(56)*. Establishes the theoretical framework for how stochastic information removal during training acts as regularisation. KV-cache eviction can be viewed as a structured, learned form of dropout over context tokens.
- Wei et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS 2022*. While not directly about eviction, demonstrates that information bottlenecks (forcing models to reason step-by-step rather than attending to everything) can improve generalisation — analogous to how eviction forces the model to reason from partial context.

---

## 3 · Per-Layer Retention Pattern Analysis

**Question:** Does NAMM evict tokens uniformly across layers, or do different layers retain different amounts of context? Does the retention pattern correlate with learning progress?

**Method:** The M3 runs log `retention/layer_{i}` at each training step, measuring the fraction of input tokens surviving eviction at each of the 16 transformer layers.

1. **Layer retention profile:** Plot average retention ratio per layer across training. This reveals whether NAMM applies uniform or layer-specific eviction pressure.
2. **Retention vs F1 correlation:** At each eval step, correlate the average retention ratio with validation F1. If higher retention leads to better F1, the model hasn't fully adapted to eviction. If the correlation weakens over training, the model is learning to compensate.
3. **Retention dynamics over training:** Plot per-layer retention as a heatmap over training steps. Look for patterns where retention shifts between layers as LoRA adapts.

**Hypothesis:** Early layers (which capture more local/syntactic features) may retain more tokens than later layers (which capture more global/semantic features). As M3 training progresses, the model may become less sensitive to retention ratio, suggesting adaptation.

**Data needed:** `retention/layer_{i}` metrics from M3 wandb runs (already logged for cs1024, cs2048, cs3072).

**Effort:** Low — analysis of existing logged metrics.

**References:**
- Michel et al. (2019). "Are Sixteen Heads Really Better than One?" *NeurIPS 2019*. Demonstrates that different attention heads and layers have different levels of importance, with many being prunable — directly relevant to understanding layer-specific eviction patterns.
- Voita et al. (2019). "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned." *ACL 2019*. Shows that attention heads specialise for different functions (positional, syntactic, rare-word), suggesting eviction may disproportionately affect certain head types.

---

## 4 · LoRA Weight Comparison (M1 vs M3)

**Question:** Does the LoRA adapter learn a fundamentally different transformation when trained under eviction? Or does it converge to a similar subspace regardless?

**Method:** Load the best M1 and M3 checkpoints and compare the learned LoRA A and B matrices directly.

1. **Per-layer weight magnitude:** Compute the Frobenius norm `||B @ A||_F` of the effective LoRA update at each layer for M1 and M3. Layers where M3 has a significantly larger norm are compensating more for eviction.
2. **Singular value spectrum:** Compute the SVD of the effective LoRA update `B @ A` for both M1 and M3. Compare the distribution of singular values — if M3 uses more of its rank-8 capacity (flatter spectrum), it needs more expressivity to handle eviction.
3. **Subspace overlap:** Compute the principal angles between the column spaces of M1's and M3's LoRA updates at each layer using `cos(theta) = sigma(U_M1^T @ U_M3)`. High overlap means eviction doesn't change *what* the adapter learns, just *how much*.

**Hypothesis:** M3 will show larger LoRA norms in later layers (where global context aggregation is more affected by eviction) and lower subspace overlap in those same layers, indicating that eviction forces a qualitatively different adaptation.

**Data needed:** M1 and M3 checkpoint files (already downloaded to `experiment_artifacts/gcs/`).

**Effort:** Medium — requires loading checkpoints and computing matrix operations, but no inference.

**References:**
- Hu et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*. Introduces LoRA and analyses the rank structure of fine-tuning updates, showing that task-specific adaptations occupy low-rank subspaces. Comparing M1 vs M3 subspaces directly extends this analysis to the eviction setting.
- Aghajanyan et al. (2021). "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." *ACL 2021*. Shows that fine-tuning updates have low intrinsic dimensionality. If eviction increases the intrinsic dimensionality (M3 needs more of its rank budget), this suggests the eviction distribution shift is non-trivial to adapt to.

---

## 5 · Attention Entropy Shift Under Eviction

**Question:** Does the model's attention distribution change when tokens are evicted? Does M3 fine-tuning cause attention to sharpen onto retained tokens?

**Method:** Run M1 and M3 checkpoints on the same test inputs and extract attention weight matrices at each layer and head.

1. **Attention entropy:** For each head at each layer, compute the Shannon entropy of the attention distribution: `H = -sum(a_i * log(a_i))`. Higher entropy indicates more distributed attention. Compare M1 entropy (full context) vs M3 entropy (evicted context).
2. **Attention mass on would-be-evicted tokens:** Using M1's checkpoint, compute how much attention goes to tokens that M2's NAMM would have evicted. This quantifies the "information at risk" — the fraction of attention that M3 must learn to redirect.
3. **Attention sink analysis:** Measure the fraction of total attention captured by the first few tokens (BOS, system prompt tokens) in M1 vs M3. If M3 shows stronger attention sinks, it may be using early tokens as a "memory buffer" for information from evicted positions.

**Hypothesis:** M3 will show lower attention entropy (more focused) than M1, particularly in later layers. The attention mass on would-be-evicted tokens in M1 quantifies the upper bound on information loss, and M3's ability to recover F1 suggests it redirects this attention to retained tokens or early sinks.

**Data needed:** Inference pass with attention output on test set using M1 and M3 checkpoints.

**Effort:** Medium — requires running inference with `output_attentions=True` and saving attention matrices.

**References:**
- Xiao et al. (2024). "Efficient Streaming Language Models with Attention Sinks." *ICLR 2024*. Demonstrates that initial tokens act as "attention sinks" that stabilise generation even when most of the KV cache is evicted. Directly relevant to understanding whether M3 amplifies this phenomenon.
- Clark et al. (2019). "What Does BERT Look At? An Analysis of BERT's Attention." *BlackboxNLP 2019*. Establishes methodology for interpreting attention patterns in transformers, including entropy analysis and attention head specialisation.
- Abnar & Zuidema (2020). "Quantifying Attention Flow in Transformers." *ACL 2020*. Introduces attention rollout for tracing information flow through layers, useful for understanding how eviction at early layers propagates to later computations.

---

## 6 · Token Importance Alignment (NAMM Scores vs Attention)

**Question:** Does NAMM's learned scoring function agree with the model's own attention-based importance? Does M3 fine-tuning improve this alignment?

**Method:** The NAMM BAM scoring network assigns an importance score to each token at each eviction step. During inference, extract both NAMM scores and the LLM's attention weights:

1. **Score-attention correlation:** For each eviction step, compute the Spearman rank correlation between NAMM's token scores and the average attention each token receives from subsequent query positions. High correlation means NAMM is evicting tokens the model doesn't attend to anyway.
2. **Alignment shift M1→M3:** Compute the same attention-based importance using M1 weights and M3 weights. If M3's attention aligns more closely with NAMM scores, the model has learned to "agree" with the eviction policy.
3. **Eviction regret:** For each evicted token, measure the total attention it would have received (from M1) in subsequent positions. High-regret evictions identify cases where NAMM removes tokens the model actually needs. Compare regret distributions across tasks to explain per-task sensitivity differences.

**Hypothesis:** NAMM scores will correlate moderately with attention weights (since NAMM is trained on task performance, not attention alignment). M3 fine-tuning should increase this correlation, as the model learns to rely more on tokens that NAMM retains.

**Data needed:** Inference pass with NAMM active, extracting both NAMM scores and attention weights. Requires hooking into `namm/policy/deep_scoring_bam.py` scoring forward pass.

**Effort:** Medium — requires custom inference script with hooks into NAMM scoring.

**References:**
- Kim et al. (2022). "Learned Token Pruning for Transformers." *KDD 2022*. Studies the alignment between learned pruning decisions and attention-based importance in vision transformers. Provides methodology for measuring score-attention correlation and pruning regret.
- Sakana AI / Munkhdalai et al. (2024). "Neural Attention Memory Models." The original NAMM paper, which uses attention-based spectrograms for scoring. Extending their analysis to measure alignment with the fine-tuned model's attention is a natural follow-up.

---

## 7 · Representation Similarity (CKA)

**Question:** How different are the internal representations of M1 vs M3? Does eviction push the model into a fundamentally different representation space, or does LoRA find a way to preserve similar representations despite missing tokens?

**Method:** Apply Centered Kernel Alignment (CKA) to compare hidden state representations between M1 and M3 at each layer:

1. **Layer-wise CKA:** For the same set of test inputs, extract hidden states at each layer from both M1 (full context) and M3 (evicted context). Compute linear CKA between the two representation matrices at each layer. CKA = 1 means identical representations (up to linear transform); CKA = 0 means completely unrelated.
2. **CKA vs eviction intensity:** Plot CKA per layer against the average retention ratio at that layer. Layers with more aggressive eviction should show lower CKA (more divergent representations).
3. **CKA heatmap (cross-layer):** Compute CKA between every pair of (M1 layer i, M3 layer j). Off-diagonal peaks would suggest that M3 has learned to shift certain computations to different layers to compensate for eviction.

**Hypothesis:** Early layers (before eviction has cumulative effects) will show high CKA. Later layers will show lower CKA, indicating that the model's high-level representations diverge when context is partially evicted. The degree of CKA drop should correlate inversely with M3's ability to recover M1-level F1.

**Data needed:** Inference pass on test set with hidden state extraction at all 16 layers for both M1 and M3 checkpoints.

**Effort:** Medium — requires inference and CKA computation, but established libraries exist.

**References:**
- Kornblith et al. (2019). "Similarity of Neural Network Representations Revisited." *ICML 2019*. Introduces CKA as a reliable metric for comparing neural network representations across layers, models, and training stages. Preferred over older metrics (CCA, SVCCA) for its robustness to dimensionality and sample size.
- Raghu et al. (2017). "SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability." *NeurIPS 2017*. Earlier representation comparison method; CKA supersedes it but the analysis framework (comparing representations across training conditions) is directly applicable.
- Neyshabur et al. (2020). "What is being transferred in transfer learning?" *NeurIPS 2020*. Uses representation similarity to understand what fine-tuning changes and preserves, providing a template for our M1-vs-M3 comparison.

---

## 8 · Probing for Residual Knowledge of Evicted Content

**Question:** When NAMM evicts tokens, does the model still retain information about them in its hidden states? Has M3 learned to compress evicted information into the representations of retained tokens?

**Method:** Train lightweight linear probes on hidden states to predict facts that were present in evicted tokens:

1. **Probe design:** For each test example, identify tokens that NAMM evicts. Formulate binary classification tasks: given the hidden state of a retained token at layer L, can a linear probe predict whether a specific evicted fact (e.g. a named entity, a date) was present in the original context?
2. **M1 vs M3 probe accuracy:** Train probes on M1 hidden states (full context, all information present) as an upper bound. Train probes on M3 hidden states (post-eviction). If M3 probes approach M1 accuracy, the model has learned to compress evicted information.
3. **Layer-wise probing:** Apply probes at each layer to identify where residual knowledge is concentrated. If it appears in later layers of M3 but not M1, this suggests the LoRA adapter is actively compressing information forward.

**Hypothesis:** M1 probes will succeed easily (information is present in full). M3 probes will show intermediate accuracy — better than chance (some compression occurred) but worse than M1 (not all information survives eviction). Accuracy will be higher in later layers if LoRA is actively compensating.

**Data needed:** Inference pass with hidden state extraction, plus construction of probe labels from evicted tokens.

**Effort:** High — requires designing probe tasks, extracting hidden states, and training probe classifiers.

**References:**
- Belinkov (2022). "Probing Classifiers: Promises, Shortcomings, and Advances." *Computational Linguistics 48(1)*. Comprehensive survey of probing methodology, including best practices for control tasks and avoiding confounds. Essential reading for designing meaningful probes.
- Voita et al. (2019). "The Bottom-up Evolution of Representations in the Transformer: A Study with Machine Translation and Language Modeling Heads." *EMNLP 2019*. Uses probing to track how information is transformed across layers, directly applicable to understanding where evicted information might be compressed.
- Hewitt & Liang (2019). "Designing and Interpreting Probes with Control Tasks." *EMNLP 2019*. Introduces control tasks to ensure probes measure genuine encoding rather than probe capacity. Critical for avoiding false positives when probing for residual knowledge.

---

## 9 · Gradient Flow and Loss Attribution Under Eviction

**Question:** Does the LoRA adapter respond to eviction events during training? When NAMM removes critical tokens, do the resulting loss spikes cause larger gradient updates to the LoRA parameters?

**Method:** Instrument M3 training to log fine-grained loss and gradient information:

1. **Per-position loss stratification:** During each training step, record the cross-entropy loss at each answer token position. Partition steps into "high eviction" (retention < median) and "low eviction" (retention > median). Compare the loss distributions — higher loss under heavy eviction would confirm that eviction causes measurable information loss during training.
2. **Gradient magnitude analysis:** Log the L2 norm of the gradient on LoRA parameters at each step, alongside the retention ratio. If gradient norms spike when retention drops, the model is actively responding to eviction events rather than ignoring them.
3. **LoRA update direction consistency:** Compute the cosine similarity between consecutive LoRA gradient directions. If this is lower during high-eviction steps, the eviction introduces noise into the optimisation — the model receives inconsistent gradient signals depending on which tokens are evicted.

**Hypothesis:** High-eviction steps will show higher loss (more information missing) and larger gradient norms (stronger adaptation signal). Over training, the loss gap between high- and low-eviction steps should narrow if the model successfully adapts to eviction.

**Data needed:** Custom training hooks to log per-position loss and gradient norms. Requires modifying `grad_lora_finetuning/trainer.py`.

**Effort:** High — requires training instrumentation and potentially rerunning training with additional logging.

**References:**
- Koh & Liang (2017). "Understanding Black-box Predictions via Influence Functions." *ICML 2017*. Provides the theoretical framework for attributing model behaviour to specific training examples. While full influence functions are expensive, the gradient-based attribution approach (measuring how much each training step's gradient contributes to the final model) is directly applicable.
- Chen et al. (2018). "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks." *ICML 2018*. Uses gradient magnitude to understand task difficulty and adaptation in multi-task learning. The analogy here is that "high eviction" and "low eviction" steps are like different tasks with different difficulty, and gradient norms reveal how the model allocates learning capacity.
- Fort et al. (2020). "Deep Learning versus Kernel Learning: An Empirical Study of Loss Landscape Geometry and the Time Evolution of the Neural Tangent Kernel." *NeurIPS 2020*. Analyses how gradient direction consistency relates to optimisation dynamics, relevant to understanding whether eviction introduces noise into the loss landscape.

---

## Recommended Execution Order

| Priority | Analysis                              | Section | Effort | New code?                |
| -------- | ------------------------------------- | ------- | ------ | ------------------------ |
| 1        | Per-task eviction sensitivity         | §1      | Low    | No                       |
| 2        | Adaptation rate / learning efficiency | §2      | Low    | Minimal                  |
| 3        | Per-layer retention patterns          | §3      | Low    | Minimal                  |
| 4        | LoRA weight comparison                | §4      | Medium | Checkpoint loading       |
| 5        | Attention entropy shift               | §5      | Medium | Inference hooks          |
| 6        | Token importance alignment            | §6      | Medium | NAMM hooks               |
| 7        | CKA representation similarity         | §7      | Medium | Inference + CKA lib      |
| 8        | Probing for residual knowledge        | §8      | High   | Probe training           |
| 9        | Gradient flow attribution             | §9      | High   | Training instrumentation |

Analyses 1–3 can be completed immediately from existing wandb data. Analyses 4–7 each require one or two inference passes over the test set. Analyses 8–9 require new training or probe-training runs.

---

## 10 · Cross-Report Synthesis

`analysis/_summary_report.md` synthesises findings across all completed reports (§0–§9), drawing out cross-cutting themes and the overall narrative. `analysis/_meta-analysis.md` provides an independent critique of the analysis pipeline, identifying limitations, potential confounds, and directions for further work.
