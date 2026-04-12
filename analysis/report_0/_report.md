# Report 0: Dataset Characteristics and Performance Hypotheses

> **TL;DR:** The 5 tasks split into two families: Qasper tasks (scientific paper QA, diverse answer types, ~192 relevant tokens per prompt in ~1.2 regions) and multi-hop tasks (2WikiMQA, HotpotQA — short factoid answers, ~1100-2034 relevant tokens across 5-9 regions). At cache=1024, Qasper's relevant tokens fit easily (97% survival) while HotpotQA-E's exceed the budget (only 60% survival). We predict multi-hop tasks will suffer most from eviction, while Qasper's localised answers are easier to preserve. These predictions are tested against actual results in Report 1 — where they turn out to be partially wrong.

> **Purpose:** Characterise the five LongBench QA tasks used in our experiments and form hypotheses about expected relative performance under each experimental condition — *before* examining any results.

---

## 1. Overview of Tasks

This project fine-tunes LLaMA 3.2-1B-Instruct on a 5-task LongBench QA subset. All tasks require the model to read a long context passage and answer a question about it. The tasks differ in source dataset, reasoning requirements, and answer format.

| Task | Source Dataset | Domain | Reasoning Type |
|------|---------------|--------|----------------|
| Qasper | Qasper (Dasigi et al., 2021) | Scientific papers | Single-passage retrieval |
| 2WikiMQA | 2WikiMultihopQA (Ho et al., 2020) | Wikipedia passages | Multi-hop reasoning |
| Qasper-E | Qasper, LongBench-E variant | Scientific papers | Single-passage retrieval |
| HotpotQA-E | HotpotQA (Yang et al., 2018), LongBench-E | Wikipedia passages | Multi-hop (bridge/comparison) |
| 2WikiMQA-E | 2WikiMultihopQA, LongBench-E variant | Wikipedia passages | Multi-hop reasoning |

All tasks are evaluated using **token-level F1 score** (qa_f1_score).

---

## 2. Task Descriptions

### 2.1 Qasper (lb/qasper)

**Source:** Qasper (Dasigi et al., 2021) via LongBench (Bai et al., 2023).

The context is a full scientific paper (abstract, introduction, methods, results, etc.). Questions are written by NLP researchers who read only the paper's title and abstract. This means questions target specific sections of the paper but the questioner does not know exactly where the answer lies.

**Answer types are diverse:**
- Short factoid answers (18%)
- Phrase-length answers (33%)
- Sentence-length answers (31%)
- Yes/no answers (11%)
- "Unanswerable" (8%)

The mean answer length is **9.5 words**, the longest among all five tasks. This diversity makes Qasper intrinsically harder for F1 scoring: the model must produce the right format (yes/no, unanswerable, or a precise extractive/abstractive answer).

**Information locality:** Generally **localised** — the answer typically comes from a specific section or paragraph of the paper. However, some questions require synthesising information from multiple sections.

### 2.2 2WikiMQA (lb/2wikimqa)

**Source:** 2WikiMultihopQA (Ho et al., 2020) via LongBench.

The context consists of multiple Wikipedia passages concatenated together. Questions require multi-hop reasoning: the model must find information in one passage, use it to identify a relevant entity, then find the answer in a different passage.

**Answer types are concentrated:**
- Short factoid (1-3 words): 82%
- Phrase: 14%
- Yes/no: 4%

Mean answer length is **2.2 words** — the shortest among all tasks. Questions typically ask about relationships between entities (e.g., "Where was the wife of Francis I Rakoczi born?").

**Information locality:** **Distributed** — answering requires combining facts from at least two different passages. The model must perform entity linking across passages.

### 2.3 Qasper-E (lb/qasper_e)

**Source:** Qasper dataset via LongBench-E (extended version with different length distributions).

Same task type as Qasper but drawn from a different sample of the Qasper dataset, with a broader length distribution. The prompt template is identical. After filtering to the 4096-6500 token range, the characteristics are very similar to base Qasper:
- Answer type distribution: 26% short factoid, 29% phrase, 28% sentence+, 11% yes/no, 6% unanswerable
- Mean answer length: **8.7 words**

**Information locality:** **Localised** — same as base Qasper.

### 2.4 HotpotQA-E (lb/hotpotqa_e)

**Source:** HotpotQA (Yang et al., 2018) via LongBench-E.

HotpotQA contains multi-hop questions requiring reasoning over two Wikipedia passages. The original dataset includes "bridge" questions (finding a bridge entity connecting two passages) and "comparison" questions (comparing attributes of two entities).

**Answer types:**
- Short factoid: 69%
- Phrase: 16%
- Yes/no: 13%

Mean answer length: **2.5 words**. The yes/no proportion (13%) is notably higher than 2WikiMQA (4%), reflecting HotpotQA's comparison questions.

**Information locality:** **Distributed** — specifically requires exactly 2 supporting passages. The context contains many distractor passages alongside the 2 gold passages.

### 2.5 2WikiMQA-E (lb/2wikimqa_e)

**Source:** 2WikiMultihopQA via LongBench-E.

Same task type as base 2WikiMQA but from the extended LongBench-E collection. After filtering:
- Short factoid: 74%
- Phrase: 18%
- Yes/no: 8%

Mean answer length: **2.4 words**. Very similar profile to base 2WikiMQA.

**Information locality:** **Distributed** — same multi-hop reasoning requirement.

---

## 3. Prompt Templates

All five tasks use prompt templates from LongBench. The templates can be grouped into two families:

### Scientific Paper QA (Qasper, Qasper-E):
```
You are given a scientific article and a question. Answer the question
as concisely as you can, using a single phrase or sentence if possible.
If the question cannot be answered based on the information in the
article, write "unanswerable". If the question is a yes/no question,
answer "yes", "no", or "unanswerable". Do not provide any explanation.

Article: {context}

Answer the question based on the above article as concisely as you can,
using a single phrase or sentence if possible. [...]

Question: {input}

Answer:
```

### Multi-hop Passage QA (2WikiMQA, HotpotQA-E, 2WikiMQA-E):
```
Answer the question based on the given passages. Only give me the answer
and do not output any other words.

The following are given passages.
{context}

Answer the question based on the given passages. Only give me the answer
and do not output any other words.

Question: {input}
Answer:
```

**Key differences:**
- The Qasper template is more detailed, explicitly mentioning unanswerable and yes/no handling.
- The multi-hop template is minimal, simply asking for "the answer" with no guidance on format.
- Both templates repeat the instruction after the context, which is helpful for models with limited attention — the instruction is present at both the start and end of the prompt.

See `prompt_templates.png` for the full templates.

---

## 4. Sample Counts and Data Splits

Filtering applied (matching experiment configuration):
- `min_conditioning_length`: 4096 tokens
- `max_conditioning_length`: 6500 tokens
- `max_answer_tokens`: 64
- Splits: `train_frac=0.7`, `val_frac=0.15`, `split_seed=42`

**Note:** The exact counts below use word-based approximation (1.3 tokens/word) matching the `init_tasks()` code. Tokenizer-based filtering in `apply_train_val_test_split()` gives slightly different numbers. The reported total from experiments is 306/64/69 = 439; our word-based approximation yields 320/67/74 = 461.

| Task | Raw | After Length Filter | Eligible (min 4096 tok) | Train | Val | Test |
|------|-----|-------------------|------------------------|-------|-----|------|
| Qasper | 200 | 180 | 95 | 66 | 14 | 15 |
| 2WikiMQA | 200 | 138 | 96 | 67 | 14 | 15 |
| Qasper-E | 224 | 178 | 124 | 86 | 18 | 20 |
| HotpotQA-E | 300 | 111 | 62 | 43 | 9 | 10 |
| 2WikiMQA-E | 300 | 142 | 84 | 58 | 12 | 14 |
| **Total** | **1224** | **749** | **461** | **320** | **67** | **74** |

**Key observations:**
- HotpotQA-E has the fewest eligible samples (62) because its raw contexts tend to be longer (mean 6658 words), and most are filtered out by the 6500-token upper bound.
- Qasper-E contributes the most samples (124), providing good coverage of scientific paper QA.
- The filtering is aggressive: only 37% of raw samples survive all filters.

---

## 5. Context and Answer Length Analysis

### 5.1 Context Lengths (Eligible Samples)

| Task | Mean (words) | Median | Min | Max | Std |
|------|-------------|--------|-----|-----|-----|
| Qasper | 3987 | 3862 | 3164 | 4958 | 508 |
| 2WikiMQA | 4037 | 3984 | 3155 | 4966 | 534 |
| Qasper-E | 4118 | 4171 | 3158 | 4964 | 484 |
| HotpotQA-E | 3737 | 3710 | 3177 | 4751 | 413 |
| 2WikiMQA-E | 3989 | 3998 | 3155 | 4993 | 554 |

After filtering, all tasks have similar context length distributions (mean ~3800-4100 words, ~4900-5350 tokens). HotpotQA-E is slightly shorter on average. See `length_distributions.png`.

### 5.2 Answer Lengths

| Task | Mean (words) | Median | Max |
|------|-------------|--------|-----|
| Qasper | 9.5 | 6 | 43 |
| 2WikiMQA | 2.2 | 2 | 7 |
| Qasper-E | 8.7 | 5 | 46 |
| HotpotQA-E | 2.5 | 2 | 11 |
| 2WikiMQA-E | 2.4 | 2 | 9 |

There is a stark divide: **Qasper tasks have much longer, more variable answers** (mean ~9 words, range 1-46) while **multi-hop tasks have very short answers** (mean ~2.3 words, mostly 1-3 words). See `answer_types.png`.

---

## 6. Information Distribution Analysis

This section analyses where answer-relevant information is located within the context, which is critical for predicting eviction sensitivity.

### 6.1 Token Eviction Rates

At cache_size=1024 with contexts of 4096-6500 tokens, approximately **75-80% of tokens are evicted**. Only ~20-25% of the original context is retained. See `eviction_analysis.png`.

| Cache Size | Approx. Tokens Retained | Approx. % Evicted |
|-----------|------------------------|-------------------|
| 1024 | 1024 of ~5000 | ~80% |
| 2048 | 2048 of ~5000 | ~60% |
| 3072 | 3072 of ~5000 | ~40% |

### 6.2 Relevant Token Analysis

To quantify how much of each context is actually needed to answer the question, we search for answer string occurrences and question entity mentions within the context, then estimate the relevant region as a +-200 character window around each occurrence. This is a lower-bound estimate — the true relevant context may be larger (e.g. surrounding sentences that provide necessary context for interpreting the answer).

See `relevant_tokens.png`, `relevant_tokens_boxplot.png`, `answer_positions.png`, and `eviction_survival.png`.

| Task | Mean Relevant Tokens | % of Context | # Distinct Regions | Answer Occurrences | Answer Position |
|------|---------------------|-------------|-------------------|-------------------|-----------------|
| Qasper | **192** | 3.9% | 1.2 | 0.6 | 0.47 (mid) |
| 2WikiMQA | **1124** | 21.6% | 5.6 | 2.9 | 0.38 |
| Qasper-E | **271** | 5.2% | 1.5 | 1.3 | 0.43 |
| HotpotQA-E | **2034** | 42.1% | 9.3 | 8.4 | 0.21 (early) |
| 2WikiMQA-E | **1173** | 22.8% | 5.8 | 2.7 | 0.38 |

**Key findings:**

- **Qasper tasks have very sparse relevant content** (~190-270 tokens, 4-5% of context, ~1.3 regions). The answer comes from a single localised passage. At cache=1024, essentially all relevant tokens fit — an ideal eviction policy could retain them with 97%+ probability.
- **Multi-hop tasks have much denser relevant content** (~1100-2000 tokens, 22-42% of context, 5-9 regions). The answer entity appears multiple times across different passages, and question entities are scattered throughout.
- **HotpotQA-E is the most demanding**: 2034 mean relevant tokens across 9.3 regions — exceeding the cache=1024 budget. Even an ideal policy cannot retain all relevant tokens at cs1024. The answer entity appears 8.4 times on average, suggesting it is mentioned across many distractor passages (not just the 2 gold ones).
- **Answer position differs**: HotpotQA-E answers tend to appear early (0.21), while Qasper answers are mid-document (0.47). This means a recency-based policy would systematically evict HotpotQA-E's early answer occurrences.

**Estimated relevant token survival** (assuming an ideal eviction policy that prioritises relevant tokens):

| Task | cache=1024 | cache=2048 | cache=3072 |
|------|-----------|-----------|-----------|
| Qasper | 97% | 100% | 100% |
| 2WikiMQA | 83% | 95% | 98% |
| Qasper-E | 96% | 98% | 100% |
| HotpotQA-E | **60%** | 87% | 98% |
| 2WikiMQA-E | 82% | 95% | 99% |

HotpotQA-E is the only task where relevant tokens substantially exceed the cache budget at cs1024. This should make it the most eviction-sensitive task under our initial hypothesis — but as Report 1 shows, the actual results tell a different story.

### 6.3 Per-Task Information Locality

**Qasper / Qasper-E (Localised):**
- The answer to a question about a scientific paper typically resides in a single paragraph or section.
- A well-trained eviction policy could learn to retain the relevant section and discard irrelevant ones (e.g., related work, bibliographic information).
- The structured nature of papers (sections, clear topic boundaries) may help the eviction policy identify relevant regions.
- However, the diverse answer types (yes/no, unanswerable) mean the model sometimes needs broad context to determine that something is *not* answerable.

**2WikiMQA / 2WikiMQA-E (Distributed, Multi-hop):**
- Answers require combining information from 2+ passages. If either passage is evicted, the answer is lost.
- The passages are Wikipedia articles about different entities — they may have little semantic overlap, making it harder for an attention-based eviction policy to identify both as relevant.
- Example: "Where was the wife of Francis I Rakoczi born?" requires (1) finding who Francis I Rakoczi's wife was in one passage, then (2) finding where she was born in another passage. Evicting either passage is fatal.

**HotpotQA-E (Distributed, Bridge/Comparison):**
- Similar to 2WikiMQA but with a specific structure: bridge questions (entity chaining) and comparison questions (comparing two entities).
- Comparison questions ("Are both X and Y described as Z?") require information about both entities — evicting either is fatal.
- Bridge questions follow a chain: passage A mentions entity B, passage B contains the answer. The chain structure means the eviction policy must retain both links.
- The context contains many **distractor passages** (typically 8-10 passages, of which only 2 are relevant). This makes the task harder for eviction: the policy must identify the 2 gold passages among many distractors.

### 6.3 Eviction Sensitivity Summary

| Task | Info Locality | Sensitivity | Reasoning |
|------|--------------|-------------|-----------|
| Qasper | Localised | **MEDIUM** | Answer in one section; eviction may lose it, but structured context helps |
| 2WikiMQA | Distributed | **HIGH** | Must retain 2+ passages; loss of either is fatal |
| Qasper-E | Localised | **MEDIUM** | Same as Qasper |
| HotpotQA-E | Distributed | **HIGH** | 2 gold passages among many distractors; both needed |
| 2WikiMQA-E | Distributed | **HIGH** | Same as 2WikiMQA |

---

## 7. Performance Hypotheses

### 7.1 Condition B0: Base Model, Full Context (No Training, No Eviction)

The base LLaMA 3.2-1B-Instruct model with full context, no fine-tuning.

**Predicted ranking (best to worst):**
1. **2WikiMQA / 2WikiMQA-E** — Short factoid answers (1-3 words) matching the instruct model's natural tendency to be concise. The multi-hop reasoning is challenging but the answer format is forgiving (short entity names get high F1 even with slight formatting differences).
2. **HotpotQA-E** — Similar short-answer format. Slightly harder due to bridge reasoning and more distractors.
3. **Qasper / Qasper-E** — The diverse answer types (yes/no, unanswerable, phrases, sentences) make this harder for an untuned model. The model may over-generate or fail to match the expected answer format. Longer answers naturally have lower F1. The "unanswerable" category is especially challenging without training.

**Reasoning:** The base instruct model should handle short-answer extraction reasonably well but struggle with Qasper's diverse answer format. F1 scoring penalises both over- and under-generation, and Qasper's longer expected answers create more opportunities for F1 loss. The multi-hop tasks, while conceptually harder, have simpler answer formats that are easier to match.

### 7.2 Condition M1: LoRA Fine-tuned, Full Context (No Eviction)

LoRA fine-tuning on q_proj and v_proj (rank 8) with full context.

**Predicted ranking (best to worst):**
1. **2WikiMQA / 2WikiMQA-E** — Short factoid answers are easiest to learn. Fine-tuning should quickly teach the model to extract entity names.
2. **HotpotQA-E** — Same reasoning, but bridge questions are slightly harder.
3. **Qasper / Qasper-E** — Fine-tuning should significantly help with format learning (yes/no, unanswerable). However, the diversity of answer types means the model must learn multiple response strategies, which is harder with limited data.

**Expected change from B0:** All tasks should improve substantially. The relative ordering may shift if Qasper benefits disproportionately from learning the answer format conventions. However, the small training set (only ~66-86 Qasper samples) limits how well the model can learn these diverse patterns.

### 7.3 Condition M2: NAMM Only, Cache Size 1024 (No LoRA, Frozen LLM)

NAMM eviction policy trained via CMA-ES, no LoRA fine-tuning. ~80% of tokens evicted.

**Predicted ranking (best to worst):**
1. **Qasper / Qasper-E** — Localised information means the eviction policy *may* learn to retain the relevant section. Even if it cannot perfectly identify the relevant section, retaining 20% of a paper randomly has a reasonable chance of keeping the relevant paragraph.
2. **2WikiMQA / 2WikiMQA-E** — Multi-hop: even a good eviction policy must retain 2+ passages. The probability of retaining both relevant passages when keeping only 20% of tokens is much lower.
3. **HotpotQA-E** — Worst among the multi-hop tasks because HotpotQA has more distractor passages, making it harder for the eviction policy to identify the 2 gold passages.

**Expected change from B0:** All tasks should degrade substantially due to information loss. The key question is whether the NAMM policy can learn to selectively retain relevant tokens. Multi-hop tasks should suffer disproportionately because:
- With random eviction, the probability of retaining a single relevant passage of length L in a context of total length T with cache size C is approximately C/T. The probability of retaining *two* independent passages is (C/T)^2 — much lower.
- A learned eviction policy can do better than random, but must solve a harder problem for multi-hop tasks (identify *multiple* scattered relevant regions).

### 7.4 Condition M3: LoRA + Frozen NAMM, Cache Size 1024

LoRA fine-tuning with a pre-trained (frozen) NAMM eviction policy active. The LLM adapts to working with a truncated cache.

**Predicted ranking (best to worst):**
1. **2WikiMQA / 2WikiMQA-E** — Even with eviction, fine-tuning teaches the model to extract answers from whatever context remains. Short factoid format is still easier to match.
2. **HotpotQA-E** — Similar reasoning but harder due to bridge questions.
3. **Qasper / Qasper-E** — The combination of information loss from eviction AND diverse answer formats makes this the hardest condition for Qasper. However, LoRA may partially compensate by teaching the model to better utilise the retained context.

**Expected change from M2:** Substantial improvement across all tasks. LoRA fine-tuning should help the model adapt to working with the compressed context. The model can learn to:
- Focus on retained tokens more effectively
- Produce answers even from incomplete context
- Learn the correct answer format (especially beneficial for Qasper's diverse types)

**Key question:** Can LoRA fine-tuning fully recover the information lost to eviction? For Qasper (localised), recovery should be more feasible. For multi-hop tasks, if both relevant passages are evicted, no amount of fine-tuning can recover the information — the fundamental information is gone.

---

## 8. Predictions Summary Table

### 8.1 Relative Task Ordering (Predicted F1, Best to Worst)

| Rank | B0 (Base, Full) | M1 (LoRA, Full) | M2 (NAMM cs1024) | M3 (LoRA+NAMM cs1024) |
|------|-----------------|------------------|-------------------|------------------------|
| 1 (best) | 2WikiMQA | 2WikiMQA | Qasper | 2WikiMQA |
| 2 | 2WikiMQA-E | 2WikiMQA-E | Qasper-E | 2WikiMQA-E |
| 3 | HotpotQA-E | HotpotQA-E | 2WikiMQA | HotpotQA-E |
| 4 | Qasper | Qasper-E | 2WikiMQA-E | Qasper-E |
| 5 (worst) | Qasper-E | Qasper | HotpotQA-E | Qasper |

### 8.2 Expected Trends Across Conditions

| Trend | Prediction | Confidence |
|-------|-----------|------------|
| M1 > B0 for all tasks | Yes, fine-tuning should help universally | High |
| M2 < B0 for all tasks | Yes, 80% eviction causes major degradation | High |
| M3 > M2 for all tasks | Yes, LoRA compensates for eviction | High |
| M3 < M1 for all tasks | Likely, eviction still loses information | Medium |
| Multi-hop tasks degrade more under eviction (M2) | Yes, distributed info is harder to retain | High |
| Qasper tasks benefit more from fine-tuning (B0 to M1) | Moderate, due to format learning | Medium |
| _E variants track their base counterparts | Yes, same task type | High |

### 8.3 Specific Predictions

1. **Largest eviction degradation (B0 to M2):** HotpotQA-E — multi-hop with many distractors, the eviction policy is least likely to retain both gold passages.

2. **Smallest eviction degradation:** Qasper/Qasper-E — localised information is easier for the eviction policy to preserve.

3. **Largest fine-tuning gain (B0 to M1):** Qasper/Qasper-E — the base model likely struggles with the diverse answer format; fine-tuning teaches it.

4. **Best recovery under M3 (closest to M1):** 2WikiMQA/2WikiMQA-E — short factoid answers are most recoverable even from incomplete context.

5. **Worst recovery under M3:** HotpotQA-E — if both gold passages are evicted, LoRA cannot recover the missing information.

---

## 9. Figures

- `dataset_characteristics.png` — Summary table of per-task characteristics including sample counts, answer types, information locality, and eviction sensitivity ratings.
- `prompt_templates.png` — Full prompt templates for each task.
- `length_distributions.png` — Context length and answer length distributions for eligible samples.
- `answer_types.png` — Stacked bar chart showing answer type breakdown per task.
- `eviction_analysis.png` — Token eviction rates at different cache sizes and retention distribution at cs=1024.
- `relevant_tokens.png` — Mean relevant tokens, relevant fraction, and number of distinct relevant regions per task.
- `relevant_tokens_boxplot.png` — Box plot of relevant token distributions per task, with cache size reference lines.
- `answer_positions.png` — Distribution of first answer occurrence position within context (0=start, 1=end).
- `eviction_survival.png` — Estimated fraction of relevant tokens surviving at each cache size, assuming ideal eviction.

---

## 10. References

- Bai, Y. et al. (2023). "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding." *arXiv:2308.14508*.
- Dasigi, P. et al. (2021). "A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers." *NAACL 2021*.
- Ho, X. et al. (2020). "Constructing A Multi-Hop QA Dataset for Comprehensive Evaluation of Reasoning Steps." *COLING 2020*.
- Yang, Z. et al. (2018). "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering." *EMNLP 2018*.
