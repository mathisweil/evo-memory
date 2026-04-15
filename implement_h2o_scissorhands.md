# Implement H2O and ScissorHands KV Cache Eviction Policies

## Goal

Add two heuristic KV cache eviction baselines — **H2O** and **ScissorHands** — to the `evo-memory` codebase so they can be used as drop-in alternatives to NAMM for evaluation. They must integrate seamlessly: the rest of the training (LoRA, ES, joint) and evaluation pipelines must remain untouched. The only change a user makes is selecting an eviction policy by name.

---

## Step 0 — Analyse the codebase before writing any code

Read and understand the following files/directories in full before planning changes:

```
namm/policy/            # existing eviction policies (recency, deep scoring)
namm/llms/              # LLM wrappers — where KV cache manipulation happens
namm/evaluation/        # evaluator that calls eviction during generation
namm/trainer.py         # NAMM training loop
scripts/run_eval.py     # evaluation entry point
scripts/run_lora.py     # LoRA training — uses namm_active flag
scripts/run_joint.py    # joint training
config/policy/          # Hydra policy configs
config/run/             # run presets (e.g. recency_baseline_llama32_1b)
```

Identify:
1. The **base class or interface** that eviction policies implement (likely in `namm/policy/`).
2. How a policy is **instantiated** from Hydra config and **injected** into the LLM wrapper / evaluator.
3. How the **recency baseline** works end-to-end — it is the simplest existing heuristic and the closest template for H2O/ScissorHands.
4. Where **attention scores** are accessible during generation (needed by both H2O and ScissorHands).
5. How `cache_size` is enforced during evaluation.

**Do not write any code until this analysis is complete and you have a clear plan.**

---

## Step 1 — Plan the implementation

Produce a brief written plan covering:
- Which files you will create (new policy classes).
- Which files you will modify (config, registration, LLM wrapper if attention scores need exposing).
- How the new policies will be selectable via Hydra config or CLI flag.
- A testing strategy (unit test + smoke eval run).

Get confirmation on the plan before proceeding.

---

## Algorithm Specifications

### H2O (Heavy-Hitter Oracle) — Zhang et al., NeurIPS 2023

**Core idea:** Maintain a fixed-budget KV cache by dynamically retaining a balance of *Heavy Hitter* tokens (highest accumulated attention scores) and *recent* tokens.

**Notation:**
- `k` = total KV cache budget (= `cache_size`, e.g. 1024)
- `k_hh` = budget allocated to heavy hitters (default: `k // 2`)
- `k_recent` = budget allocated to recent tokens (`k - k_hh`)
- `S_i` = set of token indices retained at step `i`, `|S_i| ≤ k`
- `o_i` = normalised attention vector at step `i` (softmax output, shape `[num_heads, seq_len]`)

**Algorithm (per layer, per head):**

```
Initialise: S_0 = {}, accumulated_scores = zeros(max_seq_len)

For each decoding step i:
    1. Compute attention: o_i = softmax(q_i @ K_S^T / sqrt(d_k))
       (attention is computed only over tokens currently in cache S_{i-1} ∪ {i})
    
    2. Update accumulated scores:
       For each j in S_{i-1} ∪ {i}:
           accumulated_scores[j] += o_i[j]
    
    3. If |S_{i-1}| + 1 > k:
       - Partition candidates into:
         * recent_set = last k_recent token indices
         * hh_candidates = (S_{i-1} ∪ {i}) \ recent_set
       - Among hh_candidates, keep top k_hh by accumulated_scores
       - S_i = top_k_hh ∪ recent_set
    Else:
       S_i = S_{i-1} ∪ {i}
    
    4. Evict KV pairs not in S_i from key/value cache tensors.
```

**Key implementation details:**
- The budget split is even by default: `k_hh = k_recent = k // 2`. Expose this as a config parameter `heavy_hitter_ratio` (default 0.5).
- Accumulated scores are **per-head** — each head independently decides which tokens are heavy hitters. Do NOT average across heads.
- During the **prompt/prefill phase** (processing the initial prompt), process all tokens first, then compress the KV cache to budget `k` using the same accumulated-score logic before starting generation.
- The eviction happens **after** computing attention at each step, using the scores from that step.
- Accumulated scores use a **running sum** (not average). The H2O paper notes that averaging introduces bias toward recent tokens and performs worse.
- Heavy hitters correlate with frequently co-occurring tokens (e.g., punctuation, articles). Their accumulated scores follow a power-law distribution.
- The theoretical guarantee: under a submodularity assumption on attention, the greedy H2O policy achieves `f(S) ≥ (1-α)(1-1/e) max_{|S|=k} f(S) - β`.

**No training required.** H2O is a pure inference-time heuristic applied post-hoc to any pretrained model.

---

### ScissorHands — Liu et al., NeurIPS 2023

**Core idea:** Exploit the **Persistence of Importance Hypothesis**: tokens that received high attention in recent steps will continue to receive high attention in future steps. Use a history window to identify and evict non-important tokens.

**Notation:**
- `B` = total KV cache budget (= `cache_size`)
- `w` = history window size (number of recent steps to look back for importance; default: `B // 2`)
- `r` = recent window size (tokens always kept; default: `B // 4`)
- `m` = number of tokens to drop when cache exceeds budget (default: `B // 2`)
- `α_i` = attention score vector at step `i`

**Algorithm (per layer, per head):**

```
Initialise: K_cache, V_cache = empty, n = 0

For each decoding step t:
    1. Append new token's key/value to cache: n += 1
    
    2. If n > B:
       Run Compress(K_cache, V_cache, w, r, m, t):
       
       a. importance_record I = zeros(n)  # counter for "unimportant" votes
       
       b. For each step j in [t-w, t]:  # history window
            For each token position p:
                If attention_score[j][p] < 1/t:  # below average threshold
                    I[p] += 1  # increment "unimportant" counter
       
       c. I[last r positions] = 0  # protect recent window — always keep
       
       d. keep_set = argsort(I)[:-m]  # drop m tokens with HIGHEST unimportance count
          (i.e., keep tokens that were LEAST often unimportant)
       
       e. Retain only keep_set in K_cache, V_cache
          n = n - m
```

**Key implementation details:**
- The threshold for "pivotal" is `α = 1/t` where `t` is the current sequence position. This represents an average-mixing score — tokens above this threshold are considered important.
- The importance record counts how many times a token was deemed **unimportant** (below threshold). Higher count → more likely to be evicted.
- The **recent window** (`r`) protects the most recent tokens because we lack information about their future importance.
- The **history window** (`w`) limits how far back we look, reducing variance in importance estimation.
- Compression is NOT triggered every step — only when `n > B`. After compression, `m` tokens are dropped, so the next compression happens after `m` new tokens are generated.
- Default hyperparameters from the paper: `m = 0.5 * B`, `w = B // 2`, `r = B // 4`.
- Budget allocation across layers: allocate **more budget to later layers** to compensate for lower persistence ratios observed in deeper layers (Figure 2 of the paper). For simplicity in our implementation, use uniform allocation unless a `layer_budget_schedule` config is provided.
- ScissorHands requires storing attention scores for the history window (last `w` steps). This is a small memory overhead: `w * num_heads * current_cache_len` floats.
- The persistence ratio is empirically >95% in most layers (except later ones), validating the hypothesis.
- Theoretical guarantee: the error between full-cache and budget-cache generation is bounded by `E[||x_t - x̃_t||] ≤ O((1 - B/T_max) * power_law_term)` where the power-law term decreases with stronger attention sparsity.

**No training required.** ScissorHands is a pure inference-time heuristic.

---

## Integration Requirements

### 1. Policy interface

Both new policies must implement the **same interface** as existing policies (recency, deep scoring). Study the existing base class. Each policy needs:
- An `__init__` method accepting `cache_size` and policy-specific hyperparameters.
- A method to **score or select tokens to keep/evict** given the current KV cache state and attention outputs.
- Compatibility with the existing KV cache tensor format used by the LLaMA wrapper.

### 2. Attention score access

Both H2O and ScissorHands need the **raw attention scores** (post-softmax) at each generation step. Check whether the existing LLM wrapper already exposes these (NAMM uses attention spectrograms, so there may already be a mechanism). If not, add `output_attentions=True` to the model forward call and pipe the attention tensor to the policy.

**Critical:** Attention must be per-head, not averaged across heads. Shape: `[batch, num_heads, 1, seq_len]` at each generation step.

### 3. Config integration

Create Hydra configs so users can select policies like:

```bash
# H2O evaluation
python scripts/run_eval.py \
    --run_config h2o_baseline_llama32_1b \
    --cache_size 1024

# ScissorHands evaluation  
python scripts/run_eval.py \
    --run_config scissorhands_baseline_llama32_1b \
    --cache_size 1024
```

Also support combining with LoRA (analogous to M3 — LoRA + frozen eviction policy):

```bash
# LoRA + H2O at train time
python scripts/run_lora.py \
    --config scripts/configs/m3_lora_frozen_namm_5t.yaml \
    --eviction_policy h2o \
    --cache_size 1024 \
    --run_name m3_h2o_lora

# LoRA + ScissorHands at train time
python scripts/run_lora.py \
    --config scripts/configs/m3_lora_frozen_namm_5t.yaml \
    --eviction_policy scissorhands \
    --cache_size 1024 \
    --run_name m3_scissorhands_lora
```

### 4. Eval compatibility

The new policies must work with the existing evaluation pipeline (`scripts/run_eval.py`, `scripts/eval_namm_splits.py`). Evaluation must respect:
- `cache_size=1024` (FAIR-01 constraint)
- Greedy decoding (`temperature=0.0`)
- The 5-task QA subset with 70/15/15 splits
- Output `results.json` in the same format

### 5. No training loop needed

Neither H2O nor ScissorHands requires any training. They are eval-only baselines (like B1 recency). Do NOT create training scripts for them. They only need:
- Policy implementation
- Hydra configs  
- Eval entry point compatibility

### 6. File organisation

Follow existing conventions:
```
namm/policy/h2o.py              # H2O policy class
namm/policy/scissorhands.py     # ScissorHands policy class
config/policy/h2o.yaml           # Hydra config for H2O
config/policy/scissorhands.yaml  # Hydra config for ScissorHands
config/run/h2o_baseline_llama32_1b.yaml
config/run/scissorhands_baseline_llama32_1b.yaml
tests/test_h2o.py               # unit tests
tests/test_scissorhands.py      # unit tests
```

---

## Correctness Checks

### H2O
- With `cache_size = seq_len` (no eviction), output must be identical to full-cache baseline.
- Accumulated scores must be strictly per-head — verify by checking that different heads retain different token sets.
- After prefill compression, exactly `k` tokens remain.
- The recent window is always preserved (never evicted).

### ScissorHands
- With `cache_size = seq_len`, output must be identical to full-cache baseline.
- Importance record correctly counts below-threshold occurrences over the history window only.
- Recent window tokens always have `I[p] = 0` (protected).
- After compression, exactly `n - m` tokens remain, and `n - m ≤ B`.

### Both
- Unit test: create a small synthetic attention pattern, run eviction, verify the correct tokens are retained.
- Smoke test: run eval on 5 samples from the 5-task QA subset, confirm `results.json` is produced and scores are non-zero.
- Memory: verify GPU memory usage is ≤ full-cache baseline (the whole point is memory reduction).

---

## What NOT to change

- Do not modify `namm/trainer.py` (NAMM CMA-ES training loop).
- Do not modify `scripts/run_namm.py` or `scripts/run_joint.py` internals.
- Do not modify existing NAMM policy classes (`namm/policy/` existing files).
- Do not change the evaluation metric calculation or the results.json schema.
- Do not change the LLM wrapper's core forward pass — only add optional attention output if not already available.
- Do not break the `transformers==4.41.2` version pin (critical: 4.45+ breaks DynamicCache API).

---

## Summary of deliverables

1. `namm/policy/h2o.py` — H2O eviction policy
2. `namm/policy/scissorhands.py` — ScissorHands eviction policy
3. Hydra config files for both policies and run presets
4. Any minimal modifications to LLM wrapper to expose attention scores (if not already available)
5. Unit tests for both policies
6. Updated `README.md` with new baseline commands
