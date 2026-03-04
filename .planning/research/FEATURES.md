# Feature Research

**Domain:** Gradient-based LoRA finetuning + NAMM KV-cache eviction study on LLaMA 3.2-1B
**Researched:** 2026-03-03
**Confidence:** HIGH (codebase direct inspection + published literature)

---

## Context: What Already Exists

This is a SUBSEQUENT MILESTONE. The v1.0 infrastructure is complete. Features below are
ONLY what is new for the gradient-based LoRA study (v2.0). Do not duplicate existing
infrastructure.

| Existing (Do Not Re-implement) | Location |
|-------------------------------|----------|
| NAMM CMA-ES training loop | `memory_trainer.py`, `memory_evolution/cma_es.py` |
| PEFT LoRA injection (`apply_lora_adapters`) | `memory_llms/llama.py:214` |
| LoRA flat-vector extract/inject (`get_lora_params_flat`, `set_lora_params`) | `memory_llms/base.py:108-158` |
| LoRA checkpoint save/load with graceful fallback | `memory_trainer.py:1071-1172` |
| Three-task eval harness (QASPER, NarrativeQA, PassageRetrieval) | `memory_evaluator.py`, `task_sampler.py` |
| wandb logging, Hydra configs | `cfgs/`, `memory_trainer.py` |
| Full-cache, recency, NAMM baselines | `cfgs/run/`, `exp_local/` |

The pivot from ES-based LoRA (v1 plan) to gradient-based LoRA (v2 goal) means:
- ES-01 through ES-05, DIAG-01 through DIAG-05 from v1 REQUIREMENTS.md are **superseded**
- The LoRA seam (LORA-01 through LORA-04, already complete) is **reused as-is**
- A new gradient training loop is built alongside the existing CMA-ES loop

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features the study requires to produce credible results. Missing any one makes the paper
invalid or unreproducible.

| Feature | Why Expected | Complexity | Depends On |
|---------|--------------|------------|-----------|
| **Gradient-based LoRA training loop** — AdamW optimizer, cross-entropy NTP loss, frozen base weights, only LoRA params trained | Every LoRA finetuning study uses this pattern; PEFT `requires_grad` handles frozen base automatically | MEDIUM | Existing PEFT seam (LORA-01/02) |
| **NAMM-aware forward pass during LoRA training** — model runs with NAMM active (cache_size < full) while gradients flow through LoRA params | m4 condition requires it; NAMM eviction happens during prefill/generation inside the existing custom attention; LoRA adapters sit in q/v projections which are hit before NAMM eviction | HIGH | Existing NAMM attention code + LoRA seam |
| **NAMM deactivation / full-cache mode switch** — config flag `namm_active: true/false` disables eviction without code changes | m1 (LoRA only) requires full-cache; m3 (LoRA then NAMM) requires two-phase config; same code must cover both | LOW | Existing `policy=none` Hydra key |
| **Four method configurations (m1–m4)** — Hydra-composable pipelines, not separate codepaths | Four distinct experimental conditions are the core of the study; config-driven is mandatory per PROJECT.md constraint | MEDIUM | Gradient loop + NAMM switch |
| **Task-specific NTP loss on long-context data** — loss computed over answer tokens (not full sequence) on QASPER/NarrativeQA/PassageRetrieval documents | Standard task-specific LoRA practice; answer-only masking prevents the training signal being dominated by long irrelevant context | MEDIUM | Existing task data pipeline |
| **Checkpoint save/load for gradient-trained LoRA** — save optimizer state + LoRA weights; load for downstream NAMM training (m3) or eval (m4) | Multi-stage pipelines (m3, m4) require a LoRA checkpoint from stage 1 as input to stage 2; existing `_save_ckpt` saves LoRA weights but not optimizer state | LOW | Existing checkpoint I/O |
| **Eval harness that loads LoRA weights for inference** — evaluator reads LoRA state dict from checkpoint and activates adapters before eval | Without this, all post-training eval is impossible; confirmed: existing evaluator already has graceful fallback from v1.0 | LOW | Existing `MemoryHFEvaluator` |
| **Attention entropy logging during training** — per-layer Shannon entropy of attention weights logged to wandb every N steps | PRIMARY analysis metric; attention entropy captures how focused/diffuse attention becomes under finetuning; required to answer "how do representations differ?" | MEDIUM | Forward hook on attention output |
| **Layer-wise token retention rate logging** — during NAMM-active evaluation, log fraction of tokens retained per layer per step | PRIMARY analysis metric; reveals whether LoRA changes NAMM's eviction behavior or whether NAMM resists change | MEDIUM | Hook on NAMM eviction flags |
| **Per-condition result table** — structured eval output (all methods × all tasks × all metrics) written to CSV and wandb Summary | Science is unusable without tabular comparison; expected output format for any ablation study | LOW | Eval harness |
| **Seeded reproducibility** — same seed produces same results across m1–m4; wandb logs seed + config | Any multi-condition comparison requires reproducibility; already enforced by existing `seed` Hydra key | LOW | Existing infrastructure |

### Differentiators (Competitive Advantage)

Features that make the study scientifically stronger and more publishable. Not required
for the study to run, but required for the claims the study wants to make.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Attention entropy dynamics during LoRA training** — plot entropy vs. training step per layer, not just endpoint | Shows training *dynamics*, not just final state; entropy collapse or overload reveals mechanistic effects of LoRA | MEDIUM | Forward hook on SDPA weights; log every 50 steps |
| **NAMM eviction pattern comparison across methods** — visualize which positions NAMM evicts for m2 vs m3 vs m4 | Directly answers whether LoRA changes NAMM's token selection behavior; strong mechanistic claim | HIGH | Requires collecting eviction flags per position across eval set |
| **Training loss curve for LoRA phases** — NTP loss vs. step for m1/m3/m4 LoRA training, logged to wandb | Confirms LoRA is actually learning; rules out degenerate training (flat loss = LoRA not injected correctly) | LOW | Standard wandb loss logging |
| **Gradient norm monitoring during LoRA training** — log `||∇LoRA||` per step | Detects exploding/vanishing gradients; important diagnostic given bfloat16 base + float32 LoRA mixed precision | LOW | `torch.nn.utils.clip_grad_norm_` return value |
| **E5: General-text NTP baseline** — train m1 on Wikipedia/C4 NTP instead of task-specific QA | Tests whether task-specific data is necessary; expected result: task-specific outperforms general-text on benchmarks | MEDIUM | New data loader (not in existing pipeline) |
| **E4: NAMM deactivation post-training** — train m4, then evaluate with NAMM turned off | Tests whether LoRA adapts to NAMM's eviction behavior or to the task itself; if performance drops when NAMM removed, LoRA "depended on" NAMM | LOW | Config switch only; no new code |
| **E2: Cache size sweep** — run m3/m4 at cache_size ∈ {128, 256, 512} for one condition | Reveals sensitivity of the LoRA–NAMM interaction to memory budget; important for practical claims | LOW | Config variation only |
| **E3: Dataset structure effects** — compare QASPER (extractive QA) vs NarrativeQA (reading comprehension) as LoRA training target | Controls for dataset format; if m3-on-QASPER beats m3-on-NarrativeQA on QASPER but not vice versa, dataset overfitting is confirmed | MEDIUM | Requires two training runs with different task configs |
| **E1: Iterative interleaving** — alternate N steps of LoRA then N steps of NAMM CMA-ES, repeat | Tests whether co-adaptation converges to better solutions than sequential m3; implementation requires orchestration loop | HIGH | Requires combining gradient loop and CMA-ES loop in alternating schedule |
| **Layer-wise LoRA weight norm heatmap** — per-layer L2 norm of LoRA A/B matrices, logged at end of training | Shows which layers adapt most; q_proj vs v_proj breakdown reveals whether query or value projection changes | LOW | One-time logging at checkpoint save |
| **Multi-seed error bars** — run each main condition (m1–m4) with seeds {1337, 42, 0} | Any published comparison requires error bars to be credible; 3 seeds × 4 methods × 3 tasks = 36 evals | MEDIUM | Compute cost: ~12h per seed sweep |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Gradient through NAMM policy** — train NAMM attention policy with backprop | Sounds like "true end-to-end training" | NAMM is not differentiable in its binary eviction action; CMA-ES is gradient-free by design; mixing gradient and ES updates in same loop creates unstable training | Keep NAMM training CMA-ES only (PROJECT.md constraint) |
| **Adapter merging before NAMM training** — call `merge_adapter()` to absorb LoRA into base weights before m3 NAMM training | "Cleaner" base model for NAMM to learn on | `merge_adapter()` corrupts base weights in-place; NAMM eviction then operates on merged weights that differ from eval-time weights if LoRA is re-enabled; already flagged as CRITICAL pitfall (v1 PITFALLS.md) | Use `disable_adapter_layers()` during NAMM training; keep base weights immutable |
| **Full fine-tuning (non-LoRA) for comparison** | "LoRA may miss something full FT captures" | 1B params × float32 gradients = 16 GB optimizer state alone; exceeds 4070Ti VRAM; not comparable to NAMM-scale adaptation | LoRA rank sweep (r=4, 8) covers the "more capacity" question within budget |
| **QLoRA / quantized training** | VRAM savings | Incompatible with existing bfloat16 NAMM attention code; quantization changes attention numerics in ways that affect NAMM eviction signals | Standard LoRA in float32 (already implemented) |
| **Gradient checkpointing during LoRA training** | VRAM savings on long sequences | Known bug in HuggingFace PEFT + frozen parameter gradient checkpointing (GitHub issue #23170, #1142); will silently produce wrong gradients or crash | Reduce batch size or sequence length instead |
| **vLLM or FlashAttention-2 acceleration** | Speed | NAMM requires custom attention code (stateless_parallel_modules); incompatible with fused kernels | Custom SDPA already fast enough at batch_size=4 |
| **Joint ES + gradient optimization (mode A/B from v1)** | "More principled joint optimization" | Superseded by group decision; gradient-based LoRA answers the research questions more directly; ES-based LoRA was overengineered for the actual study goals | Gradient LoRA only (this milestone) |
| **Multiple LLM architectures (Mistral, Phi-3, etc.)** | "Generalizability claims" | Breaks comparability with existing NAMM baselines which are all LLaMA 3.2-1B; adds training budget × N models | Single-model study is publishable with clear scope statement |
| **Real-time attention visualization dashboard** | "Interactive analysis" | Engineering cost disproportionate to scientific value; wandb already provides time-series | Log entropy scalars to wandb; produce static plots at analysis time |

---

## Feature Dependencies

```
[Gradient LoRA Training Loop]
    └──requires──> [PEFT LoRA Seam] (LORA-01/02 — DONE)
    └──requires──> [Task NTP data pipeline] (existing task_sampler, partial reuse)
    └──requires──> [AdamW optimizer on LoRA params only]
                       └──requires──> [requires_grad only on LoRA params] (PEFT does this)

[NAMM-aware LoRA Forward]
    └──requires──> [Gradient LoRA Training Loop]
    └──requires──> [NAMM active during generation] (existing; config-driven)

[m1 config]
    └──requires──> [Gradient LoRA Training Loop]
    └──requires──> [NAMM deactivation switch] (policy=none, cache_size=4096)

[m2 config] ──uses──> [Existing NAMM CMA-ES training] (DONE — existing checkpoint)

[m3 config]
    └──requires──> [m1 config] (stage 1: LoRA training)
    └──requires──> [LoRA checkpoint save with optimizer state]
    └──requires──> [NAMM CMA-ES training on LoRA-pretrained model] (stage 2)

[m4 config]
    └──requires──> [m2 config] (stage 1: NAMM training)
    └──requires──> [NAMM-aware LoRA Forward] (stage 2: LoRA training with NAMM active)

[Attention Entropy Logging]
    └──requires──> [Forward hook on attention layers]
    └──enhances──> [m1 vs m4 analysis]

[Layer-wise Token Retention Logging]
    └──requires──> [Hook on NAMM eviction flags]
    └──enhances──> [m3 vs m4 analysis]

[E1 Iterative Interleaving]
    └──requires──> [m3 config] (same stages, just repeated)
    └──requires──> [Orchestration loop over (LoRA steps, NAMM steps)]
    └──conflicts──> [simple sequential pipeline] (needs interleave controller)

[E4 NAMM Deactivation Post-training]
    └──requires──> [m4 config] (must have a NAMM-active trained LoRA checkpoint)
    ──config-only──> [NAMM deactivation switch]

[E5 General-text NTP]
    └──requires──> [Gradient LoRA Training Loop]
    └──requires──> [New data loader] (Wikipedia/C4 — NOT in existing pipeline)
    └──conflicts──> [existing task_sampler] (different data format)

[Per-condition Result Table]
    └──requires──> [m1 eval] + [m2 eval] + [m3 eval] + [m4 eval]
    └──requires──> [Eval harness loading LoRA from checkpoint] (already works, v1.0)
```

### Dependency Notes

- **m3 requires m1 completion:** The LoRA-pretrained checkpoint from m1 is the input to m3's NAMM training phase. m3 cannot start until m1's checkpoint exists and is validated on eval.
- **m4 requires m2 completion:** The NAMM-pretrained checkpoint from m2 (already exists: `exp_local/.../ckpt.pt`) is the input to m4's LoRA training phase. m4 can start from the existing NAMM checkpoint immediately.
- **E1 requires both gradient loop and CMA-ES loop to be orchestratable:** This is the highest-complexity secondary experiment. Defer until m1–m4 are complete.
- **Attention entropy conflicts with gradient checkpointing:** Hooks on attention weights during backward pass interact with gradient checkpointing. Since gradient checkpointing is already excluded (anti-feature above), this conflict is resolved.
- **E5 conflicts with existing task_sampler:** The existing pipeline reads LongBench format. Wikipedia/C4 NTP requires a different loader. Implement as a separate data module; do not modify task_sampler.

---

## MVP Definition

### Launch With (Core Study — Required for Paper)

Minimum features to produce the four main conditions and the two primary analysis metrics.

- [ ] **Gradient LoRA training loop (task-specific NTP, AdamW)** — the new capability that makes this milestone; without it no m1/m3/m4 results exist
- [ ] **NAMM-aware forward pass during LoRA training** — required for m4; without it m4 is identical to m1
- [ ] **m1 config + run** — LoRA-only baseline; anchor of the comparison
- [ ] **m3 config + run** — LoRA then NAMM; tests whether finetuned model trains better NAMM policies
- [ ] **m4 config + run** — NAMM then LoRA with NAMM active; tests whether NAMM presence during finetuning matters
- [ ] **Attention entropy logging** — primary mechanistic metric; required to answer "how do representations differ?"
- [ ] **Layer-wise token retention logging** — primary NAMM-interaction metric
- [ ] **Per-condition eval on all 3 tasks** — QASPER, NarrativeQA, PassageRetrieval from each trained checkpoint
- [ ] **Checkpoint save with optimizer state** — required for multi-stage m3/m4 pipelines
- [ ] **Result CSV + wandb Summary table** — usable output format for paper

### Add After Validation (Secondary Experiments)

Add once m1–m4 produce clean results and the core comparison is interpretable.

- [ ] **E2: Cache size sweep (cs=128, 256, 512)** — trigger: m3/m4 results show cache-sensitivity worth quantifying
- [ ] **E4: NAMM deactivation post-training** — trigger: m4 results show interesting behavior worth probing
- [ ] **E3: Dataset structure effects** — trigger: m1 vs m3 gap differs by task, suggesting dataset confound
- [ ] **Layer-wise LoRA weight norm heatmap** — trigger: attention entropy shows interesting layer-specific effects
- [ ] **Multi-seed runs (seeds 42, 0)** — trigger: preparing for paper submission

### Future Consideration (v3+)

Features to defer until core results are published or milestone v2 complete.

- [ ] **E1: Iterative interleaving** — HIGH complexity orchestration; only meaningful if sequential m3 underperforms; defer until m3 baseline established
- [ ] **E5: General-text NTP (Wikipedia/C4)** — requires new data loader; interesting but not essential for core claims; defer
- [ ] **NAMM eviction position visualization** — scientifically rich but engineering-heavy; defer to analysis phase if time permits
- [ ] **Multi-seed error bars across all conditions** — 12h+ compute; schedule after core results confirm direction

---

## Feature Prioritization Matrix

| Feature | Research Value | Implementation Cost | Priority |
|---------|---------------|---------------------|----------|
| Gradient LoRA training loop | HIGH | MEDIUM | P1 |
| NAMM-aware forward (m4) | HIGH | HIGH | P1 |
| m1 config + run | HIGH | LOW | P1 |
| m3 config + run | HIGH | LOW (once m1 done) | P1 |
| m4 config + run | HIGH | LOW (once NAMM-aware forward done) | P1 |
| Attention entropy logging | HIGH | MEDIUM | P1 |
| Layer-wise token retention logging | HIGH | MEDIUM | P1 |
| Checkpoint save with optimizer state | HIGH | LOW | P1 |
| Per-condition result table | HIGH | LOW | P1 |
| Training loss curve logging | MEDIUM | LOW | P2 |
| Gradient norm monitoring | MEDIUM | LOW | P2 |
| E4: NAMM deactivation post-training | MEDIUM | LOW | P2 |
| E2: Cache size sweep | MEDIUM | LOW | P2 |
| Layer-wise LoRA weight norm heatmap | MEDIUM | LOW | P2 |
| E3: Dataset structure effects | MEDIUM | MEDIUM | P2 |
| Multi-seed runs (3 seeds) | HIGH | MEDIUM | P2 |
| E5: General-text NTP | LOW | HIGH | P3 |
| E1: Iterative interleaving | MEDIUM | HIGH | P3 |
| NAMM eviction position visualization | MEDIUM | HIGH | P3 |

**Priority key:**
- P1: Must have — study is not runnable without these
- P2: Should have — strengthens claims and publishability
- P3: Nice to have — interesting but not essential; defer

---

## Detailed Feature Specifications

### F1: Gradient-Based LoRA Training Loop

**What it is:** Standard AdamW training loop over task-specific next-token prediction loss.
The base LLaMA weights are frozen (PEFT `requires_grad=False` on base, `True` on LoRA A/B).
Gradients flow only through LoRA A and LoRA B matrices. Loss is cross-entropy over
answer tokens (mask out context/question tokens to prevent long-context domination).

**Implementation pattern (HIGH confidence — standard PEFT+HuggingFace practice):**
```python
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],  # LoRA params only
    lr=1e-4,  # typical LoRA LR range: 1e-5 to 5e-4
    weight_decay=0.01
)
for batch in dataloader:
    output = model(**batch, labels=answer_masked_labels)
    loss = output.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
```

**Key correctness requirement:** `lora_dropout=0.0` already set in `apply_lora_adapters()` —
no change needed. Dropout during training would interact badly with NAMM's stateful KV cache.

**Expected behavior:** Loss decreases monotonically over training steps. If loss is flat or
increases, LoRA injection has failed (silent zero-injection pitfall — existing test_lora_seam.py
checks this). Loss magnitude: initially ~4-6 nats for 1B LLM on QA tasks; trained to ~2-3 nats.

**NAMM interaction:** During m4, NAMM is active during the forward pass. This means the
model sees a truncated KV cache (cache_size=128 or 256) while computing cross-entropy loss.
The gradient signal therefore "knows about" NAMM's eviction choices. This is the core
mechanistic hypothesis: does training with an active eviction policy change what LoRA learns?

### F2: NAMM-Aware Forward Pass During LoRA Training (m4)

**What it is:** The existing NAMM attention code (`stateless_parallel_modules/attention.py`)
runs during the forward pass with cache_size < 4096. LoRA adapters modify the q/v projections
BEFORE the eviction decision. So the gradient from the NTP loss flows through:
`output → lm_head → last_hidden_state → transformer layers → (q_proj+lora_A@lora_B) → SDPA → NAMM eviction`.

**Critical subtlety:** NAMM eviction is non-differentiable (binary keep/drop decision). The
gradient stops at the eviction boundary. LoRA gradients are computed from the post-eviction
hidden states, meaning LoRA learns to produce representations that are useful *given* NAMM's
eviction pattern. This is the mechanism by which NAMM presence during finetuning could matter.

**Expected behavior for m4:** If NAMM evicts tokens that LoRA would have attended to, the NTP
loss will be higher than m1 (which sees full cache). This is expected and intentional — we want
to measure whether LoRA compensates for NAMM's eviction by changing attention patterns
(measurable via attention entropy logging).

**Implementation complexity:** MEDIUM-HIGH. The key risk is that NAMM's population-parallel
infrastructure (pop_size=8 NAMM params evaluated simultaneously) conflicts with the single-
forward-pass gradient loop. For m4, we use the BEST NAMM params (from the existing
checkpoint, pop_size=1 at eval time) — NOT the population-parallel evaluation. This matches
the existing `_evaluate` code path, not `_train_step`.

### F3: Four Method Configurations (m1–m4)

All four methods are config-driven. The key Hydra switches are:
- `namm_active: bool` — whether NAMM eviction runs during LoRA training
- `lora_training_phase: bool` — whether gradient LoRA training runs
- `namm_training_phase: bool` — whether CMA-ES NAMM training runs
- `lora_init_from: path | null` — load LoRA weights from a prior checkpoint (m4)
- `namm_init_from: path | null` — load NAMM weights from a prior checkpoint (m3, m4)

| Method | lora_training_phase | namm_active during LoRA | namm_training_phase | init_from |
|--------|---------------------|------------------------|---------------------|-----------|
| m1 | true | false (full cache) | false | null |
| m2 | false | N/A | true | null (existing ckpt) |
| m3 | true → then false → true | false | true | lora_ckpt from m1 stage |
| m4 | true | true (NAMM active) | false | namm_ckpt from m2 |

m3 is a two-stage pipeline: run m1 (LoRA training, no NAMM), save LoRA checkpoint, then
run NAMM CMA-ES training starting from the LoRA-pretrained base. This is sequential, not
interleaved (E1 is the interleaved variant).

m4 is also two-stage: start from existing NAMM checkpoint (already done), then run LoRA
gradient training with NAMM active using best NAMM params frozen.

### F4: Attention Entropy Logging

**What it is:** Per-layer Shannon entropy of the attention weight distribution, logged during
training every 50 gradient steps and during evaluation. For attention matrix A ∈ R^{T×T},
entropy H = -Σ_j A_{i,j} log(A_{i,j}) averaged over heads and batch.

**Why it matters:** Attention entropy captures whether the model's attention is focused (low
entropy, attending to few tokens) or diffuse (high entropy, spreading across all tokens). LoRA
finetuning typically DECREASES entropy (more focused attention). NAMM-aware finetuning (m4)
may show different entropy dynamics because evicted tokens create "holes" in the attention
pattern, potentially forcing more focused attention on retained tokens.

**Implementation:** Register forward hooks on `nn.MultiheadAttention` or the custom SDPA
call inside `StatelessAttention`. Collect attention weights (already computed for NAMM),
compute entropy, log to wandb. Existing code already computes attention weights for NAMM
scoring — hook into that path.

**Expected findings (hypotheses to verify):**
- m1: entropy decreases over training (standard LoRA behavior)
- m4: entropy decreases faster or to lower floor (NAMM forces focused attention)
- m2 vs m4: if m4 entropy differs from m2, LoRA is changing NAMM's attention landscape

### F5: Layer-wise Token Retention Rate

**What it is:** Fraction of tokens retained (not evicted) by NAMM per layer per evaluation
step. For cache_size=128 on 1024-token context, baseline retention ≈ 12.5%. If LoRA
finetuning changes which tokens are "important" (per attention scores), NAMM will change
its eviction pattern.

**Why it matters:** This is the primary metric for answering "does LoRA change NAMM's
behavior?". If retention rates are identical across m2 and m4, LoRA's adaptation to NAMM
is purely in the downstream processing of retained tokens. If retention rates differ, LoRA
has changed what NAMM considers important.

**Implementation:** Hook into NAMM's eviction flag computation in `memory_policy/`.
Log `retained_fraction[layer_idx]` per eval step to wandb.

---

## Implementation Order (Critical for Phase Dependencies)

1. **Gradient LoRA training loop** — foundation; everything else builds on this
2. **m1 run and validation** — confirms the loop works; provides LoRA checkpoint for m3
3. **Attention entropy logging + token retention logging** — add hooks before m3/m4 runs
4. **m4 run** — reuses existing NAMM checkpoint; tests NAMM-aware forward
5. **m3 run** — uses m1 LoRA checkpoint; tests sequential ordering
6. **E2/E4 sweeps** — config-only variations once m3/m4 are clean
7. **E3/E5** — new data requirements; defer until core results complete
8. **E1** — highest complexity; defer until all sequential conditions done

---

## Sources

- Direct codebase inspection: `memory_trainer.py`, `memory_llms/llama.py`, `memory_llms/base.py`, `stateless_parallel_modules/attention.py`, `cfgs/`
- HuggingFace PEFT v0.11.1 documentation — gradient flow with frozen base model, `requires_grad` behavior (HIGH confidence)
- ALTER (arXiv 2025): attention entropy as a diagnostic for LoRA effects on token selection — confirms entropy logging approach (MEDIUM confidence)
- CS224R Stanford project (2025): representation drift analysis via attention entropy during RL finetuning — supports entropy as primary mechanistic metric (MEDIUM confidence)
- Attention-Gate paper (arXiv 2410.12876, 2025): finetuning reshapes token eviction patterns — supports the hypothesis that LoRA changes NAMM behavior (MEDIUM confidence)
- PEFT GitHub issues #23170, #1142 — gradient checkpointing incompatibility with frozen parameters (HIGH confidence — confirmed bug, affects anti-feature decision)
- PROJECT.md, REQUIREMENTS.md, ROADMAP.md — project constraints and existing infrastructure (HIGH confidence)

---

## Open Questions for Phase-Specific Research

- **NTP loss masking strategy:** Should loss be masked to answer tokens only, or computed over full sequence? Answer-only masking is standard practice but reduces effective batch size on short answers (QASPER answers are often 1-3 tokens). Needs empirical pilot: compare loss curves for both strategies on 10% of data before committing.
- **LoRA learning rate for long-context tasks:** Standard LR (1e-4) may be too high when NAMM is active (m4), because the gradient landscape is noisier (eviction creates gradient discontinuities). May need lower LR (1e-5) for m4. Flag for hyperparameter search in early phase.
- **Number of gradient steps per LoRA phase:** For m1/m3: how many gradient steps on QASPER before overfitting? With 180 training samples, overfitting risk is real. Validate with train/val split of existing LongBench data.
- **NAMM params during m4 LoRA training:** Use best single-population params (pop_size=1, best member from existing checkpoint) or mean params? Mean params are smoother but may not represent NAMM's actual test-time behavior. Use best params (same as `_evaluate` code path).
- **E1 interleaving frequency:** "Every N gradient steps switch to M CMA-ES iters" — what are N and M? No literature guidance for this specific setting. E1 is exploratory; treat N and M as hyperparameters and report sensitivity.

---
*Feature research for: gradient-based LoRA finetuning + NAMM KV-cache eviction study*
*Researched: 2026-03-03*
*Replaces: v1.0 FEATURES.md (ES-based LoRA, superseded by group decision)*
