---
name: research-context
description: Concept reference for the four papers this codebase implements (LoRA, NAMM, Evolution Strategies, LongBench), the M1–M4 experimental conditions, and the research questions. Load this skill when coding decisions hinge on *why* the experiment is structured the way it is — e.g. choosing whether a refactor changes meaningful behaviour, naming a new variable, deciding which condition a script belongs to, interpreting an unexpected eval result, or sanity-checking that a config edit preserves the FAIR-01 comparison.
---

# Research context (for coding decisions)

This skill exists so that when you are editing code, you can tell whether a change preserves the experimental semantics or quietly breaks them. It is **not** a paper-writing guide.

## What this project actually studies

Whether a learned KV-cache eviction policy (NAMM) and a parameter-efficient adapter (LoRA) should be **co-trained**, given that LoRA is normally fine-tuned with full context but deployed under a memory budget. The mismatch between train-time and test-time context is the central distribution shift the project measures.

- Base model: **Llama-3.2-1B-Instruct**.
- Benchmark: a **5-task LongBench QA subset** — `qasper`, `2wikimqa`, `qasper_e`, `hotpotqa_e`, `2wikimqa_e`.
- Hardware: single GPU.
- Metric: token-level F1 on the **test** split (70 prompts).

## The four papers in `papers/`

### LoRA — Hu et al., 2021 (`papers/2106.09685v1.pdf`)
Freezes pretrained `W₀` and learns a low-rank update `ΔW = B A`, with `A ∈ ℝ^{r×d}`, `B ∈ ℝ^{d×r}`, `r ≪ d`. `B` is initialised to zero so the adapted model starts identical to the base. Only `A` and `B` are optimised; `W₀` and its optimiser state never enter VRAM. The empirical claim is that the adaptation update has low intrinsic rank, so very small `r` recovers most of full-fine-tune quality.

**Implications for the code:**
- We adapt only `q_proj` and `v_proj`. Adding `k_proj` / `o_proj` / MLP modules changes capacity in a way that breaks comparison to M1.
- Convention `α = 2 r`. The rank sweep keeps this ratio fixed.
- `B` zero-init means a freshly-initialised LoRA adapter must produce identical logits to the base model. If a smoke test shows logit drift at step 0, that is a real bug, not floating-point noise.

### NAMM — Cetin et al., ICLR 2025 (`papers/2410.13166v4.pdf`)
A small network that decides, per layer and per head, which KV tokens to keep in cache. Conditions exclusively on features extracted from the **attention matrix** — not on token identities or hidden states — which makes it architecture-agnostic. Tokens with negative scores are evicted (threshold mode), or only the top-`k` are kept (top-k mode). Token selection is binary and non-differentiable, which is why training uses **CMA-ES** rather than backprop.

**Implications for the code:**
- Anything that introduces gradient flow through the eviction step is wrong by construction. If you find yourself reaching for `torch.where` with grads or a Gumbel-softmax relaxation, stop — that is not what NAMM is.
- Scoring conditions on attention features only. Adding hidden-state inputs to the policy network breaks the architecture-agnostic property and makes M2 checkpoints unloadable.
- This project trains NAMMs **from scratch** on Llama-3.2-1B-Instruct. Sakana's pretrained checkpoints are *not* used. Anywhere the code defaults to a pretrained NAMM, that is a bug for our experiments.
- Top-k vs threshold-only are different objectives. M1–M4 use top-k (`cache_size=1024`). Threshold-only is a separate variant.

### Evolution Strategies — Salimans et al., 2017 (`papers/1703.03864v2.pdf`)
Black-box optimiser: perturb parameters with Gaussian noise, evaluate fitness for each perturbation, update the mean toward better-scoring directions. No gradients, no value function. This codebase uses **CMA-ES** (covariance-matrix-adapted ES) — not isotropic ES — for NAMM training, which additionally adapts the search-distribution covariance.

**Implications for the code:**
- ES is in the loop because eviction is non-differentiable. Do not "simplify" it away.
- ES fitness is noisy. Eval-loss spikes between CMA-ES generations are expected; do not add anti-spike clamping.
- `scripts/run_es.py` is a separate, standalone use of ES that perturbs LoRA / model weights (not NAMM weights). It is a control, not part of the M1–M4 main table — do not conflate the two code paths when refactoring.

### LongBench — Bai et al., ACL 2024 (`papers/2024.acl-long.172.pdf`)
21-dataset bilingual long-context benchmark; we use a 5-task English QA subset. Official scoring is token-level F1 with lowercase / strip-articles / strip-punctuation normalisation, reproduced in `namm/evaluation/`.

**Implications for the code:**
- Do not change the F1 normaliser. M2 checkpoints were selected against this exact metric — switching to a stricter or laxer one re-orders the validation history and invalidates "best checkpoint" decisions already saved to GCS.
- The 5-task subset is fixed: `qasper, 2wikimqa, qasper_e, hotpotqa_e, 2wikimqa_e`. Adding or substituting tasks breaks FAIR-01.

## Distribution-shift hypothesis (the thing the experiment is built around)

M1 (LoRA only) trains with the full KV cache and is evaluated with `cache_size=1024`. The LoRA learns from attention patterns over the full context — patterns that no longer exist at eval, where 75–85% of tokens are gone. Hypothesis: this train/test mismatch is a real source of error, and conditions that train under the same compressed context they will be evaluated under (M3, M4) should close the gap.

When you change anything that affects either training context or eval context, ask: *does this change preserve the train/eval relationship that the comparison depends on?* If you make M1 train with `cache_size=1024`, you have not improved M1 — you have built a different experiment.

## M1–M4 conditions (the table the code keeps in sync)

| ID | Adapter | NAMM at train | NAMM at eval | Cache | Script | Config |
|---|---|---|---|---|---|---|
| **B0** | none | — | none | ∞ | `run_eval.py` | `full_cache_baseline_llama32_1b` |
| **B1** | none | — | recency heuristic | 1024 | `run_eval.py` | `recency_baseline_llama32_1b` |
| **M1** | LoRA r∈{4,8,16}, SFT | off (full cache) | none (eval row uses 1024) | 1024 | `run_lora.py` | `scripts/configs/m1_lora_5t.yaml` |
| **M2** | none (frozen LLM) | CMA-ES (200 gens) | learned NAMM | 1024 | `run_namm.py` | `run@_global_=namm_bam_i1_llama32_1b_5t` |
| **M3** | LoRA r=8, SFT | frozen M2 NAMM | frozen M2 NAMM | 1024 | `run_lora.py` | `scripts/configs/m3_lora_frozen_namm_5t.yaml` |
| **M4** | LoRA r=8, SFT | co-trained NAMM | co-trained NAMM | 1024 | `run_joint.py --adapter_type lora` | `scripts/configs/m4_joint_lora_5t.yaml` |

**Compute parity is part of the design.** M4 runs 3 outer loops × (67 NAMM gens + 50 LoRA epochs) = 201 NAMM generations and 150 LoRA epochs total — exactly matching M2 and M1. Any refactor that lets M4's totals drift from M1+M2 invalidates the comparison even if the per-step code is correct. The 3-loop schedule supersedes an earlier 2×(100+75) design; see `docs/m4_joint_training_analysis.md`.

## Naming quirks (the ones that bite when reading code)

- **M3 historical naming.** The M3 condition was renamed in source from `rh_m4_frozen_5t` / `lora_rh_m4_instruct_5t.yaml` to `m3_lora_frozen_namm_5t` / `m3_lora_frozen_namm_5t.yaml`. The "m4" in the historical identifier is a leftover from before the M-numbering existed — it has nothing to do with M4 (joint). Historical artefacts still reference the old strings: WandB run names (`rh_m4_5t_cs*`), GCS checkpoint paths (`gs://.../lora-m4-frozen-5t-...`), and the on-disk `results/main_table_5t/M4/` directory all contain M3 results. Do not rename those external strings.
- **M4 only ever means joint.** M4 uses `run_joint.py`, never `run_lora.py`.
- **`rh_*` is a historical project tag** that no longer appears in current source — only in external artefacts (WandB, GCS, results dirs).
- **Stages are 0-indexed.** With `--num_outer_loops 3` (the M4 schedule), the final adapter checkpoint is `adapter/stage_2/`.
- **`namm/latest.pt` is overwritten every NAMM stage.** It always reflects the most recent stage; it is not a "best so far" checkpoint.

## Research questions (so you know what each condition is *for*)

- **RQ1 — M3 vs M1.** Does training an adapter under a learned eviction policy improve final QA F1 versus training with the full cache?
- **RQ2 — M4 vs M3.** Does co-training the eviction policy with the adapter beat training the adapter under a frozen policy?
- **RQ3 — A1 (M1 rank sweep).** How does adapter capacity (r ∈ {4, 8, 16}) interact with eviction-aware training?
- **RQ4 — A4 (M4 NAMM-on vs NAMM-off).** Are the jointly trained NAMM and LoRA co-dependent, or has the LoRA absorbed the eviction knowledge such that the NAMM becomes optional at inference?

When triaging an unexpected eval result, ask which RQ it speaks to. A surprise at A4 changes the modularity story; a surprise at A1 changes the rank-selection justification. They are not interchangeable.
