# Project Research Summary

**Project:** NAMM + Gradient-Based LoRA Study (v2.0)
**Domain:** Gradient LoRA finetuning atop gradient-free NAMM KV-cache eviction system, LLaMA 3.2-1B
**Researched:** 2026-03-03
**Confidence:** HIGH

## Executive Summary

This project adds gradient-based LoRA finetuning to an existing, working NAMM CMA-ES system. The pivot from ES-based LoRA (v1.0) to gradient-based LoRA (v2.0) was a group decision that dramatically simplifies implementation: a standard AdamW training loop over task-specific NTP loss replaces the perturbation-and-tell ES loop. The entire existing NAMM infrastructure (CMA-ES trainer, PEFT LoRA seam from Phase 2, evaluator, checkpoint I/O) is reused. The new work is a single new class `LoRAGradTrainer` plus a `LongBenchNTPDataset`, wired into `main.py` with a method key (m1-m4) that routes the right trainer sequence. No new packages are required: AdamW, CrossEntropyLoss, DataLoader, and the cosine LR scheduler are all already present in the `th2` environment.

The core scientific goal is to compare four training configurations — m1 (LoRA only), m2 (NAMM only, already done), m3 (LoRA then NAMM), m4 (NAMM then LoRA with NAMM active) — across three LongBench tasks (QASPER, NarrativeQA, PassageRetrieval). Two primary analysis metrics drive the mechanistic story: per-layer attention entropy (does LoRA change attention focus?) and layer-wise token retention rate (does LoRA change what NAMM evicts?). These metrics are computed with 5-line PyTorch expressions logged to wandb — no external analysis library is needed or safe to add.

The dominant risk is the incompatibility between the existing codebase's blanket `@torch.no_grad()` decorators throughout `memory_trainer.py` and the gradient training requirement. The only safe resolution is a completely separate trainer class that never inherits the no-grad context. The second-tier risks are NAMM's non-differentiable eviction (not a bug — a design fact to document) and checkpoint handoff between pipeline stages (needs explicit config validation and a checkpoint inspector). A 21-minute estimated LoRA training run on the 4070Ti makes iteration fast; m4 total pipeline (Stage 1 NAMM ~3h + Stage 2 LoRA ~21min) fits in a single session.

---

## Key Findings

### Recommended Stack

No new packages are needed. The `th2` conda environment already contains everything required: PyTorch 2.7.0+cu128, Transformers 4.41.2, PEFT 0.11.1, datasets 2.20.0, accelerate 1.12.0. The new components are all standard PyTorch: `torch.optim.AdamW`, `torch.utils.data.DataLoader`, `torch.nn.utils.clip_grad_norm_`, and `get_cosine_schedule_with_warmup` (already importable from `transformers`).

The recommended AdamW config for this setup is `lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01` — the lower beta2 is standard for LLM finetuning (matches the Llama-3 paper), and weight decay prevents LoRA matrix divergence on a small dataset (200 QASPER documents). Gradient clipping at `max_norm=1.0` is mandatory. LoRA config change from v1.0: add `lora_dropout=0.05` for gradient training regularization (was 0.0 for ES where dropout was irrelevant). Keep `r=4, target_modules=['q_proj', 'v_proj'], lora_alpha=4, bias='none'`. Keep LoRA weights in float32 (existing design decision, prevents underflow at small rank). Do not use AMP/GradScaler, bitsandbytes quantization, TRL SFTTrainer, TransformerLens, or PEFT 0.12.x — all are incompatible with the existing NAMM attention infrastructure.

**Core technologies:**
- `torch.optim.AdamW`: gradient optimization of LoRA params — already installed, new usage pattern only
- `get_cosine_schedule_with_warmup` (transformers): LR schedule with warmup — already installed, 1-line import
- `torch.utils.data.DataLoader` + custom `LongBenchNTPDataset`: teacher-forced NTP training data — must be written (~50 lines); the existing `TaskSampler` drives generation, not teacher-forcing, and cannot be reused for training loss
- PEFT 0.11.1: LoRA injection — already validated in Phase 2, keep at 0.11.1 exactly
- Hydra + shell script: multi-stage pipeline orchestration — shell script passes checkpoint paths between stages (Hydra multirun cannot wire runtime values between runs)

### Expected Features

The study requires ten P1 features to produce any results, four P2 features to make results publishable, and three P3 features that are explicitly deferred. The boundary is decisive: without the P1 features, no m1/m3/m4 results exist; without P2 features, the results are not credible for a paper.

**Must have (table stakes — P1):**
- Gradient-based LoRA training loop (AdamW, NTP loss, frozen base, LoRA params only trained)
- NAMM-aware forward pass during LoRA training (m4 requires NAMM active with best params frozen)
- Four method configurations m1/m2/m3/m4 as Hydra-composable pipelines
- Task-specific NTP loss with masking (padding tokens masked with -100; answer-token masking optional)
- Checkpoint save/load including Adam optimizer state (required for multi-stage pipeline handoff)
- Attention entropy logging per layer (primary mechanistic metric)
- Layer-wise token retention rate logging (primary NAMM-interaction metric)
- Per-condition eval on all 3 tasks (QASPER, NarrativeQA, PassageRetrieval) after each training run
- Per-condition result CSV and wandb Summary table

**Should have (competitive — P2):**
- Training loss curve and gradient norm logging (confirms LoRA is learning; detects degenerate runs)
- E4: NAMM deactivation post-training (config-only; tests whether LoRA depends on NAMM)
- E2: Cache size sweep at cs=128/256/512 (config-only; reveals NAMM-LoRA sensitivity to memory budget)
- Multi-seed runs (seeds 1337, 42, 0) for publishable error bars

**Defer (v3+):**
- E1: Iterative interleaving of LoRA and NAMM training steps (highest-complexity orchestration; only meaningful after sequential m3 baseline established)
- E5: General-text NTP on Wikipedia/C4 (requires new data loader; not essential for core claims)
- NAMM eviction position visualization (engineering-heavy; defer to analysis phase)

### Architecture Approach

The architecture decision is a clean separation: new `LoRAGradTrainer` class in a new file (`lora_grad_trainer.py`), never touching the existing `MemoryTrainer`. Both trainers share `WrappedLlamaForCausalLM` (the model object) and delegate to `MemoryHFEvaluator` for LongBench scoring. Multi-stage pipeline coordination lives in `main.py`, not inside either trainer — each stage is a complete trainer invocation with checkpoint handoff. Three new files are created; seven existing files remain unchanged.

**Major components:**
1. `LoRAGradTrainer` (new, `lora_grad_trainer.py`) — gradient training loop, AdamW, DataLoader, NTP loss, analysis hooks; `namm_active` flag distinguishes m1/m3 (no eviction) from m4 (NAMM active, best params frozen)
2. `LongBenchNTPDataset` (new, `lora_ntp_dataset.py`) — wraps existing HF LongBench dataset into teacher-forced (input_ids, labels) pairs; left-truncates to `max_seq_len`; cannot reuse `TaskSampler` which drives generation, not teacher-forcing
3. `main.py` routing (modified) — dispatches to the right trainer sequence based on `cfg.method`; handles checkpoint path handoff between stages
4. `MemoryTrainer` (existing, unchanged) — CMA-ES NAMM evolution; all under `@torch.no_grad()`; reused as-is for m2 and the NAMM stage of m3
5. `MemoryHFEvaluator` (existing, unchanged) — LongBench generation scoring; delegated to by both trainers for eval

For m4's NTP forward pass: NAMM eviction happens in the forward pass but is non-differentiable (topk + index-select). Gradients flow through retained tokens only, reaching LoRA A/B matrices via q_proj and v_proj. This is correct behavior and the mechanistic basis of the m4 hypothesis. For multi-chunk documents, detach `past_key_values` between chunks (truncated BPTT) to prevent computation-graph explosion.

### Critical Pitfalls

1. **`@torch.no_grad()` blanket decorator kills LoRA gradients** — `memory_trainer.py` decorates 18+ methods including `__init__` and `train()`. Any LoRA gradient training code placed inside or called from these methods will have its gradient graph silently destroyed. `loss.backward()` raises `RuntimeError` or produces zero gradients without warning. Prevention: `LoRAGradTrainer` must be a completely separate class. Verify with `assert loss.requires_grad` after first NTP forward pass.

2. **NAMM eviction is non-differentiable — LoRA gradients stop at eviction boundary** — `BinarySelection.select_new_tokens()` uses `torch.topk()` and integer index selection. In m4, LoRA receives gradient signal only through retained tokens. This is not a bug; document it in the m4 training loop and confirm with `assert all(p.grad is not None for p in lora_params)` after first m4 backward.

3. **`model.generate()` reuse for NTP training silently produces zero gradients** — `MemoryHFEvaluator` calls `model.generate()` which runs non-differentiably. The LoRA trainer must implement a separate teacher-forced forward pass (`model(input_ids=..., labels=...)`). Prevention: `assert loss.grad_fn is not None` after first training forward pass.

4. **Checkpoint handoff between m3 and m4 fails silently on LoRA config mismatch** — if LoRA rank or target_modules differ between stages, the existing `_load_ckpt` fallback initializes LoRA from zero without raising an error. Prevention: checkpoint inspector utility validates `lora_config` fields before any multi-stage run; `assert lora_B.weight.norm() > 0` after loading a checkpoint that should contain trained LoRA.

5. **Mixed bfloat16/float32 precision with AMP corrupts LoRA weights mid-training** — `torch.cuda.amp.autocast` silently downcasts float32 LoRA weights to bfloat16, defeating the float32 precision guarantee. Prevention: no AMP and no GradScaler anywhere in the LoRA training loop. Assert `p.dtype == torch.float32` for all LoRA params after the first optimizer step.

---

## Implications for Roadmap

The existing roadmap (Phases 1-7 infrastructure, 8-10 science) is structurally correct. The research confirms the remaining phases should proceed in this order.

### Phase 3: Gradient LoRA Training Loop and NTP Dataset

**Rationale:** Foundation for everything. Without a working gradient training loop, no m1/m3/m4 results exist. This phase is entirely infrastructure — no science yet. Must come first because the two-stage pipelines (m3, m4) both depend on it. The existing Phase 2 LoRA seam (PEFT injection, flat-vector API, checkpoint I/O) is already done and is the direct dependency.

**Delivers:** `LoRAGradTrainer` with AdamW, cosine LR, gradient clipping, and `LongBenchNTPDataset` with left-truncation and pad-collate. A 10-step smoke test confirms loss decreases and LoRA params have non-zero gradients.

**Addresses:** F1 (gradient LoRA loop), partial m1 config, checkpoint save with Adam optimizer state.

**Avoids:** P1 (no-grad kills gradients — new class, never inside MemoryTrainer), P4 (generate() for training — dedicated NTP forward pass), P9 (mixed precision — no AMP), P10 (OOM — max_seq_len=1024 default, batch_size=1 + grad accumulation).

**Research flag:** Standard patterns — AdamW + PEFT + NTP teacher forcing is canonical. No additional research needed beyond STACK.md documentation. Implementation is mechanical.

### Phase 4: m1 Condition Run and Validation

**Rationale:** m1 (LoRA only, no NAMM, full cache) is the anchor of the comparison. Its checkpoint is also the input to m3 Stage 2 (NAMM training on LoRA-pretrained model). m3 cannot start until m1 is validated. Running m1 first also validates the training loop end-to-end on real QASPER data and establishes the empirical timing baseline.

**Delivers:** Trained m1 LoRA checkpoint on QASPER. Eval scores on all 3 LongBench tasks. wandb run with loss curve and gradient norm logged. Baseline comparison point for all subsequent conditions.

**Addresses:** m1 config + run (P1), training loss curve + gradient norm logging (P2), per-condition eval (P1).

**Avoids:** P6 (checkpoint handoff — m1 checkpoint is input to m3; must pass the checkpoint inspector), P3 (PEFT injection ordering — re-verify 32-adapter assertion in new trainer init), P7 (Adam state on resume — save optimizer state from the start).

**Research flag:** No research needed. m1 is the simplest condition. Run, validate loss decreases, confirm LoRA weight norms grow from initialization, save checkpoint.

### Phase 5: m4 Condition (NAMM Active During LoRA Training)

**Rationale:** m4 reuses the existing NAMM checkpoint (already available at `exp_local/.../ckpt.pt`), so it does not depend on m3. It can start immediately after Phase 3 infrastructure is in place. Running m4 before m3 validates the NAMM-active forward pass — the hardest integration — early, while m1's checkpoint matures for m3.

**Delivers:** Trained m4 LoRA checkpoint (NAMM frozen at best params, LoRA trained with NAMM evicting at cache_size=128). Eval scores on all 3 tasks. Gradient-norm assertion confirming LoRA gradients flow through retained tokens.

**Addresses:** F2 (NAMM-aware forward), m4 config (P1), core NAMM-LoRA interaction hypothesis.

**Avoids:** P2 (non-differentiable eviction — document and assert grad non-zero), P5 (NAMM state interferes with gradient graph — explicit torch.enable_grad in training path, NAMM runs under no_grad), P1 (no-grad kills gradients — inherited from Phase 3 trainer design).

**Research flag:** Needs a dedicated gradient-flow test before the phase is declared done. Specifically: confirm `lora_params[i].grad` is non-zero after the first m4 backward, and that gradient norms are smaller but non-zero compared to m1 (NAMM eviction removes some gradient paths). This is internal verification, not external research.

### Phase 6: Analysis Metrics (Attention Entropy and Token Retention)

**Rationale:** Add analysis hooks before running m3 so that all remaining conditions are logged with the primary metrics from the start. Running m3 without the hooks means a second eval pass to collect entropy data. These metrics are the mechanistic backbone of the paper.

**Delivers:** Forward hooks for per-layer attention entropy logged every eval_interval steps. Layer-wise token retention rate logged using the existing `record_eval_stats` flag on MemoryPolicy. Both metrics visible in wandb for m1 and m4 runs before m3 begins.

**Addresses:** F4 (attention entropy), F5 (token retention), analysis coverage for all conditions.

**Avoids:** P8 (entropy hooks interfere with PEFT hooks — register after get_peft_model(), hook at LlamaMemoryAttention module level not at q_proj; benchmark overhead before enabling for full training runs).

**Research flag:** Hook ordering with PEFT requires a one-step verification test before enabling: entropy values must differ between base model and LoRA-injected model on the same input. If they are identical, the hook is capturing pre-LoRA activations and the registration point is wrong. No external research needed.

### Phase 7: m3 Condition and Full Comparison Table

**Rationale:** m3 (LoRA then NAMM, sequentially) depends on the m1 checkpoint from Phase 4. Once m1 is done and analysis hooks are in place (Phase 6), m3 can run end-to-end. This completes the four main conditions and produces the core scientific output.

**Delivers:** Trained m3 checkpoint. Full four-condition comparison table (m1, m2, m3, m4) on all 3 LongBench tasks. Result CSV and wandb Summary table. The core scientific output of the v2.0 milestone.

**Addresses:** m3 config + run (P1), per-condition result table (P1), Stage 2 NAMM training using existing MemoryTrainer (unchanged).

**Avoids:** P6 (checkpoint handoff m1 to m3 Stage 2 — checkpoint inspector utility; assert lora_B.norm() > 0 on load), P7 (Adam state lost on resume — confirmed saved in Phase 3 implementation).

**Research flag:** m3 Stage 2 is standard MemoryTrainer NAMM CMA-ES training — well-understood, no research needed. Stage 1 is m1 training, already validated in Phase 4. Pipeline coordination in main.py follows the documented shell-script pattern.

### Phase 8: Secondary Experiments (E2, E4, Multi-Seed)

**Rationale:** Once m1-m4 are clean and the core comparison is interpretable, run the P2 secondary experiments. E2 (cache size sweep) and E4 (NAMM deactivation post-training) are config-only variations — no new code. Multi-seed runs require only repeating completed conditions with different seeds.

**Delivers:** E2: m3/m4 at cache_size=128/256/512. E4: m4 evaluated with NAMM deactivated post-training. Multi-seed error bars for paper-ready results.

**Addresses:** E2/E4 (P2 differentiators), multi-seed runs (P2).

**Research flag:** No research needed. All three are config variations or repeat runs of already-validated conditions.

### Phase 9: Paper Analysis and Interpretation

**Rationale:** After all experimental conditions are logged in wandb, the analysis phase produces static plots and interpretation. No new training runs unless E3/E5 experiments are triggered by the findings.

**Delivers:** Layer-wise LoRA weight norm heatmaps. Attention entropy vs. training step plots. m3 vs m4 retention rate comparison. Draft tables and figures for paper.

**Addresses:** Layer-wise weight norm heatmap (P2), NAMM eviction pattern comparison (conditional on findings).

**Research flag:** No research needed. Pure analysis of existing wandb data.

### Phase Ordering Rationale

- Phase 3 before all others: no gradient training without the training loop — hard technical dependency.
- Phase 4 (m1) before Phase 7 (m3): m3 Stage 1 is m1; checkpoint handoff requires m1 to be validated first.
- Phase 5 (m4) can proceed in parallel with Phase 4 (m1): m4 uses the existing NAMM checkpoint, not m1's output.
- Phase 6 (analysis hooks) before Phase 7 (m3): ensures m3 data is logged with full metrics from the start.
- Phase 8 (secondary experiments) after Phase 7: secondary experiments only make sense if the core comparison is interpretable.
- The existing Phase 2 LoRA seam (LORA-01 through LORA-04) is complete and is the direct dependency for Phase 3.

### Research Flags

Phases needing deeper verification during implementation (not external research — internal codebase testing):
- **Phase 5 (m4 NAMM-active forward):** Gradient isolation between NAMM eviction and LoRA backward needs an explicit test plan. Confirm `lora_params[i].grad is not None` after first m4 backward and that gradient norms are non-zero. This is the mechanistic core of m4.
- **Phase 6 (analysis hooks):** PEFT hook ordering with `LlamaMemoryAttention` is untested. Requires a one-step smoke test before enabling hooks in any training run.

Phases with standard patterns (skip additional research):
- **Phase 3 (gradient training loop):** AdamW + PEFT + NTP is canonical and thoroughly documented in STACK.md.
- **Phase 4 (m1 run):** Simplest condition; no new integration points beyond Phase 3.
- **Phase 7 (m3 run):** Stage 2 is existing MemoryTrainer; Stage 1 is validated m1 training.
- **Phase 8 (secondary experiments):** All are config variations of validated conditions.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All packages measured in th2; no new installs required; compute time estimates extrapolated (LOW confidence for gradient timing specifically — will update after first Phase 4 run) |
| Features | HIGH | Direct codebase inspection + published literature; feature dependencies fully mapped; open questions on NTP masking strategy and optimal LR for m4 are explicitly flagged |
| Architecture | HIGH | Based on complete codebase analysis of all modified files; component boundaries are explicit; data flow for m4 gradient step is traced through source code line-by-line |
| Pitfalls | HIGH | 10 pitfalls identified; 7 from direct codebase inspection (PRIMARY confidence); 2 from PEFT/PyTorch docs (MEDIUM); 1 from theoretical inference requiring empirical validation |

**Overall confidence:** HIGH

### Gaps to Address

- **NTP loss masking strategy (open question):** Full-sequence NTP vs. answer-token-only masking. Answer-only is standard but QASPER answers are often 1-3 tokens, which may produce insufficient gradient signal. Resolution: run a pilot comparing both strategies on 10% of data in Phase 4 before committing. Default to full-sequence NTP with an `answer_only_masking` config flag.

- **LoRA learning rate for m4 (open question):** Standard LR (2e-4) may be too high when NAMM is active because eviction creates gradient discontinuities making the loss landscape noisier. May need lower LR (2e-5) for m4. Resolution: log gradient norms in m4 from the first training run; if norms are erratic or loss diverges, halve the LR.

- **Number of gradient steps before overfitting (open question):** With ~180 training samples, overfitting risk is real. Use a train/val split (160/20) and monitor validation NTP loss to determine early stopping. Flagged in FEATURES.md as needing empirical pilot in Phase 4.

- **NAMM params for m4 LoRA training (open question):** Use best single-population params from existing checkpoint (not mean params). This matches the `_evaluate` code path. Should be verified empirically by comparing m4 trained with best-params-frozen vs mean-params-frozen on one eval task.

- **Compute timing for gradient training (LOW confidence estimate):** The ~21-minute estimate for 500 gradient steps is extrapolated from CMA-ES timing, not from an actual gradient training run. The first Phase 4 run will establish real timing. Budget for up to 2x variance.

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection: `memory_trainer.py` (`@torch.no_grad()` at 18+ locations), `memory_llms/llama.py` (`apply_lora_adapters`, forward with NTP loss path), `memory_llms/base.py` (`get_lora_params_flat`), `stateless_parallel_modules/attention.py` (SDPA, no attention weight output), `memory_policy/deep_selection.py` (`BinarySelection.select_new_tokens()` uses topk + index-select — non-differentiable), `memory_policy/base_dynamic.py` (`threshold_score_idxs` uses torch.topk), `memory_evaluator.py` (generation-only path, no NTP loss interface)
- PyTorch 2.7 docs: `torch.optim.AdamW` signature, `@torch.no_grad()` inheritance semantics, `torch.enable_grad()` context manager behavior
- Transformers 4.41.2 source: `LlamaAttention.output_attentions` behavior, `get_cosine_schedule_with_warmup` availability
- PEFT 0.11.1 source: hook registration order in `tuners/lora/layer.py`, `enable_input_require_grads` absence confirmed
- Measured QASPER token lengths: LongBench dataset, 200 samples, mean=3619, max=14660

### Secondary (MEDIUM confidence)
- ALTER (arXiv 2025): attention entropy as diagnostic for LoRA effects on token selection — supports entropy logging approach
- CS224R Stanford (2025): representation drift via attention entropy during RL finetuning — supports entropy as primary mechanistic metric
- Attention-Gate (arXiv 2410.12876, 2025): finetuning reshapes token eviction patterns — supports the hypothesis that LoRA changes NAMM behavior
- LLaMA-3 paper: beta2=0.95 for LLM finetuning
- Lightning AI 300-experiment LoRA study: lr=2-3e-4 optimal range for LoRA on LLMs
- PEFT GitHub issues #23170, #1142: gradient checkpointing + frozen params incompatibility — justifies gradient checkpointing as an anti-feature
- PyTorch forum: `no_grad` decorator edge cases with `enable_grad` context manager

### Tertiary (LOW confidence)
- Attention entropy hook interference with PEFT hooks — inferred from hook registration documentation; not empirically tested in this codebase
- m4 NAMM token-selection bias on LoRA representation learning — theoretical inference; empirical validation required via m3 vs m4 comparison
- Gradient training compute timing (~21 minutes per 500 steps) — extrapolated from CMA-ES measured timing; not measured directly

---
*Research completed: 2026-03-03*
*Supersedes v1.0 SUMMARY.md (ES-based LoRA, now archived in git history)*
*Ready for roadmap: yes*
