# Section C — Eviction Mask Drift Analysis: Implementation Task

You are working in the `evo-memory` repository, which studies adapting LLMs to KV-cache eviction (specifically, NAMM — Neural Attention Memory Models). I need your help to implement the instrumentation, analyses, and figures for **Section C** of our paper, which studies **how much the frozen NAMM's eviction behaviour changes when the underlying LLM is fine-tuned**.

This is a three-phase task: **(1) understand the repo, (2) write a plan and get it approved implicitly by proceeding, (3) implement it and produce runnable commands.** Do not skip phase 1.

---

## Background: what Section C is about (read this carefully)

The paper's method (§3.3) proposes fine-tuning the base LLM with a *frozen* NAMM in the loop. NAMM produces eviction masks by computing features from the LLM's attention activations. The concern the paper explicitly flags and leaves to experiments:

> fine-tuning the LLM changes the representations used by the eviction rule, so the effective conditioning distribution induced by eviction can itself drift during optimization.

In other words: even though NAMM's weights are frozen, fine-tuning the LLM changes the *inputs* to NAMM (attention activations), so NAMM will evict different tokens. If that drift is large, the training objective is chasing a moving target and the method is unstable.

We are NOT going to retrain anything. Training checkpoints mid-run are not available. Instead we answer the concern as an **endpoint comparison**: for one fixed frozen NAMM, we compare the eviction masks it produces when the underlying LLM is:

- **B0**: base pre-trained Llama-3.2-1B-Instruct (no LoRA)
- **M1**: LoRA fine-tuned without NAMM in the loop (plain task fine-tune)
- **M4**: LoRA fine-tuned jointly with NAMM active (our method)

The same NAMM checkpoint is used in all three conditions. The underlying LLM is the only thing that changes. This directly answers "how much does fine-tuning move NAMM's eviction decisions?"

---

## What the paper already has (so you know what to connect to)

The paper already contains, in Section 5.4:

- **NAMM–attention alignment (Spearman ρ between NAMM scores and LoRA-layer attention):** M1 = −0.115, M2 = −0.151, M4 = −0.168. This is adjacent to what Section C does and you should make sure the new analyses are consistent with these numbers (ideally reproducible from the same dumped tensors).
- **Hidden-state drift analysis (Section 5.3):** per-layer L2 distance between hidden states, showing drift is concentrated in the last ~3 layers. Section C should check whether mask drift has a similar late-layer concentration.

All reported F1 numbers for the relevant endpoints are already computed and live in `eval_results/all_results_current_split.json`. Key numbers for context (micro F1 on the 70-prompt test split, cache_size=1024 where applicable):

- B0 full cache: 22.0
- M2 (base + NAMM cs1024): 12.6
- M1 full cache: 32.6
- M1 + NAMM cs1024: 19.0
- M4 + NAMM cs1024: 28.9
- A4 (M4 LoRA, full cache, NAMM ablated): 29.3

Key paths (verify them yourself — do not trust these blindly):

- NAMM checkpoint used throughout: `eval_results/namm_cs1024_maskfix/ckpt.pt`
- M1 LoRA checkpoint: `checkpoints_backup/lora_m1_lr1e4_matched/best_ckpt.pt`
- M4 LoRA checkpoint: `checkpoints_backup/lora_m4_cs1024_maskfix/best_ckpt_step260_val52.06.pt`
- Diagnostic script that likely already handles some of this: `scripts/check_eviction_stats.py`

---

## What I need Section C to produce

### Primary metrics

For a fixed set of prompts (use the 70-prompt test split; the 154-prompt extended-test is a nice-to-have for the appendix), under a fixed NAMM, for each of {B0, M1, M4} as the underlying LLM, log and then compare:

1. **Binary keep/evict masks** — per layer, per head, per eviction step, which token positions NAMM retains. This is the primary object.
2. **Raw NAMM scores** — the scalar score per token, per layer, per head, per eviction step, *before* the threshold/top-k cutoff is applied. Needed to distinguish "masks agree because scores barely moved" from "masks agree by coincidence near the decision boundary."
3. **Per-layer retention rate** — fraction of tokens kept at each layer (averaged over heads and prompts).

### Plots (these are the actual figures I'll put in the paper)

- **C1 — Mask-overlap summary (headline).** Pairwise average Jaccard/IoU between retained-token sets for {B0, M1, M4}, averaged over prompts, heads, and eviction steps. Present EITHER as a 3×3 heatmap OR as grouped bars by layer with three pairings (B0↔M1, B0↔M4, M1↔M4). Your call on which is clearer; do whichever reads best. Include the two anchoring reference lines described in "Baselines" below.
- **C2 — Per-layer retention profile.** X-axis = layer index, Y-axis = mean keep-rate, three lines (B0, M1, M4). This tells us whether LoRA shifts *how much* NAMM evicts, not just *what* it evicts.
- **C3 (appendix) — NAMM score distribution shift.** Histogram or KDE of raw NAMM scores under B0 vs M1 vs M4, with a vertical line at the decision threshold. Include a one-number summary (KS distance between each pair of distributions) in a small table or in-caption.
- **C4 (optional, appendix) — Per-layer IoU drift.** 1 − IoU as a function of layer index for B0↔M4. Compare visually to the hidden-state-drift figure from §5.3 to check whether mask drift is also late-layer concentrated.

### Baselines for interpretability (important — do not skip)

A raw IoU number is meaningless in isolation. Compute and plot as reference lines on C1:

- **Cross-prompt IoU within B0.** Average pairwise IoU of retained-token sets across *different prompts* under B0. This is NAMM's natural prompt-to-prompt variability. If fine-tuning-induced drift is smaller than this, the concern is minor.
- **Random-baseline IoU.** Expected IoU of two random masks with the same retention rate k/n. This is just `k/n` for top-k mode; derive it analytically and include it.

### Consistency check

Recompute the Spearman ρ between NAMM scores and mean LoRA-layer attention for M1 and M4 from the same dumped tensors. The results should match the paper's existing −0.115 / −0.168 (within noise). If they don't, flag it — it means either the existing numbers used a different subset or your logging hooks are wrong.

---

## Task for Claude Code

### Phase 1 — Understand the repo (do this first, do not skip)

Explore the repository and figure out:

1. The evaluation pipeline structure: how `scripts/run_eval.py` is invoked, what configs it consumes, how models are loaded, how a LoRA checkpoint is attached, how a NAMM checkpoint is attached, how eviction is actually applied at inference, how cache_size is enforced.
2. Where in the inference loop NAMM runs: at what granularity are scores computed, where is the top-k cutoff applied, where is the binary mask materialised, what tensor shapes are involved (layer × head × position or something else), how does "every 256 processed tokens" manifest in code.
3. What `scripts/check_eviction_stats.py` already does. If it already dumps masks or scores, you're extending it. If it only prints aggregate stats, you're adding tensor-dumping infrastructure to the eviction pipeline and a new analysis script.
4. How prompts are fed through the model during evaluation — chunked through NAMM or full-prompt. The project notes mention M1 processes the full prompt in one go while M2/M3 require chunking; verify this and handle it correctly in the logging code.
5. Which test split is the 70-prompt set and how to target exactly that split deterministically. The split is `train_frac=0.7, val_frac=0.15, split_seed=42, min_conditioning_length=4096, max_conditioning_length=6500`.
6. Where figures are generated today (`scripts/generate_paper_figures.py`?) and follow the same conventions for figure style so Section C figures match the existing paper figures aesthetically.

**Do not assume any of the above. Read the code.** Use `rg` / `grep` aggressively. Look at actual imports, function signatures, and tensor shapes. If something is unclear after reading, log it as an open question in the plan rather than guessing.

### Phase 2 — Write a plan

Produce a file `SECTION_C_PLAN.md` in the repo root describing:

1. **Concrete understanding of the eviction pipeline** based on what you found. Include: file paths, function names, where NAMM's mask/scores are available in memory, and where you will hook in.
2. **The instrumentation changes you will make.** For each file you intend to modify or create, describe: the purpose, the new function/flag, the data format of what's being dumped (tensor shapes, dtype, what keys in the saved dict). Prefer extending existing code over writing parallel new pipelines.
3. **The dump format.** I suggest one `.pt` (or `.npz`) file per (condition × prompt), saved under `eval_results/section_c_dumps/{B0,M1,M4}/prompt_{idx}.pt`, containing:
   - `mask`: `[n_eviction_steps, n_layers, n_heads, max_seq_len]` bool, True = kept. Pad with False.
   - `scores`: same shape, float. NaN where padding.
   - `retained_indices`: a list-of-lists for each (step, layer, head) if the dense tensor is too large — your call.
   - `prompt_meta`: `{task, prompt_length, source_file_id}`.
   - `config`: `{llm_id, lora_path, namm_path, cache_size}`.
   If this is wrong for the repo's data layout (e.g. NAMM operates at a different granularity), use what's right. Explain your choice.
4. **The analysis script(s)** you will add. Likely a single `scripts/analyze_mask_drift.py` that loads the dumps, computes all metrics, writes CSVs of the numbers, and regenerates the figures. The script must be deterministic and re-runnable. Figures go to `figures/section_c/`.
5. **The exact list of eval runs needed and the CLI commands that will run them.** Each run should write its dumps to a distinct output directory. These become what I actually execute.
6. **Open questions / risk flags.** Anything you couldn't resolve from the code, anything you're guessing about, anywhere the dump could be prohibitively large (NAMM masks across all layers × heads × steps × positions can be huge — if so, propose a compression strategy, e.g. sparse-index storage or sampling prompts/heads).
7. **How Section 5.4's Spearman numbers will be reproduced from the new dumps.** One paragraph.

Write this plan out fully before writing any code. Be specific — file paths, function names, tensor shapes, CLI flags. If the plan is vague, the implementation will be wrong.

### Phase 3 — Implement

After writing the plan, implement it. Concretely:

1. Add/modify the code to dump masks + scores during NAMM inference, gated behind a new CLI flag (something like `--dump_namm_state=<path>`) so it only runs when asked.
2. Add the analysis script `scripts/analyze_mask_drift.py` that consumes the dumps and produces:
   - `eval_results/section_c_metrics.json`: all numeric results (pairwise IoU means + stds, per-layer retention rates, KS distances, reference-baseline IoU values, Spearman reproduction).
   - `figures/section_c/C1_mask_overlap.{png,pdf}`
   - `figures/section_c/C2_retention_by_layer.{png,pdf}`
   - `figures/section_c/C3_score_distributions.{png,pdf}`
   - `figures/section_c/C4_iou_by_layer.{png,pdf}` (optional per your judgment)
3. Make sure the figures match the existing paper figure style (fonts, colour palette, sizing). Look at existing figures in `figures/` — `overall_mean_f1_test.pdf`, `per_task_f1_test.pdf`, `drift_ratio.pdf`, `hidden_states.png`, `attention_mass_shift.png` — and match them.
4. Add a short smoke test that runs the full pipeline on 2 prompts end-to-end to catch shape bugs before I spend GPU time on the full set.
5. Write a `SECTION_C_REPORT.md` in the repo root that explains:
   - What the figures show, numerically. Fill in the numbers once you have them from smoke-testing; put `[TODO: fill after full run]` placeholders where I need to run the real evals to populate.
   - How to interpret each figure (the one-line reading for a reviewer).
   - The interpretation template for each possible outcome (e.g. "if B0↔M4 IoU > cross-prompt IoU baseline, the concern is unfounded; if it's below, flag as limitation").
   - Any caveats you discovered during implementation.

### Phase 4 — Deliver the commands I will run

At the end, output a clearly-marked section titled **"Commands to run"** with every shell command I need to execute in order. Group them into:

1. **Smoke test** — fastest path to confirm nothing is broken (2 prompts, 1 condition).
2. **Full dumps** — the three real eval runs (B0+NAMM, M1+NAMM, M4+NAMM on the 70-prompt test set).
3. **Analysis & figures** — the single command that consumes the dumps and produces everything.
4. **(Optional) Extended-test dumps** — same as (2) but on the 154-prompt extended-test split, for appendix material.

Each command must be literal, copy-pasteable, and include all necessary config overrides. Do not write "something like"; write the exact command.

---

## Constraints and style

- Do not retrain anything. No NAMM training, no LoRA training. Read-only use of the existing checkpoints.
- Do not change any existing eval behaviour by default. All new logging must be off unless a flag is passed, so existing reproducibility is preserved.
- Match the repo's existing conventions (Hydra configs, logging style, output directory layout). Do not introduce a new config system.
- Keep the dump size reasonable. If the naive dense mask tensor is >5GB total across all runs, switch to sparse storage or sampled logging and flag the choice in the plan.
- The analysis must be fully deterministic given the dumps. Set seeds anywhere randomness could creep in (subsampling, KDE estimators, etc.).
- If you find the NAMM mask is already accessible from an existing API (e.g. as an attribute on the policy object after each forward), prefer hooking that over duplicating the computation. Do not recompute what the pipeline already has.

---

## What I want back from you

In order:

1. A summary of what you discovered about the repo in Phase 1 (a few paragraphs, the things that matter — not an exhaustive dump).
2. `SECTION_C_PLAN.md` as a concrete proposal.
3. The implementation (code changes, new script, smoke test).
4. `SECTION_C_REPORT.md` with interpretation templates.
5. The final "Commands to run" block.

Begin with Phase 1. Do not start writing code until you have explored the repo and written the plan.
