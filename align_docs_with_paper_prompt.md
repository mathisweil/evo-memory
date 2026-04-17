# Align project documentation with the final paper

## Goal

The repo's three project documents — `README.md`, `experiment_specification.md`, and `analysis_specification.md` — were written during the research phase and reflect an earlier version of the project (different terminology, more experiments, more analyses than the paper ended up containing). The paper has now been finalised and submitted to TACL. Your task is to review all three documents against the actual paper, then update them so that a reader who clones this repo can reproduce exactly what is in the paper, without being misled by orphan sections describing experiments or analyses that never made the cut.

**Inputs assumed to be provided to you:**
- The final paper PDF (attached by the user).
- The three documents above, in the repo root.
- The refactored `scripts/` layout from the previous refactoring task.

Do not start editing until you have completed Phase 1.

---

## Phase 1 — Read everything first

1. **Read the paper end-to-end.** Extract: the exact model variants and inference regimes named in the paper, every experimental configuration that appears in a figure or a numerical claim, every analysis that appears in Section 5 or the appendices, and every dataset/split/hyperparameter detail.
2. Read `README.md`, `experiment_specification.md`, `analysis_specification.md` in full.
3. Build a **mapping table**: each paper figure/table/numerical claim → which doc section currently describes how to reproduce it → whether that description is accurate, partially accurate, or missing.
4. Build an **orphan list**: doc sections that describe experiments or analyses that are *not* in the paper.
5. Report both back before making any edits.

---

## Key paper facts (use these to spot mismatches, don't copy them verbatim)

### Terminology — this is the biggest source of drift

The project docs use **M1 / M2 / M3 / M4** nomenclature throughout. The paper uses a different scheme:

| Paper term | Meaning | Old doc term (approx.) |
|---|---|---|
| **Base** | pre-trained Llama-3.2-1B, no fine-tuning | B0 |
| **FTS** | fine-tuned standard: LoRA trained with full cache | M1 |
| **FTE** | fine-tuned with eviction: LoRA trained with NAMM active | M3 |
| **FC** (inference regime) | full cache at eval | — |
| **EC** (inference regime) | evicted cache at eval (NAMM applied) | — |

Configurations are named `{variant}-{regime}`, e.g. `FTS-EC` = LoRA-trained-under-full-cache, evaluated with NAMM active. The paper's main result (Figure 1) reports six such configs: Base-FC, Base-EC, FTS-FC, FTS-EC, FTE-FC, FTE-EC.

There is no M2-as-a-model-variant in the paper — NAMM is treated as a fixed eviction policy with its own training procedure, not a model condition. There is no M4 in the paper — joint NAMM + LoRA training is explicitly framed as **future work** in Section 6 / Section 5.3's closing paragraph.

### What the paper actually contains

**Section 5.1 — Main result.** F1 bar chart across all six configs (Base-FC, Base-EC, FTS-FC, FTS-EC, FTE-EC, FTE-FC). One cache size (K=1024). One NAMM checkpoint. Single test split.

**Section 5.2 — Performance attribution.** Three analyses:
- Next-token logit JS divergence between FC and EC pairs, per model variant.
- Per-layer ℓ2 hidden-state drift under FC → EC, ratio of FTE drift to FTS drift.
- Attention mass directed at the first / middle / last third of the prompt, FTE-FC minus FTS-FC, per layer.

**Section 5.3 — Eviction mask stability.** Pairwise mean Jaccard index between retained token sets under Base-EC, FTS-EC, FTE-EC.

**Appendix A — Additional eviction mask figures.** Per-layer fraction of retained prompt positions (union over heads) for Base/FTS/FTE under the frozen NAMM; NAMM raw score distributions pooled across the test split with pairwise KS distances.

**Appendix B — Hyperparameters.** NAMM CMA-ES config (pop size 8, 200 iters, BAM scoring, K=1024). LoRA config (rank 8, alpha 16, 150 epochs, effective batch 16, bfloat16).

**Appendix C — Implementation differences from the original NAMM.** Cache-size-based (top-K) eviction rule vs. the original threshold-only rule; an attention-mask bug fix in the Sakana reference implementation.

### Datasets & splits

Three sources: Qasper, 2WikiMultihopQA, HotpotQA — plus LongBench-E variants of Qasper and 2WikiMultihopQA, yielding five sub-corpora. Filtered to 4096–6500 tokens → 440 sequences, stratified 70/15/15 with `split_seed=42`. Sampling weight is inversely proportional to source size during training. A separate **extended test set** of 154 examples with lengths 6500–8192 tokens is used *only* for out-of-distribution evaluation. Metric: token-level F1. Greedy decoding at eval (T=0).

### Things the current docs describe that are NOT in the paper

Flag every occurrence of these for deletion or demotion:

- **H2O eviction baseline** (`run_eval.py --run_config h2o_baseline_...`)
- **ScissorHands eviction baseline**
- **LoRA + H2O / LoRA + ScissorHands** combined runs
- **LoRA rank sweep (r=4, 8, 16)** — paper uses r=8 only, with no ablation.
- **Recency-only eviction baseline (B1)**
- **Truncation baselines** (Trunc/plain, Trunc/lora_m1)
- **Joint NAMM + LoRA training (M4)** — future work in paper, not a result. Keep the code, but demote any doc section that presents it as a completed experiment.
- **Ablation A4 "NAMM disabled at eval on M4 checkpoint"** — this concept partially survives in the paper as the FTE-FC configuration (FTE with NAMM off at eval), but the framing and checkpoint are different. Check before preserving.
- **Evolution Strategies (ES) training** — the paper exclusively uses LoRA gradient-based fine-tuning. ES code exists in the repo but should not feature in reproduction instructions; mention only briefly in the README as exploratory code.
- **Analyses §2 (adaptation rate), §4 (LoRA weight comparison), §5 (attention entropy), §6 (token importance alignment / NAMM-attention correlation), §7 (CKA), §8 (probing), §9 (gradient flow)** — none of these appear in the paper. §0–§1 partially survive as context but are not paper results.

### What the docs currently under-describe

The paper includes several analyses that do not have clean counterparts in the current `analysis_specification.md` and need adding:

- JS divergence of next-token logits (Section 5.2, Figure 2).
- Per-layer hidden state ℓ2 drift ratio (Section 5.2, Figure 3).
- Attention mass by prompt-third (Section 5.2, Figure 4).
- Jaccard index across retained-token sets (Section 5.3, Figure 5).
- Extended-test OOD evaluation on the 6500–8192-token set.

---

## Phase 2 — Decide: merge, keep-separate, or delete

Before editing anything, make a recommendation on the relationship between `experiment_specification.md` and `analysis_specification.md`:

- **Option A — merge into one `REPRODUCTION.md`.** Organise strictly by paper figure/table. "To reproduce Figure 1: run X, Y, Z. To reproduce Figure 3: run A, then compute B." Reader-centric, minimises overlap, matches how people actually use paper code.
- **Option B — keep them separate but purge both.** `experiment_specification.md` covers data generation (the training/eval runs that produce checkpoints and results.json files); `analysis_specification.md` covers post-hoc analyses that consume those artifacts. Contributor-centric, preserves the pipeline structure.
- **Option C — delete `analysis_specification.md` entirely** and fold its surviving content (the four paper analyses above) into an "Analyses" section at the end of `experiment_specification.md`.

Consider especially: given how much of the current `analysis_specification.md` is dead (analyses 2, 4–9 either not in the paper or never completed), Option A or C may be cleaner than Option B. State your recommendation with reasoning and wait for approval before executing the structural change. After approval, proceed to Phase 3.

---

## Phase 3 — Edit

Whichever structural option was approved in Phase 2:

### Terminology pass (applies to all three docs)
Replace M1/M2/M3/M4 → Base/FTS/FTE consistently. Where a config file is still named `m3_lora_frozen_namm_5t.yaml` on disk, keep the filename but annotate it as "(FTE — fine-tuning under eviction)". Do not rename config files as part of this pass — that's a separate refactor.

### `README.md`
- Update the **Project Structure** tree to reflect the post-refactor `scripts/` layout (`run/`, `analysis/`, `reporting/`, `infra/`).
- Update every example command (`scripts/run_lora.py` → `scripts/run/run_lora.py`, etc.).
- Remove or demote references to: H2O, ScissorHands, LoRA rank sweep, recency, truncation, joint (M4).
- Add a brief "Paper" section near the top: one paragraph summarising what the paper shows, plus a pointer to `experiment_specification.md` / `REPRODUCTION.md` for reproduction details.
- Keep the Setup, Configuration, Dependencies, and TPU sections as-is unless path changes affect them.

### `experiment_specification.md` (or merged doc)
- Rename terminology throughout.
- Delete sections describing H2O, ScissorHands, LoRA rank sweep, recency, truncation, M4 joint training (move to an "Exploratory / Not in Paper" appendix if you want to preserve them for the curious reader, otherwise delete).
- Confirm that every command block reflects the new `scripts/run/...` paths.
- Add a "Reproducing the paper" section at the top that lists the minimum set of runs needed: Base eval, FTS training + eval under FC and EC, FTE training + eval under FC and EC, NAMM training. Link each to the paper figure/table it feeds.
- Keep FAIR-01 constraints — the paper implicitly follows them, and they're still useful.

### `analysis_specification.md` (if it survives Phase 2)
- Delete dead analyses (§2, §4–§9, possibly §0, §1 — check each against the paper).
- Add the four paper analyses that are missing: JS divergence, hidden state drift, attention thirds, Jaccard index. For each, describe: what it consumes, what it produces, which paper figure/table it maps to, a minimal command to reproduce it.
- Add the extended-test OOD evaluation.
- Remove the "Analysis Status" table — after purging, everything left is "done".
- Remove the `analysis/_summary_report.md` and `analysis/_meta-analysis.md` pointers at the end unless those files still exist and are worth referencing.

---

## Phase 4 — Cross-checks

1. Grep all three docs for `M1`, `M2`, `M3`, `M4` as standalone tokens. Any remaining hits should be either (a) inside a filename that legitimately still has that name on disk, or (b) a deliberate historical reference. Report hits.
2. Grep for `H2O`, `ScissorHands`, `scissorhands`, `h2o`, `recency`, `rank sweep`, `truncation`. Any remaining hits should be justified.
3. Grep for `joint`, `M4`, `m4_joint`. These should only appear in clearly-demarcated "future work" or "exploratory" context.
4. For every numerical claim in the paper (F1 values in Figure 1, JS divergence values in Figure 2, Jaccard values in Figure 5, KS distances in Appendix A), verify that the corresponding reproduction instructions in the docs would plausibly produce that number. You are not re-running anything — you are checking that the documented commands point at the right checkpoints, data splits, and metrics.
5. Confirm every `scripts/` path in every doc resolves to a real file under the refactored layout.

---

## Constraints

- **Do not create any commits.** Leave changes staged or unstaged for user review. No `git commit`, `git push`, or branching.
- Do not modify the paper PDF or the `scripts/` folder. This task is documentation only.
- Do not rename config YAML files, checkpoint files, or output directory structures. Filenames that still say `m3_...` or `m4_...` can be annotated but not moved.
- Preserve anything in the docs that is factually correct *and* reflected in the paper, even if its phrasing feels dated.
- If you are uncertain whether a section should be deleted, demoted to an appendix, or kept, **flag it and ask** rather than act.
- Do not fabricate numbers. If the paper reports F1=28.9 for FTE-EC, do not paraphrase as "~29" or "around 29"; quote as 28.9 or leave the number out of the doc entirely.

---

## Final deliverable

A summary report containing:
1. The Phase 1 mapping table (paper artifact → doc section → status).
2. The Phase 1 orphan list (doc sections that don't correspond to anything in the paper).
3. Your Phase 2 recommendation (merge / keep-separate / delete) with reasoning.
4. A diff-style summary of what was added, removed, renamed, or moved in each of the three docs.
5. Output of each Phase 4 cross-check.
6. A list of anything you flagged and didn't resolve.
7. A suggested commit message (or two) for the user to use.
