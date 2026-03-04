# Roadmap: Joint NAMM + LoRA-ES Training

## Milestones

- [x] **v1.0 Branch + LoRA Seam** - Phases 1-2 (complete 2026-03-02)
- [ ] **v2.0 NAMM + Gradient LoRA Study** - Phases 3-8 (in progress)

## Phases

<details>
<summary>v1.0 Branch + LoRA Seam (Phases 1-2) - COMPLETE 2026-03-02</summary>

### Phase 1: Branch Setup
**Goal**: The working branch exists, the existing NAMM trainer runs cleanly on it, and there is a verified starting point for all subsequent work.
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01
**Success Criteria** (what must be TRUE):
  1. `git branch dev/joint-namm-lora-es` exists locally, forked from `dev/llama_1b_namm` at commit 21b2880
  2. The existing QASPER NAMM eval script runs to completion on the new branch and reproduces the known QASPER score (within 0.1 F1) without any code changes
  3. `git log --oneline` shows the new branch head is one commit ahead of `dev/llama_1b_namm` (branch-creation commit only, no stray changes)
**Plans**: 1 plan

Plans:
- [x] 01-01-PLAN.md — Verify branch state, run NAMM smoke eval on GPU, commit phase-1 marker

### Phase 2: LoRA Seam + Correctness Gate
**Goal**: LLaMA 3.2-1B accepts LoRA weights injected from a flat float32 vector, checkpoints save and restore LoRA state, and four critical correctness pitfalls (injection ordering, merge corruption, silent zero-injection, bfloat16 underflow) are caught by passing unit tests before any training attempt.
**Depends on**: Phase 1
**Requirements**: LORA-01, LORA-02, LORA-03, LORA-04
**Success Criteria** (what must be TRUE):
  1. Running the unit test suite produces zero failures; the test log shows the expected LoRA module count (`num_layers * len(target_modules)`) = 32 for the default q_proj+v_proj config (LLaMA 3.2-1B has 16 hidden layers, not 14)
  2. Saving a checkpoint then loading it into a fresh model instance yields bit-identical LoRA weights and restores the LoRA config (rank, target modules); a NAMM-only checkpoint loads without error under the same code (graceful fallback verified)
  3. Running a full population evaluation cycle (pop_size=8, batch_size=4) leaves base model weights bit-for-bit unchanged, confirmed by a checksum assertion in the test suite
  4. LoRA weights are stored in float32 in the checkpoint (not bfloat16), confirmed by inspecting `ckpt['lora_state_dict']` dtypes in the test log
**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md — Add apply_lora_adapters() to WrappedLlamaForCausalLM; add get_lora_params_flat() / set_lora_params() to MemoryModelWrapper (LORA-01, LORA-02) [2026-03-02]
- [x] 02-02-PLAN.md — Extend _save_ckpt / _load_ckpt with LoRA state dict, config, joint_es_mode; graceful fallback for NAMM-only checkpoints (LORA-03) [2026-03-02]
- [x] 02-03-PLAN.md — Write tests/test_lora_seam.py with 6 pytest tests covering all LORA-04 assertions; requires GPU on sideswipe/prowl (LORA-04) [2026-03-02]

</details>

---

### v2.0 NAMM + Gradient LoRA Study (In Progress)

**Milestone Goal:** Implement gradient-based LoRA finetuning alongside NAMM CMA-ES training, run all four main conditions (m1, m3, m4-frozen, m4-iterative) and secondary experiments, collect analysis metrics across all conditions, and produce a comparison table with multi-seed error bars that answers whether NAMM's presence during finetuning matters.

- [ ] **Phase 3: Gradient Training Loop** - Build LoRAGradTrainer, LongBenchNTPDataset, and artifact/eval infrastructure; gate with gradient-flow unit test; bake in FAIR-02 and ARTIFACT-01 from the start
- [ ] **Phase 4: m1 + m4-frozen Runs** - Run and validate LoRA-only (m1) and NAMM-active LoRA (m4-frozen) on QASPER; enforce FAIR-01 token budget; collect token retention metrics; produce anchor checkpoints
- [ ] **Phase 5: m4-iterative Run** - Build interleaving orchestration controller; run alternating NAMM CMA-ES and LoRA gradient training; validate after m4-frozen is confirmed working
- [ ] **Phase 6: Analysis Metrics + m3 Run** - Add post-hoc attention entropy script; run m3 two-stage pipeline using m1 checkpoint; produce four-condition comparison table
- [ ] **Phase 7: Secondary Experiments** - Run E2 (cache sweep), E3 (dataset variation), E4 (NAMM deactivation), E5 (general-text NTP) as config-driven variations
- [ ] **Phase 8: Multi-seed Reproduction** - Rerun all four main conditions (m1, m3, m4-frozen, m4-iterative) with 2-3 seeds; export final comparison table with error bars

## Phase Details

### Phase 3: Gradient Training Loop
**Goal**: A standalone `LoRAGradTrainer` class and `LongBenchNTPDataset` exist with the correct artifact contract (ARTIFACT-01) and eval protocol (FAIR-02) baked in from the start; a 10-step smoke test confirms loss decreases and LoRA parameters have non-zero gradients; `main.py` routes to the trainer via Hydra config.
**Depends on**: Phase 2
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05, TRAIN-06, PIPE-01, FAIR-02, ARTIFACT-01
**Success Criteria** (what must be TRUE):
  1. Running a 10-step smoke test on the `LongBenchNTPDataset` with `LoRAGradTrainer` shows monotonically decreasing NTP loss over the 10 steps and `assert loss.requires_grad` passes after the first forward pass
  2. After the first backward pass, `all(p.grad is not None for p in lora_params)` is True; base model parameters have `grad is None` (frozen base confirmed); `all(p.dtype == torch.float32 for p in lora_params)` passes (no AMP downcast)
  3. A checkpoint saved by `LoRAGradTrainer` contains the AdamW optimizer state dict alongside the LoRA state dict; loading it into a fresh `LoRAGradTrainer` resumes without error (TRAIN-05)
  4. Setting `trainer_type: namm_es` routes to `MemoryTrainer`; setting `trainer_type: lora_grad` routes to `LoRAGradTrainer` — confirmed by the class name printed at run start; every run writes its artifacts into `results/{method}/{seed}/` with checkpoint, config YAML, metrics CSV, wandb run ID, and eval outputs present (ARTIFACT-01)
  5. All conditions share the same eval code path and wandb group (`Llama-3.2-1B/grad-lora-study`); a single `run_eval.py` entry point handles all conditions without per-condition branching (FAIR-02)
**Plans**: 4 plans

Plans:
- [x] 03-01-PLAN.md — Write `lora_ntp_dataset.py` with `LongBenchNTPDataset` — wraps HF LongBench splits into teacher-forced (input_ids, labels) pairs with left-truncation to max_seq_len and pad-collate (TRAIN-01) [2026-03-04]
- [ ] 03-02-PLAN.md — Write `lora_grad_trainer.py` with `LoRAGradTrainer` — AdamW, cosine LR with warmup, gradient clipping, gradient accumulation, NTP loss, wandb logging; PEFT gradient fix forward hook; checkpoint I/O with AdamW state; NAMM-active mode; artifact contract (TRAIN-02, TRAIN-03, TRAIN-05, TRAIN-06, ARTIFACT-01)
- [ ] 03-03-PLAN.md — Write `tests/test_lora_grad_trainer.py` with 6 pytest tests: loss.requires_grad, LoRA grad non-None, base grad None, float32 dtype, AdamW state saved/loaded, NAMM-active LoRA grads (TRAIN-04, TRAIN-05, TRAIN-06)
- [ ] 03-04-PLAN.md — Add `trainer_type` dispatch to `main.py`; create `cfgs/trainer/lora_grad.yaml` with locked hyperparameters; write `run_eval.py` skeleton enforcing shared eval protocol (PIPE-01, FAIR-02)

### Phase 4: m1 + m4-frozen Runs
**Goal**: The m1 condition (LoRA finetuning, no NAMM, full cache) and the m4-frozen condition (NAMM frozen at best params, LoRA trained with NAMM active and evicting at cache_size=128) both run end-to-end on QASPER using the same total token budget (FAIR-01), produce validated checkpoints, and deliver eval scores on all three LongBench tasks; token retention is logged during m4-frozen training.
**Depends on**: Phase 3
**Requirements**: PIPE-02, PIPE-04, FAIR-01, ANLYS-01, EXP-01, EXP-03
**Success Criteria** (what must be TRUE):
  1. Both m1 and m4-frozen training runs complete on GPU (sideswipe/prowl); wandb shows `train/loss` and `train/grad_norm` time series with non-constant values for each; token budgets match within 1% (same number of gradient updates on the same total tokens, confirmed from run logs) — FAIR-01 enforced
  2. Both checkpoints pass the inspector: `lora_B.weight.norm() > 0` (LoRA trained, not zero-initialized) and `lora_config` fields match the run config; m4-frozen NAMM policy param norms are identical between iteration 0 and final iteration (NAMM is frozen, not updated)
  3. Per-layer token retention rates appear in wandb as `retention/layer_{i}` time series during m4-frozen training; values are strictly between 0.0 and 1.0 (NAMM is actively evicting tokens) — ANLYS-01
  4. Evaluating both checkpoints on all three LongBench tasks produces scores logged to wandb under `Llama-3.2-1B/grad-lora-study`; both runs have complete artifact sets in `results/m1/{seed}/` and `results/m4_frozen/{seed}/` (ARTIFACT-01 contract satisfied)
**Plans**: TBD

Plans:
- [ ] 04-01: Write `cfgs/run/m1_lora_only.yaml` — LoRA gradient training, full cache (namm_active: false), QASPER training split; set token budget (num_steps x batch_tokens) as the shared FAIR-01 baseline (PIPE-02, FAIR-01)
- [ ] 04-02: Write `cfgs/run/m4_frozen.yaml` — loads existing NAMM checkpoint, freezes NAMM params, enables namm_active: true, cache_size=128, same token budget as m1; add token retention logging hooks to `LoRAGradTrainer` for namm_active mode (PIPE-04, ANLYS-01)
- [ ] 04-03: SSH to sideswipe/prowl; run m1 training; verify loss curve and checkpoint inspector; eval on all three tasks; save artifact set (EXP-01)
- [ ] 04-04: SSH to sideswipe/prowl; run m4-frozen training; verify gradient norms non-zero through retained tokens; verify NAMM params frozen; eval on all three tasks; save artifact set (EXP-03)

### Phase 5: m4-iterative Run
**Goal**: An interleaving orchestration controller alternates between NAMM CMA-ES steps and LoRA gradient steps at a configurable frequency; the m4-iterative condition runs end-to-end on QASPER and produces a validated checkpoint and eval scores comparable to m4-frozen.
**Depends on**: Phase 4
**Requirements**: TRAIN-07, PIPE-05, EXP-04
**Success Criteria** (what must be TRUE):
  1. The interleaving controller exists as a standalone orchestration class; setting `interleave_freq: N` in config causes it to run N LoRA gradient steps then 1 NAMM CMA-ES step, cycling until the total token budget is exhausted; the wandb run shows alternating `namm/fitness` and `lora/loss` log entries confirming both are updating
  2. After each NAMM CMA-ES step within the interleaved run, NAMM fitness is non-decreasing (NAMM is learning, not degrading under LoRA interference); after each LoRA gradient step, `all(p.grad is not None for p in lora_params)` would pass (LoRA is still receiving gradient signal)
  3. The m4-iterative checkpoint passes the inspector; eval on all three LongBench tasks produces scores logged to wandb alongside m1 and m4-frozen; artifact set is complete in `results/m4_iterative/{seed}/`
**Plans**: TBD

Plans:
- [ ] 05-01: Write interleaving orchestration controller (`lora_namm_interleaver.py`) — alternates `LoRAGradTrainer` gradient steps and `MemoryTrainer` CMA-ES steps with configurable frequency; coordinates checkpoint handoff between the two trainers each cycle; respects total token budget for gradient steps (TRAIN-07)
- [ ] 05-02: Write `cfgs/run/m4_iterative.yaml` — uses interleaving controller, configures interleave_freq, same FAIR-01 token budget for LoRA steps; same token budget constraint as m4-frozen (PIPE-05)
- [ ] 05-03: SSH to sideswipe/prowl; run m4-iterative; monitor both fitness and loss curves in wandb; eval checkpoint on all three tasks; save artifact set (EXP-04)

### Phase 6: Analysis Metrics + m3 Run
**Goal**: A post-hoc attention entropy analysis script loads any checkpoint and produces per-head entropy comparisons across conditions; the m3 two-stage pipeline (LoRA finetuning then NAMM CMA-ES) runs using the m1 checkpoint as handoff; a four-condition comparison table (m1, m3, m4-frozen, m4-iterative) plus the existing m2 baseline is exported as CSV and logged to wandb.
**Depends on**: Phase 4, Phase 5
**Requirements**: ANLYS-02, ANLYS-03, PIPE-03, EXP-02
**Success Criteria** (what must be TRUE):
  1. Running `python analysis/attention_entropy.py --ckpt [path]` outputs a per-head entropy CSV; entropy values measurably differ between the base model and an m1/m4-frozen checkpoint on the same input (script is capturing LoRA-modified activations, not pre-LoRA)
  2. The m3 Stage 2 input checkpoint passes the inspector: `lora_B.weight.norm() > 0` confirming the m1-trained LoRA weights are present; the NAMM CMA-ES Stage 2 run completes with a non-flat `fitness/best` curve (NAMM learns on top of the LoRA-finetuned model)
  3. The five-condition comparison table (m1, m2, m3, m4-frozen, m4-iterative) on all three LongBench tasks exists as both a wandb Summary table and a local `results/comparison_all_conditions.csv` file; every cell is populated (no missing scores)
**Plans**: TBD

Plans:
- [ ] 06-01: Write `analysis/attention_entropy.py` — loads checkpoint, patches LlamaAttention to output weights (non-SDPA path), computes per-head entropy per layer, saves CSV; smoke-test that entropy differs between base and finetuned model on the same input (ANLYS-02)
- [ ] 06-02: Write `cfgs/run/m3_lora_then_namm.yaml` and pipeline shell script — Stage 1 uses m1 config, Stage 2 passes m1 checkpoint path to MemoryTrainer init_from; add checkpoint inspector validation at handoff (PIPE-03)
- [ ] 06-03: SSH to sideswipe/prowl; run m3 pipeline (Stage 1: LoRA, Stage 2: NAMM CMA-ES); eval final m3 checkpoint on all three tasks; save artifact set (EXP-02)
- [ ] 06-04: Write comparison table script — aggregates wandb eval runs for all five conditions into CSV and wandb Summary table; include all three task columns (ANLYS-03)

### Phase 7: Secondary Experiments
**Goal**: E2 (cache size sweep on m4-frozen), E3 (dataset variation — m1 and m4-frozen on NarrativeQA and PassageRetrieval), E4 (NAMM deactivation — eval m4-frozen checkpoint with NAMM off), and E5 (general-text NTP finetuning) are run as config-driven variations and their results are logged and added to the results directory.
**Depends on**: Phase 6
**Requirements**: EXP-05, EXP-06, EXP-07, EXP-08
**Success Criteria** (what must be TRUE):
  1. Four m4-frozen runs at cache_size={64, 128, 256, 512} complete on GPU; wandb shows four separate runs each logging eval scores on all three tasks; QASPER score varies across cache sizes (not all identical), indicating NAMM sensitivity to memory budget (E2)
  2. m1 and m4-frozen runs using NarrativeQA and PassageRetrieval as training tasks complete and produce checkpoint + artifact sets; at least one condition shows a different relative ordering across training datasets (E3 reveals dataset structure sensitivity or confirms robustness)
  3. Evaluating the m4-frozen checkpoint with `namm_active: false` (full cache passthrough) produces QASPER/NarrativeQA/PassageRetrieval scores logged to wandb; the delta between m4-frozen-with-NAMM and m4-frozen-without-NAMM is recorded in the results CSV (E4 quantifies NAMM dependence at inference time)
  4. The E5 general-text NTP run completes using a subset of C4/RedPajama as training data; eval on all three tasks produces scores logged to wandb; the result can be compared to m1 (task-specific NTP) to assess the value of task-specific finetuning (E5)
**Plans**: TBD

Plans:
- [ ] 07-01: Write `cfgs/run/e2_cache_sweep.yaml` — m4-frozen config with cache_size as a Hydra sweep variable over {64, 128, 256, 512}; SSH to GPU and run sweep (EXP-05)
- [ ] 07-02: Write `cfgs/run/e3_narrativeqa.yaml` and `cfgs/run/e3_passageret.yaml` — m1 and m4-frozen configs with training task swapped to NarrativeQA and PassageRetrieval respectively; SSH to GPU and run (EXP-06)
- [ ] 07-03: Write `cfgs/run/e4_namm_deactivation.yaml` — m4-frozen checkpoint eval with namm_active: false; run eval and add delta column to results CSV (EXP-07)
- [ ] 07-04: Write `cfgs/run/e5_general_text.yaml` — LoRA training on C4/RedPajama subset (add `GeneralTextNTPDataset` if needed); eval on all three tasks; log to wandb (EXP-08)

### Phase 8: Multi-seed Reproduction
**Goal**: The four main conditions (m1, m3, m4-frozen, m4-iterative) are rerun with 2-3 random seeds; results are reported with mean and standard deviation across seeds; a final comparison table with error bars is exported as the definitive v2.0 scientific output.
**Depends on**: Phase 7
**Requirements**: REPRO-01, EXP-09
**Success Criteria** (what must be TRUE):
  1. Each of the four main conditions has at least 2 completed runs with different seeds (seeds from {1337, 42, 0}); each run has its artifact set in `results/{method}/{seed}/`; the wandb group shows at minimum 8 runs (4 conditions x 2 seeds) with complete eval scores
  2. The final comparison table (`results/final_comparison.csv`) contains one row per condition, with mean and standard deviation columns for each of the three tasks; standard deviations are non-zero (runs with different seeds produce different scores, confirming seed sensitivity is measured, not assumed zero)
  3. The comparison table is reproduced by a single deterministic script (`analysis/make_final_table.py`) that reads only from `results/` artifact directories — not from wandb API calls — so it can be regenerated offline
**Plans**: TBD

Plans:
- [ ] 08-01: Write multi-seed runner script — iterates over seeds {1337, 42, 0} and methods {m1, m3, m4-frozen, m4-iterative}; skips already-completed seed/method combinations (idempotent); SSH to GPU and run all outstanding combinations (REPRO-01, EXP-09)
- [ ] 08-02: Write `analysis/make_final_table.py` — reads artifact directories, computes mean/std per condition per task, exports `results/final_comparison.csv` and a markdown-formatted table for the paper; smoke-test by regenerating from existing single-seed results (REPRO-01, EXP-09)

## Progress

**Execution Order:**
Phases execute in numeric order: 3 → 4 → 5 → 6 → 7 → 8
Note: Phase 5 (m4-iterative) depends on Phase 4 (m4-frozen validated) — must not start before m4-frozen is confirmed working. Phase 6 (m3 + analysis) depends on Phase 4 (m1 checkpoint) and Phase 5 (m4-iterative results available for comparison table).

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Branch Setup | v1.0 | 1/1 | Complete | 2026-03-02 |
| 2. LoRA Seam + Correctness Gate | v1.0 | 3/3 | Complete (GPU test pending) | 2026-03-02 |
| 3. Gradient Training Loop | v2.0 | 1/4 | In progress | - |
| 4. m1 + m4-frozen Runs | v2.0 | 0/4 | Not started | - |
| 5. m4-iterative Run | v2.0 | 0/3 | Not started | - |
| 6. Analysis Metrics + m3 Run | v2.0 | 0/4 | Not started | - |
| 7. Secondary Experiments | v2.0 | 0/4 | Not started | - |
| 8. Multi-seed Reproduction | v2.0 | 0/2 | Not started | - |
