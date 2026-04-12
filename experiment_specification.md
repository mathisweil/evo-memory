# evo-memory — Experiment Specification

**Model:** Llama-3.2-1B-Instruct  
**Benchmark:** 5-task LongBench QA subset (token-level F1)  
**Hardware:** Single NVIDIA GPU  
**No pretrained NAMM checkpoint used** — all NAMM training is from scratch.

---

## 0 · Fairness Constraints (FAIR-01)

All conditions in the main results table must satisfy every constraint below. Any violation invalidates the comparison.

- **Data equivalence** — identical 5-task subset, splits, and filtering across all runs (`train_frac=0.7`, `val_frac=0.15`, `split_seed=42`; `min_conditioning_length=4096`, `max_conditioning_length=6500`, `max_answer_tokens=64`)
- **Model equivalence** — all conditions start from the same base Llama-3.2-1B-Instruct weights, no pretrained adapters
- **Memory equivalence** — all conditions evaluated with `cache_size=1024` at test time
- **Decoding equivalence** — greedy decoding (`temperature=0.0`) for all final evaluations

> **Compute anchor (gradient conditions):** M1 trains for 150 epochs with effective batch size 16. M2 standalone NAMM uses 200 CMA-ES generations.

---

## 1 · Recommended Execution Order

| Step | Experiment | Dependency |
|------|------------|------------|
| 1 | B0 — baseline eval | none |
| 2 | B1 — recency eviction eval | none |
| 3 | M1-LoRA — rank sweep (r=4, 8, 16) | none |
| 4 | M2 — standalone NAMM training | none |
| 5 | M3-LoRA — LoRA with frozen NAMM at train-time | M2 checkpoint |
| 6 | M4-LoRA — joint LoRA + NAMM | none |
| 7 | A1 — LoRA rank sweep analysis | M1-r4, M1-r8, M1-r16 |
| 8 | A4 — NAMM disabled at eval | M4-LoRA checkpoint |

---

## 2 · Tier 1 — Baselines

These require no training. Run first to establish reference scores.

---

### B0 · Base model, full KV cache

Evaluates raw Llama-3.2-1B-Instruct on the 5-task QA subset with no KV cache limit, no eviction, no fine-tuning. This is the absolute performance floor.

```bash
python scripts/run_eval.py \
    --run_config full_cache_baseline_llama32_1b \
    --output_dir experiments/experiment_N/baseline
```

| Parameter | Value |
|-----------|-------|
| `es_checkpoint` | null — evaluates base model weights |
| `namm_checkpoint` | null — no eviction, full KV cache |
| `run_config` | `full_cache_baseline_llama32_1b` — no-eviction policy |
| `cache_size` | null — no limit |
| Tasks | 5-task QA subset (override task config to `rh_multi_qa_5t`) |
| Splits | `train_frac=0.7`, `val_frac=0.15`, `split_seed=42` |
| Output | `experiments/experiment_N/baseline/results.json` |

---

### B1 · Base model, recency eviction

Evaluates the base model with a fixed recency policy: keeps the most recently written tokens, evicts oldest first. No learned policy, no training. Establishes what a naive heuristic achieves at `cache_size=1024`.

```bash
python scripts/run_eval.py \
    --run_config recency_baseline_llama32_1b \
    --cache_size 1024 \
    --output_dir experiments/experiment_N/es_recency/b1_recency \
    --override "task@_global_=rh_multi_qa_5t"
```

| Parameter | Value |
|-----------|-------|
| `run_config` | `recency_baseline_llama32_1b` — matches `config/run/recency_baseline_llama32_1b.yaml` |
| `cache_size` | 1024 |
| Tasks | 5-task QA subset (override task config to `rh_multi_qa_5t`) |
| Splits | `train_frac=0.7`, `val_frac=0.15`, `split_seed=42` |
| Output | `experiments/experiment_N/es_recency/b1_recency/results.json` |

---

## 3 · Tier 2 — Four Main Conditions

These fill the primary results table. All four use the same 5-task QA dataset with 70/15/15 splits and must be evaluated with `cache_size=1024` and greedy decoding for a fair comparison.

---

### M1 · LoRA only — SFT fine-tuning, no NAMM

Supervised fine-tuning (SFT) with LoRA on the 5-task QA subset with full KV cache during training. Uses chat-template formatted prompts with answer-only loss. Run three times across the rank sweep. **r=8 is the main table entry**; r=4 and r=16 feed ablation A1.

#### M1-r4

```bash
python scripts/run_lora.py \
    --config scripts/configs/lora_rh_m1_instruct_5t.yaml \
    --run_name m1_r4 \
    --lora_rank 4 \
    --lora_alpha 8
```

#### M1-r8 (main table)

```bash
python scripts/run_lora.py \
    --config scripts/configs/lora_rh_m1_instruct_5t.yaml \
    --run_name m1_r8
```

#### M1-r16

```bash
python scripts/run_lora.py \
    --config scripts/configs/lora_rh_m1_instruct_5t.yaml \
    --run_name m1_r16 \
    --lora_rank 16 \
    --lora_alpha 32
```

| Parameter | Value |
|-----------|-------|
| Config | `scripts/configs/lora_rh_m1_instruct_5t.yaml` |
| `method` | `rh_m1_lora_instruct_5t` |
| `sft_mode` | true — supervised fine-tuning with chat template |
| `learning_rate` | 5e-5 |
| `num_epochs` | 150 |
| `batch_size` | 4 |
| `gradient_accumulation_steps` | 4 (effective batch = 16) |
| `max_seq_len` | 7000 |
| `namm_active` | false — full KV cache during training |
| `eval_interval` | 2 steps |
| `lora_rank` | 8 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.1 |
| `lora_target_modules` | `[q_proj, v_proj]` |
| `warmup_ratio` | 0.03 |
| `weight_decay` | 0.01 |
| `max_grad_norm` | 1.0 |
| Tasks | 5-task QA: qasper, 2wikimqa, qasper_e, hotpotqa_e, 2wikimqa_e |
| Splits | `train_frac=0.7`, `val_frac=0.15`, `split_seed=42` (306 train / 64 val / 69 test) |
| Filtering | `min_conditioning_length=4096`, `max_conditioning_length=6500`, `max_answer_tokens=64` |
| `seed` | 42 |
| Output | `experiments/experiment_N/m1_lora_only/{m1_r4,m1_r8,m1_r16}/` |

---

### M2 · Standalone NAMM — frozen LLM, trained from scratch

Trains the NAMM eviction policy via CMA-ES on a frozen Llama-3.2-1B-Instruct using the 5-task QA subset. LLM weights do not change. **No pretrained checkpoint is used** — training starts from random NAMM parameters. This is the learned-eviction-only baseline and the source checkpoint for M3.

```bash
python scripts/run_namm.py \
    'run@_global_=namm_bam_i1_llama32_1b_5t' \
    wandb_run_name=m2_namm_standalone \
    wandb_group_name=main_conditions \
    seed=1337
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Config | `config/config.yaml` (Hydra) | |
| `run` preset | `namm_bam_i1_llama32_1b_5t` | BAM policy, i1 architecture, 5-task |
| `pop_size` | 8 | From preset |
| `elite_ratio` | 0.5 | Standard CMA-ES setting |
| `init_sigma` | 0.065 | Already validated in prior runs |
| `memory_policy_fixed_delay` | 256 | From preset |
| `min_conditioning_length` | 4096 | Filters out short prompts |
| `max_conditioning_length` | 6500 | Filters which prompts are eligible |
| `max_memory_length` | 1024 | KV cache budget (4× compression ratio) |
| `max_answer_tokens` | 64 | |
| `samples_batch_size` | 8 | Prompts per task per step |
| `batch_size` | 4 | Sequences per GPU forward pass |
| `max_iters` | 200 | Preset default; 200 generations |
| `eval_interval` | 5 | |
| `max_new_tokens` | 256 | Chunk size for generation |
| `save_checkpoint_every` | null (save every iter) | Set to N to reduce I/O |
| Tasks | 5-task QA: qasper, 2wikimqa, qasper_e, hotpotqa_e, 2wikimqa_e | |
| Splits | `train_frac=0.7`, `val_frac=0.15` (306 train / 64 val / 69 test) | |
| `seed` | 1337 | |
| Output | `outputs/{date}/{time}/` (Hydra default) | |

> If training curves are still improving at generation 200, extend to 300+, but do not plan for it upfront.

> **Threshold-only variant:** append `threshold_only=true scoring_initializer=2` to run M2 with the original NAMM paper's eviction rule (score threshold only, no hard top-k cap). The cache size will vary dynamically per step.
> ```bash
> python scripts/run_namm.py \
>     'run@_global_=namm_bam_i1_llama32_1b_5t' \
>     threshold_only=true \
>     scoring_initializer=2 \
>     wandb_run_name=m2_namm_threshold
> ```
> `scoring_initializer=2` is required: with the default value of 0 the CMA-ES mean starts at the eviction boundary (score=0) and collapses to all-evict immediately. Starting at 2 places every token above threshold so CMA-ES can learn selective eviction.

---

### M3 · LoRA with frozen NAMM at train-time

LoRA is fine-tuned (SFT) while the frozen M2 NAMM is active during training. This is the intermediate condition between M1 (no NAMM) and M4 (joint). Isolates whether NAMM eviction during gradient steps helps independently of co-optimisation. Requires M2 to be complete.

```bash
python scripts/run_lora.py \
    --config scripts/configs/lora_rh_m4_instruct_5t.yaml \
    --run_name m3_lora_frozen_namm \
    --namm_checkpoint <path-to-m2-checkpoint>
```

| Parameter | Value |
|-----------|-------|
| Config | `scripts/configs/lora_rh_m4_instruct_5t.yaml` |
| `method` | `rh_m4_frozen_5t` |
| `sft_mode` | true — supervised fine-tuning with chat template |
| `namm_active` | true + NAMM checkpoint (required) |
| `learning_rate` | 1e-4 |
| `num_epochs` | 150 |
| `batch_size` | 1 |
| `gradient_accumulation_steps` | 16 (effective batch = 16) |
| `max_seq_len` | 7000 |
| `lora_rank` | 8 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.05 |
| `lora_target_modules` | `[q_proj, v_proj]` |
| `eval_interval` | 2 steps |
| `cache_size` | 1024 |
| Tasks | 5-task QA: qasper, 2wikimqa, qasper_e, hotpotqa_e, 2wikimqa_e |
| Splits | `train_frac=0.7`, `val_frac=0.15`, `split_seed=42` (306 train / 64 val / 69 test) |
| Filtering | `min_conditioning_length=4096`, `max_conditioning_length=6500`, `max_answer_tokens=64` |
| `seed` | 42 |
| What it answers | Does training LoRA under compressed context help, even without co-optimising the NAMM? |

---

### M4-LoRA · Joint — simultaneous alternating NAMM + LoRA (primary gradient contribution)

NAMM and LoRA are co-trained in alternating stages: Stage A evolves the NAMM for N CMA-ES iterations; Stage B gradient-updates the LoRA for E epochs. Repeated for K outer loops. **Cold-start only** — no pretrained NAMM checkpoint. Must use the same 5-task dataset, splits, and filtering as M1–M3.

```bash
python scripts/run_joint.py \
    --config scripts/configs/joint_default.yaml \
    --run_name m4_joint_lora \
    --adapter_type lora \
    --num_outer_loops 2 \
    --namm_iterations_per_stage 100 \
    --lora_epochs_per_stage 75 \
    --lora_rank 8 \
    --cache_size 1024
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Config | `scripts/configs/joint_default.yaml` | |
| `adapter_type` | `lora` | Overrides config default of `es` |
| `num_outer_loops` | 2 | |
| `namm_iterations_per_stage` | 100 | 2 × 100 = 200 total — matches M2 |
| `lora_epochs_per_stage` | 75 | 2 × 75 = 150 total epochs — matches M1 |
| `lora_rank` | 8 | Matches M1-r8 |
| `lora_alpha` | 16 | Matches M1 |
| `learning_rate` | 5e-5 | Matches M1 |
| `sft_mode` | true | Matches M1 |
| `cache_size` | 1024 | NAMM evicts during training |
| `namm_checkpoint` | null — cold-start | No pretrained checkpoint |
| `eval_after_each_loop` | true | Per-loop F1 saved in `results.json` |
| Tasks | 5-task QA: qasper, 2wikimqa, qasper_e, hotpotqa_e, 2wikimqa_e | |
| Splits | `train_frac=0.7`, `val_frac=0.15`, `split_seed=42` | |
| Filtering | `min_conditioning_length=4096`, `max_conditioning_length=6500`, `max_answer_tokens=64` | |
| Output | `experiments/experiment_N/joint_lora/m4_joint_lora/` | |

> The default `joint_default.yaml` may need overrides to match the M1/M2 hyperparameters above. Ensure the LoRA stages use SFT mode with the same learning rate and batch size as M1.

---

## 4 · Tier 3 — Ablations

These support the discussion sections and answer the research questions. Most require Tier 2 to be complete first.

---

### A1 · LoRA rank sweep (r = 4, 8, 16)

No additional runs needed — results come from M1-r4, M1-r8, and M1-r16. Report all three F1 scores in a table in the ablation section to justify the r=8 choice for all other conditions.

| | |
|---|---|
| Source runs | M1-r4, M1-r8, M1-r16 |
| Metric | 5-task avg F1 on test split |
| What it answers | Effect of adapter expressivity; justifies rank selection |

---

### A4 · NAMM disabled at eval on M4 checkpoint (modularity test)

Takes the fully trained M4 checkpoint and evaluates it twice: once with NAMM active, once with NAMM disabled (full KV cache). The F1 delta isolates how much of M4's performance comes from the LoRA alone versus the LoRA+NAMM system.

#### M4 — NAMM active (standard)

```bash
python scripts/run_eval.py \
    --es_checkpoint experiments/experiment_N/joint_lora/m4_joint_lora/adapter/stage_1/ \
    --namm_checkpoint experiments/experiment_N/joint_lora/m4_joint_lora/namm/latest.pt \
    --cache_size 1024 \
    --output_dir experiments/ablations/a4_modularity/m4_namm_on
```

#### M4 LoRA only — NAMM disabled

```bash
python scripts/run_eval.py \
    --es_checkpoint experiments/experiment_N/joint_lora/m4_joint_lora/adapter/stage_1/ \
    --output_dir experiments/ablations/a4_modularity/m4_namm_off
# No --namm_checkpoint → full KV cache, no eviction
```

| | |
|---|---|
| Adapter checkpoint | `joint_lora/m4_joint_lora/adapter/stage_1/` — stages are 0-indexed; final stage with `--num_outer_loops 2` is `stage_1` |
| NAMM checkpoint | `joint_lora/m4_joint_lora/namm/latest.pt` — written after each NAMM stage; always reflects the most recent stage |
| What it answers | RQ4: are jointly-trained NAMM and LoRA co-dependent, or is the LoRA sufficient alone? |
| Expected finding | M4-NAMM-on > M4-NAMM-off ≈ M1 (LoRA adapts to compressed context; removing NAMM at eval hurts) |

---

## 5 · Results Collection

### Generating the cross-experiment report

```bash
python scripts/generate_report.py \
    --experiment_dir experiments/experiment_N/ \
    --output experiments/experiment_N/paper_results.csv
```

### Expected `results.json` schema

```json
{
  "f1": 0.XXX,
  "exact_match": 0.XXX,
  "num_samples": 69,
  "cache_size": 1024,
  "method": "rh_m1_lora_instruct_5t"
}
```

Joint runs produce a list of entries (one per outer loop), giving per-loop learning curves for free.

### Main results table (FAIR-01 gradient conditions)

| Condition | Run name | F1 (test) | Cache |
|-----------|----------|-----------|-------|
| B0  Base model, full cache | `baseline_eval` | — | full |
| B1  Base model + recency | `b1_recency` | — | 1024 |
| M1-LoRA  LoRA only (r=8) | `m1_r8` | — | 1024 |
| M2  Standalone NAMM | `m2_namm_standalone` | — | 1024 |
| M3-LoRA  LoRA + frozen NAMM | `m3_lora_frozen_namm` | — | 1024 |
| M4-LoRA  Joint LoRA + NAMM | `m4_joint_lora` | — | 1024 |

---

## 6 · Completed Runs & Checkpoints

All runs below use the 5-task LongBench QA subset (qasper, 2wikimqa, qasper_e, hotpotqa_e, 2wikimqa_e) on LLaMA 3.2-1B-Instruct. WandB entity: `SNLP_NAMM`, project: `memory_evolution_hf`.

### M2 — Standalone NAMM (CMA-ES, no LoRA)

These are pure NAMM eviction policy training runs. The long auto-generated names encode hyperparameters: `p8` = pop_size 8, `8qs` = samples_batch_size 8, `256fixDel` = fixed_delay 256.

| Cache | WandB run name | WandB ID | State | Iters | GCS checkpoint |
|-------|----------------|----------|-------|-------|----------------|
| 1024 | `rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs1024` | `lenhmfb1` | finished | 200 | `gs://statistical-nlp/NAMM_checkpoints/pretrained/namm-5t-cs1024-llama32-1b/` |
| 2048 | `rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs2048` | `y5fdw0f9` → `ccflnsds` | finished | 200 | `gs://statistical-nlp/NAMM_checkpoints/pretrained/namm-5t-cs2048-llama32-1b/` |
| 3072 | `rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs3072` | `quc95irz` | finished | 200 | **not uploaded** |

The cs2048 run was split across two wandb segments: `y5fdw0f9` (iters 0–30, failed on `scaup-l`) was resumed with `scratch=false` as `ccflnsds` (iters 30–200, on `mandarin-l`). Earlier duplicate runs also exist at these names with different IDs (crashed/failed); the IDs above are the definitive finished runs.

#### M2 best validation metrics

| Cache | Best iter | Val mean F1 | Val tasks_aggregate |
|-------|-----------|-------------|---------------------|
| 1024 | 105 | 27.90 | 0.00279 |
| 2048 | 40 | 27.67 | 0.00277 |
| 3072 | — | — | — |

### M1 — LoRA only (no NAMM, full context)

Single logical training run split across 3 WandB segments due to crashes/resumes. Best checkpoint is from the final segment (`qfoxxi2m`).

| Segment | WandB run name | WandB ID | State | Epochs | Steps | Best val avg F1 |
|---------|----------------|----------|-------|--------|-------|-----------------|
| 1 | `rh_m1_5t_v2` | `kz6vqo2o` | crashed | 5 | 125 | 39.15 |
| 2 | `rh_m1_5t_resume` | `x9a4smmf` | failed | 10 | 250 | 40.60 |
| 3 | `rh_m1_5t_resume` | `qfoxxi2m` | killed | 28 | 684 | **45.48** (step 336) |

**GCS checkpoint:** `gs://statistical-nlp/NAMM_checkpoints/pretrained/lora-m1-5t-llama32-1b/` (best_ckpt.pt at step 336)

### M3 — LoRA with frozen NAMM at train-time

These runs train LoRA adapters while the frozen M2 NAMM actively evicts KV cache tokens. Named `rh_m4_5t_cs*` in WandB (historical naming from the `rh_m4_frozen` method config).

| Cache | WandB run name | WandB ID | State | Epochs | Steps | Best val avg F1 | GCS checkpoint |
|-------|----------------|----------|-------|--------|-------|-----------------|----------------|
| 1024 | `rh_m4_5t_cs1024` | `ovosogkj` | crashed | 25 | 609 | **45.59** (step 340) | `gs://statistical-nlp/NAMM_checkpoints/pretrained/lora-m4-frozen-5t-cs1024-llama32-1b/` |
| 2048 | `rh_m4_5t_cs2048` | `m4knrhmr` | crashed | 15 | 371 | **44.86** (step 244) | `gs://statistical-nlp/NAMM_checkpoints/pretrained/lora-m4-frozen-5t-cs2048-llama32-1b/` |
| 3072 | `rh_m4_5t_cs3072` | `4sgkswa6` | finished | 4 | 117 | 33.37 | **not uploaded** |

The cs1024 and cs2048 runs crashed before completing all 150 epochs but best checkpoints were saved before crash. The cs3072 run finished but at only epoch 4 — may need rerunning with more epochs.

> **Naming note:** WandB names `rh_m4_5t_cs*` refer to M3 (frozen NAMM), not M4 (joint). The `rh_m4` prefix comes from the `rh_m4_frozen` method in the LoRA config. The NAMM checkpoint used by each M3 run is the corresponding M2 run at the same cache size.

> **WARNING — results directory mislabelling:** The `results/main_table_5t/M4/` directory contains M3 (frozen NAMM) eval results, NOT M4 (joint) results. This mislabelling propagated from the `rh_m4_frozen` config name into `scripts/organize_eval_results.py` and all downstream plots/reports. The correct mapping is: `results/main_table_5t/M4/ → experiment spec M3`. Similarly, `A4/` ablations disable the frozen NAMM from M3 checkpoints, not from a joint M4 run. All analysis reports written before this note was added use "M4" to mean M3-frozen-NAMM.

### M4 — Joint LoRA + NAMM

**Not yet started.** No runs matching the M4 joint training protocol exist in WandB.

### B0, B1 — Baselines

**Not yet run** as formal tracked experiments. Baseline F1 values are available from the `lora/baseline_lb_avg_f1` metric logged at the start of LoRA runs, but dedicated eval runs with `results.json` output have not been created.

### A1 — LoRA rank sweep

**Not yet run.** Only r=8 has been trained (M1). r=4 and r=16 still needed.

### A4 — NAMM disabled at eval

**Not yet run.** Requires M4 to be complete first.

### Completion summary

| Step | Experiment | Status |
|------|------------|--------|
| 1 | B0 — baseline eval | **not started** |
| 2 | B1 — recency eviction eval | **not started** |
| 3 | M1-LoRA — rank sweep | **partial** (r=8 done; r=4, r=16 not started) |
| 4 | M2 — standalone NAMM | **done** (cs1024, cs2048, cs3072 all finished; cs3072 GCS upload pending) |
| 5 | M3-LoRA — LoRA + frozen NAMM | **done** (cs1024, cs2048 best ckpts saved; cs3072 may need more epochs) |
| 6 | M4-LoRA — joint LoRA + NAMM | **not started** |
| 7 | A4 — NAMM disabled at eval | **not started** (blocked on M4) |

---

## 7 · Smoke Tests

Run these before committing to any full experiment to confirm the pipeline works end-to-end.

#### M1 smoke test

```bash
python scripts/run_lora.py \
    --config scripts/configs/lora_rh_m1_instruct_5t.yaml \
    --run_name smoke_m1 \
    --num_epochs 1 \
    --eval_interval 5 \
    --no-gcs
```

#### M4-LoRA joint smoke test

```bash
python scripts/run_joint.py \
    --run_name smoke_joint_lora \
    --adapter_type lora \
    --num_outer_loops 2 \
    --namm_iterations_per_stage 3 \
    --lora_epochs_per_stage 1 \
    --population_size 2 \
    --mini_batch_size 2
```

#### Eval smoke test

```bash
python scripts/run_eval.py \
    --run_config full_cache_baseline_llama32_1b \
    --num_samples 10
```
