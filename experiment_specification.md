# evo-memory — Experiment Specification

**Model:** Llama-3.2-1B-Instruct  
**Benchmark:** Qasper (token-level F1)  
**Hardware:** Single NVIDIA GPU  
**No pretrained NAMM checkpoint used** — all NAMM training is from scratch on Qasper.

---

## 0 · Fairness Constraints (FAIR-01)

All conditions in the main results table must satisfy every constraint below. Any violation invalidates the comparison.

- **Compute equivalence** — all methods see the same number of training tokens from Qasper
- **Data equivalence** — identical splits across all runs (`train_split=0.8`, `val_split=0.1`, `split_seed=42`)
- **Model equivalence** — all conditions start from the same base Llama-3.2-1B-Instruct weights, no pretrained adapters
- **Memory equivalence** — all conditions evaluated with `cache_size=1024` at test time
- **Decoding equivalence** — greedy decoding (`temperature=0.0`) for all final evaluations

> **Compute anchor (gradient conditions):** M1 trains for 2 epochs with `gradient_accumulation_steps=16`. M4-LoRA joint must match this: use `--num_outer_loops 2 --lora_epochs_per_stage 1`. M2 standalone NAMM uses 300 CMA-ES generations; M3 must also use 300.
>
> **ES compute note:** ES and gradient-based conditions are **not compute-equivalent** and must not be directly compared in the main FAIR-01 table. ES is gradient-free but requires many forward passes per weight update (population_size=8 × mini_batch_size=16 = 128 evaluations per iteration; 300 iterations ≈ 38,400 forward passes, versus ~1,600 gradient steps for M1-LoRA). ES variants are reported in a separate section with their own comparisons. Data and model equivalence constraints (splits, seeds, base weights) still apply to all ES conditions.

---

## 1 · Recommended Execution Order

| Step | Experiment | Dependency |
|------|------------|------------|
| 1 | B0 — baseline eval | none |
| 2 | B1 — recency eviction eval | none |
| 3 | M1-LoRA — rank sweep (r=4, 8, 16) | none |
| 4 | M1-ES — ES-only fine-tuning | none |
| 5 | M2 — standalone NAMM training | none |
| 6 | M3-LoRA — sequential: M1-r8 → NAMM | M1-r8 checkpoint |
| 7 | M3-ES — sequential: M1-ES → NAMM | M1-ES checkpoint |
| 8 | M4-LoRA — joint LoRA + NAMM | none |
| 9 | M4-ES — joint ES + NAMM | none |
| 10 | A2 — cache size sweep | M2 checkpoint |
| 11 | A4 — NAMM disabled at eval | M4-LoRA checkpoint |
| 12 | A5-LoRA — LoRA with frozen NAMM at train-time | M2 checkpoint |
| 13 | A5-ES — ES with frozen NAMM at train-time | M2 checkpoint |

---

## 2 · Tier 1 — Baselines

These require no training. Run first to establish reference scores.

---

### B0 · Base model, full KV cache

Evaluates raw Llama-3.2-1B-Instruct on Qasper with no KV cache limit, no eviction, no fine-tuning. This is the absolute performance floor.

```bash
python scripts/run_eval.py \
    --run_config full_cache_baseline_llama32_1b \
    --train_split 0.8 \
    --output_dir experiments/experiment_N/baseline
```

| Parameter | Value |
|-----------|-------|
| `es_checkpoint` | null — evaluates base model weights |
| `namm_checkpoint` | null — no eviction, full KV cache |
| `run_config` | `full_cache_baseline_llama32_1b` — no-eviction policy |
| `cache_size` | null — no limit |
| Output | `experiments/experiment_N/baseline/results.json` |

---

### B1 · Base model, recency eviction

Evaluates the base model with a fixed recency policy: keeps the most recently written tokens, evicts oldest first. No learned policy, no training. Establishes what a naive heuristic achieves at `cache_size=1024`.

```bash
python scripts/run_eval.py \
    --run_config recency_baseline_llama32_1b \
    --cache_size 1024 \
    --train_split 0.8 \
    --output_dir experiments/experiment_N/es_recency/b1_recency \
    --override "task@_global_=qasper"
```

| Parameter | Value |
|-----------|-------|
| `run_config` | `recency_baseline_llama32_1b` — matches `config/run/recency_baseline_llama32_1b.yaml` |
| `cache_size` | 1024 |
| task override | `task@_global_=qasper` — recency config defaults to `lb_3subset_eval`; override to match all other conditions |
| Output | `experiments/experiment_N/es_recency/b1_recency/results.json` |

---

## 3 · Tier 2 — Four Main Conditions

These fill the primary results table. All four must be evaluated with `cache_size=1024` and greedy decoding for a fair comparison.

---

### M1 · LoRA only — gradient fine-tuning, no NAMM

Standard gradient-based LoRA fine-tuning on Qasper with full KV cache during training. Run three times across the rank sweep. **r=8 is the main table entry**; r=4 and r=16 feed ablation A1.

#### M1-r4

```bash
python scripts/run_lora.py \
    --config scripts/lora_m1_only.yaml \
    --run_name m1_r4 \
    --lora_rank 4 \
    --lora_alpha 8
```

#### M1-r8 (main table)

```bash
python scripts/run_lora.py \
    --config scripts/lora_m1_only.yaml \
    --run_name m1_r8
# lora_rank=8, lora_alpha=null (defaults to rank=8) — from config
```

#### M1-r16

```bash
python scripts/run_lora.py \
    --config scripts/lora_m1_only.yaml \
    --run_name m1_r16 \
    --lora_rank 16 \
    --lora_alpha 32
```

| Parameter | Value |
|-----------|-------|
| Config | `scripts/lora_m1_only.yaml` |
| `method` | `m1_lora_only` |
| `learning_rate` | 2e-4 |
| `num_epochs` | 2 — FAIR-01 anchor |
| `gradient_accumulation_steps` | 16 |
| `batch_size` | 1 |
| `max_seq_len` | 3500 |
| `namm_active` | false — full KV cache during training |
| `eval_interval` | 40 steps |
| `lora_target_modules` | `[q_proj, v_proj]` |
| Output | `experiments/experiment_N/m1_lora_only/{m1_r4,m1_r8,m1_r16}/results.json` |

> `lora_alpha=null` defaults to `alpha=rank`. Override explicitly when sweeping ranks: r=4 → alpha=8, r=8 → alpha=null, r=16 → alpha=32.

---

### M1-ES · ES only — evolutionary fine-tuning, no NAMM

ES fine-tuning of all base LLM weights on Qasper with full KV cache during training. No LoRA adapters — ES perturbs the full weight space. **Not compute-equivalent to M1-LoRA** (see FAIR-01 ES note); reported in the ES variants section of the results.

```bash
python scripts/run_es.py \
    --config scripts/es_m1_only.yaml \
    --run_name m1_es
```

| Parameter | Value |
|-----------|-------|
| Config | `scripts/es_m1_only.yaml` |
| `method` | `es_only` — auto-detected from null `namm_checkpoint` |
| `run_config` | `full_cache_es_llama32_1b` — no eviction, Qasper task |
| `sigma` | 0.001 |
| `alpha` | 0.0005 |
| `population_size` | 8 |
| `num_iterations` | 300 — matches M2 NAMM generation count |
| `mini_batch_size` | 16 |
| `train_split` | 0.8, `split_seed=42` |
| `namm_checkpoint` | null — full KV cache, no eviction |
| Output | `experiments/experiment_N/es_only/m1_es/` |

> Eval at end: pass the final ES checkpoint to `run_eval.py` with `--run_config full_cache_baseline_llama32_1b` (no NAMM) or `--namm_checkpoint <m2>` (with NAMM) to compare against LoRA conditions.

---

### M2 · Standalone NAMM — frozen LLM, trained from scratch on Qasper

Trains the NAMM eviction policy via CMA-ES on a frozen Llama-3.2-1B-Instruct. LLM weights do not change. **No pretrained checkpoint is used** — training starts from random NAMM parameters. This is the learned-eviction-only baseline and the source checkpoint for M3 and A5.

```bash
python scripts/run_namm.py \
    run=namm_bam_i1_llama32_1b \
    wandb_run_name=m2_namm_standalone \
    wandb_group_name=main_conditions \
    seed=42 \
    trainer_config.max_iters=299
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Config | `config/config.yaml` (Hydra) | |
| `run` preset | `namm_bam_i1_llama32_1b` | BAM policy, i1 architecture |
| `pop_size` | 8 | From preset |
| `elite_ratio` | 0.5 | Standard CMA-ES setting |
| `init_sigma` | 0.065 | Already validated in prior runs |
| `memory_policy_fixed_delay` | 256 | From preset |
| `max_conditioning_length` | 4086 | ~4k token context |
| `max_memory_length` | 1024 | 4× compression ratio |
| `mini_batch_size` | 16 | From preset |
| `trainer_config.max_iters` | 299 | MemoryTrainer loops `range(0, max_iters+1)` → 300 generations; preset default is 200 |
| Generations | 300 | ~13.5 hours on a single GPU |
| Task | Qasper only | Single-task; no incremental evolution needed |
| Output | `outputs/{date}/{time}/` (Hydra default) | |

> 300 flat generations on a single task is sufficient. Sakana used incremental evolution across three tasks to handle multi-task LongBench complexity — that does not apply here. If training curves are still improving at generation 300, extend to 400, but do not plan for it upfront.

---

### M3 · Sequential — LoRA fine-tuning, then NAMM on top

Two-stage sequential pipeline. Stage 1 fine-tunes with LoRA (identical to M1-r8 — reuse that checkpoint if already complete). Stage 2 trains the NAMM policy on the frozen LoRA-adapted model.

#### Stage 1 — LoRA (reuse M1-r8 if already run)

```bash
python scripts/run_lora.py \
    --config scripts/lora_m1_only.yaml \
    --run_name m3_stage1_lora
# Identical to M1-r8 — skip if checkpoint already exists
```

#### Stage 2 — NAMM on LoRA-adapted model

```bash
python scripts/run_namm.py \
    run=namm_bam_i1_llama32_1b \
    wandb_run_name=m3_namm_on_lora \
    wandb_group_name=main_conditions \
    seed=42 \
    trainer_config.max_iters=299 \
    adapter_path=experiments/experiment_N/m1_lora_only/m1_r8/checkpoints/
```

| Parameter | Value |
|-----------|-------|
| Stage 1 config | `scripts/lora_m1_only.yaml` — rank=8, 2 epochs |
| Stage 2 config | `config/config.yaml` (Hydra) — 300 CMA-ES generations |
| Stage 2 init | LoRA-adapted weights from Stage 1 merged into base LLM, then fresh NAMM parameters trained on top |
| `adapter_path` | Path to Stage 1 LoRA checkpoint dir — handled in `namm/run_utils.py:make_eval_model` |
| `max_memory_length` | 1024 |
| Output | `outputs/{date}/{time}/` (Hydra) |

---

### M3-ES · Sequential — ES fine-tuning, then NAMM on top

ES analogue of M3. Stage 1 fine-tunes all base LLM weights with ES (reuse M1-ES checkpoint if available). Stage 2 trains the NAMM policy on the frozen ES-adapted model. ES weights are loaded into the model via the `es_checkpoint_path` Hydra override.

#### Stage 1 — ES (reuse M1-ES if already run)

```bash
python scripts/run_es.py \
    --config scripts/es_m1_only.yaml \
    --run_name m3_es_stage1
# Identical to M1-ES — skip if checkpoint already exists
```

#### Stage 2 — NAMM on ES-adapted model

```bash
python scripts/run_namm.py \
    run=namm_bam_i1_llama32_1b \
    wandb_run_name=m3_namm_on_es \
    wandb_group_name=es_conditions \
    seed=42 \
    trainer_config.max_iters=299 \
    es_checkpoint_path=experiments/experiment_N/es_only/m1_es/checkpoints/es_checkpoint_final.pt
```

| Parameter | Value |
|-----------|-------|
| Stage 1 config | `scripts/es_m1_only.yaml` — 300 ES iterations, full KV cache |
| Stage 2 config | `config/config.yaml` (Hydra) — 300 CMA-ES generations |
| Stage 2 init | ES-fine-tuned weights applied to base LLM, then fresh NAMM parameters trained on top |
| `es_checkpoint_path` | Path to final ES checkpoint — loaded in `namm/run_utils.py:make_eval_model` |
| `max_memory_length` | 1024 |
| Output | `outputs/{date}/{time}/` (Hydra) |

---

### M4-LoRA · Joint — simultaneous alternating NAMM + LoRA (primary gradient contribution)

NAMM and LoRA are co-trained in alternating stages: Stage A evolves the NAMM for N CMA-ES iterations; Stage B gradient-updates the LoRA for E epochs. Repeated for K outer loops. **Cold-start only** — no pretrained NAMM checkpoint.

```bash
python scripts/run_joint.py \
    --config scripts/joint_default.yaml \
    --run_name m4_joint_lora \
    --adapter_type lora \
    --num_outer_loops 2 \
    --namm_iterations_per_stage 150 \
    --lora_epochs_per_stage 1 \
    --lora_rank 8 \
    --cache_size 1024
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Config | `scripts/joint_default.yaml` | |
| `adapter_type` | `lora` | Overrides config default of `es` |
| `num_outer_loops` | 2 | |
| `namm_iterations_per_stage` | 150 | 2 × 150 = 300 total — matches M2 (FAIR-01) |
| `lora_epochs_per_stage` | 1 | 2 × 1 = 2 total epochs — matches M1 (FAIR-01) |
| `lora_rank` | 8 | Matches M1-r8 |
| `lora_alpha` | null → defaults to rank | Override to 16 if explicit control is needed |
| `learning_rate` | 2e-4 | From `joint_default.yaml` |
| `cache_size` | 1024 | NAMM evicts during training |
| `namm_checkpoint` | null — cold-start | No pretrained checkpoint |
| `eval_after_each_loop` | true | Per-loop F1 saved in `results.json` |
| Output | `experiments/experiment_N/joint_lora/m4_joint_lora/` | |

> The default `joint_default.yaml` has `num_outer_loops=5` and `namm_iterations_per_stage=50` (250 total NAMM iters, 5 LoRA epochs). **Override both** as shown above to satisfy FAIR-01.

---

### M4-ES · Joint — simultaneous alternating NAMM + ES (primary ES contribution)

ES analogue of M4-LoRA. Stage A trains NAMM via CMA-ES; Stage B perturbs all base LLM weights via ES (no LoRA). Alternates for K outer loops. `run_joint.py` already supports this via `--adapter_type es`.

```bash
python scripts/run_joint.py \
    --config scripts/joint_default.yaml \
    --run_name m4_joint_es \
    --adapter_type es \
    --num_outer_loops 2 \
    --namm_iterations_per_stage 150 \
    --adapter_iterations_per_stage 150 \
    --population_size 8 \
    --mini_batch_size 16 \
    --cache_size 1024
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| `adapter_type` | `es` | ES weight perturbation instead of LoRA gradient steps |
| `num_outer_loops` | 2 | |
| `namm_iterations_per_stage` | 150 | 2 × 150 = 300 total NAMM iters — matches M2 |
| `adapter_iterations_per_stage` | 150 | 2 × 150 = 300 total ES iters — matches M1-ES |
| `population_size` | 8 | |
| `mini_batch_size` | 16 | |
| `cache_size` | 1024 | NAMM evicts during training |
| `namm_checkpoint` | null — cold-start | |
| `eval_after_each_loop` | true | |
| Output | `experiments/experiment_N/joint_es/m4_joint_es/` | |

> A4 modularity test for M4-ES: pass `experiments/experiment_N/joint_es/m4_joint_es/adapter/stage_1/checkpoints/es_checkpoint_final.pt` as `--es_checkpoint` and `namm/latest.pt` as `--namm_checkpoint` to `run_eval.py`.

---

## 4 · Tier 3 — Ablations

These support the discussion sections and answer the research questions. Most require Tier 2 to be complete first.

---

### A1 · LoRA rank sweep (r = 4, 8, 16)

No additional runs needed — results come from M1-r4, M1-r8, and M1-r16. Report all three F1 scores in a table in the ablation section to justify the r=8 choice for all other conditions.

| | |
|---|---|
| Source runs | M1-r4, M1-r8, M1-r16 |
| Metric | Qasper token-level F1 on test split |
| What it answers | Effect of adapter expressivity; justifies rank selection |

---

### A2 · Cache size sweep (512, 1024, 2048 tokens)

Eval-only using the M2 checkpoint at three cache sizes. Plots the accuracy vs. memory Pareto curve. The 1024 result comes directly from M2 training — only 512 and 2048 require new eval runs.

#### M2 at cache_size=512

```bash
python scripts/run_eval.py \
    --namm_checkpoint <path-to-m2-checkpoint> \
    --cache_size 512 \
    --train_split 0.8 \
    --output_dir experiments/ablations/a2_cache/m2_cache512
```

#### M2 at cache_size=2048

```bash
python scripts/run_eval.py \
    --namm_checkpoint <path-to-m2-checkpoint> \
    --cache_size 2048 \
    --train_split 0.8 \
    --output_dir experiments/ablations/a2_cache/m2_cache2048
```

| | |
|---|---|
| Checkpoint | M2 final NAMM checkpoint from `outputs/{date}/{time}/` |
| Cache sizes tested | 512, 1024 (from training), 2048 |
| What it answers | Memory-accuracy tradeoff; replicates NAMM paper Fig. 5 at 1B scale |

---

### A4 · NAMM disabled at eval on M4 checkpoint (modularity test)

Takes the fully trained M4 checkpoint and evaluates it twice: once with NAMM active, once with NAMM disabled (full KV cache). The F1 delta isolates how much of M4's performance comes from the LoRA alone versus the LoRA+NAMM system.

#### M4 — NAMM active (standard)

```bash
python scripts/run_eval.py \
    --es_checkpoint experiments/experiment_N/joint_lora/m4_joint_lora/adapter/stage_1/ \
    --namm_checkpoint experiments/experiment_N/joint_lora/m4_joint_lora/namm/latest.pt \
    --cache_size 1024 \
    --train_split 0.8 \
    --output_dir experiments/ablations/a4_modularity/m4_namm_on
```

#### M4 LoRA only — NAMM disabled

```bash
python scripts/run_eval.py \
    --es_checkpoint experiments/experiment_N/joint_lora/m4_joint_lora/adapter/stage_1/ \
    --train_split 0.8 \
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

### A5 · LoRA with frozen NAMM at train-time

LoRA is fine-tuned while the frozen M2 NAMM is active during training. This is the intermediate condition between M1 (no NAMM) and M4 (joint). Isolates whether NAMM eviction during gradient steps helps independently of co-optimisation. Requires M2 to be complete.

```bash
python scripts/run_lora.py \
    --config scripts/lora_rh_m4_instruct.yaml \
    --run_name a5_lora_frozen_namm \
    --namm_checkpoint <path-to-m2-checkpoint>
```

| Parameter | Value |
|-----------|-------|
| Config | `scripts/lora_rh_m4_instruct.yaml` |
| `method` | `rh_m4_frozen` |
| `namm_active` | true + NAMM checkpoint (required) |
| `learning_rate` | 1e-4 (lower than M1 for NAMM stability — from config) |
| `lora_alpha` | 16, `dropout=0.05` |
| `cache_size` | 1024 |
| Training tasks | qasper + multifieldqa_en + hotpotqa + 2wikimqa (from config override) |
| `max_seq_len` | 6600 |
| What it answers | Does training LoRA under compressed context help, even without co-optimising the NAMM? |

> **Confound warning:** A5-LoRA uses multi-task training (4 tasks, `max_seq_len=6600`) while M1-LoRA uses single-task Qasper (`max_seq_len=3500`). Either restrict A5-LoRA to Qasper-only for a clean ablation, or report it as a separate multi-task result. Evaluate on the Qasper test set regardless so scores are comparable.

---

### A5-ES · ES with frozen NAMM at train-time

ES analogue of A5-LoRA. All base LLM weights are optimised via ES while the frozen M2 NAMM evicts tokens at `cache_size=1024` during fitness evaluation. Isolates whether training under compressed context helps ES independently of co-optimisation.

```bash
python scripts/run_es.py \
    --config scripts/es_m1_only.yaml \
    --run_name a5_es_frozen_namm \
    --namm_checkpoint <path-to-m2-checkpoint> \
    --run_config namm_bam_i1_llama32_1b \
    --cache_size 1024 \
    --method es_namm
```

| Parameter | Value |
|-----------|-------|
| Config | `scripts/es_m1_only.yaml` (overrides `run_config` and `cache_size` via CLI) |
| `method` | `es_namm` — explicit; prevents auto-detection confusion |
| `run_config` | `namm_bam_i1_llama32_1b` — loads NAMM architecture for eviction |
| `namm_checkpoint` | M2 final checkpoint — NAMM is frozen, only LLM weights evolve |
| `cache_size` | 1024 |
| `num_iterations` | 300 — from config |
| `train_split` | 0.8, `split_seed=42` — from config |
| Task | Qasper only (from `namm_bam_i1_llama32_1b`) |
| Output | `experiments/experiment_N/es_namm/a5_es_frozen_namm/` |
| What it answers | Does ES fine-tuning under compressed context help, even without co-optimising the NAMM? |

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
  "num_samples": 200,
  "cache_size": 1024,
  "method": "m1_lora_only"
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
| M3-LoRA  Sequential (LoRA → NAMM) | `m3_namm_on_lora` | — | 1024 |
| M4-LoRA  Joint LoRA + NAMM | `m4_joint_lora` | — | 1024 |

### ES variants table (reported separately — not FAIR-01 comparable to gradient table)

| Condition | Run name | F1 (test) | Cache |
|-----------|----------|-----------|-------|
| B0  Base model, full cache | `baseline_eval` | — | full |
| B1  Base model + recency | `b1_recency` | — | 1024 |
| M1-ES  ES only | `m1_es` | — | full |
| M1-ES + M2 NAMM  ES weights + frozen NAMM eval | `m1_es` (eval w/ NAMM) | — | 1024 |
| M2  Standalone NAMM | `m2_namm_standalone` | — | 1024 |
| M3-ES  Sequential (ES → NAMM) | `m3_namm_on_es` | — | 1024 |
| M4-ES  Joint ES + NAMM | `m4_joint_es` | — | 1024 |

---

## 6 · Smoke Tests

Run these before committing to any full experiment to confirm the pipeline works end-to-end.

#### M1 smoke test

```bash
python scripts/run_lora.py \
    --config scripts/lora_m1_only.yaml \
    --run_name smoke_m1 \
    --num_epochs 1 \
    --eval_interval 5 \
    --no-gcs
```

#### M1-ES smoke test

```bash
python scripts/run_es.py \
    --config scripts/es_m1_only.yaml \
    --run_name smoke_es \
    --num_iterations 2 \
    --population_size 2 \
    --mini_batch_size 2 \
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

#### M4-ES joint smoke test

```bash
python scripts/run_joint.py \
    --run_name smoke_joint_es \
    --adapter_type es \
    --num_outer_loops 2 \
    --namm_iterations_per_stage 3 \
    --adapter_iterations_per_stage 3 \
    --population_size 2 \
    --mini_batch_size 2
```

#### Eval smoke test

```bash
python scripts/run_eval.py \
    --run_config full_cache_baseline_llama32_1b \
    --num_samples 10
```
