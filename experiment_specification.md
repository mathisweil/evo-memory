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

WandB base: `https://wandb.ai/SNLP_NAMM/memory_evolution_hf/runs/`

| Step | Experiment          | Dependency | Status              | WandB run                    |
| ---- | ------------------- | ---------- | ------------------- | ---------------------------- |
| 1    | B0 — baseline eval  | none       | done                | eval only                    |
| 2    | B1 — recency eval   | none       | done                | eval only                    |
| 3    | M1 — LoRA (r=8)     | none       | done                | [seg1][m1a] [seg2][m1b] [seg3][m1c]|
| 4    | M2 — standalone NAMM| none       | done                | [`lenhmfb1`][m2]             |
| 5    | M3 — LoRA+frozen NAMM| M2 ckpt   | maskfix killed@425  | [`h0bzg6on`][m3]             |
| 6    | M4 — joint LoRA+NAMM| none       | not started         | —                            |
| 7    | A1 — rank sweep     | none       | not started         | —                            |
| 8    | A4 — NAMM off eval  | M3 ckpt    | done (on M3, not M4)| eval only                    |

[m1a]: https://wandb.ai/SNLP_NAMM/memory_evolution_hf/runs/kz6vqo2o
[m1b]: https://wandb.ai/SNLP_NAMM/memory_evolution_hf/runs/x9a4smmf
[m1c]: https://wandb.ai/SNLP_NAMM/memory_evolution_hf/runs/qfoxxi2m
[m2]: https://wandb.ai/SNLP_NAMM/memory_evolution_hf/runs/lenhmfb1
[m3]: https://wandb.ai/SNLP_NAMM/memory_evolution_hf/runs/h0bzg6on

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

| Parameter         | Value                                                        |
| ----------------- | ------------------------------------------------------------ |
| `es_checkpoint`   | null — evaluates base model weights                          |
| `namm_checkpoint` | null — no eviction, full KV cache                            |
| `run_config`      | `full_cache_baseline_llama32_1b` — no-eviction policy        |
| `cache_size`      | null — no limit                                              |
| Tasks             | 5-task QA subset (override task config to `rh_multi_qa_5t`)  |
| Splits            | `train_frac=0.7`, `val_frac=0.15`, `split_seed=42`          |
| Output            | `experiments/experiment_N/baseline/results.json`             |

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

| Parameter    | Value                                                                                  |
| ------------ | -------------------------------------------------------------------------------------- |
| `run_config` | `recency_baseline_llama32_1b` — matches `config/run/recency_baseline_llama32_1b.yaml`  |
| `cache_size` | 1024                                                                                   |
| Tasks        | 5-task QA subset (override task config to `rh_multi_qa_5t`)                            |
| Splits       | `train_frac=0.7`, `val_frac=0.15`, `split_seed=42`                                    |
| Output       | `experiments/experiment_N/es_recency/b1_recency/results.json`                          |

---

## 3 · Tier 2 — Four Main Conditions

These fill the primary results table. All four use the same 5-task QA dataset with 70/15/15 splits and must be evaluated with `cache_size=1024` and greedy decoding for a fair comparison.

---

### M1 · LoRA only — SFT fine-tuning, no NAMM

Supervised fine-tuning (SFT) with LoRA (r=8) on the 5-task QA subset with full KV cache during training. Uses chat-template formatted prompts with answer-only loss. This is the main table entry; r=4 and r=16 are trained separately in ablation A1.

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

| Parameter                     | Value                                                  |
| ----------------------------- | ------------------------------------------------------ |
| Config                        | `scripts/configs/lora_rh_m1_instruct_5t.yaml`          |
| `method`                      | `rh_m1_lora_instruct_5t`                               |
| `sft_mode`                    | true — SFT with chat template                          |
| `learning_rate`               | 5e-5                                                   |
| `num_epochs`                  | 150                                                    |
| `batch_size`                  | 1 (matches M3/M4)                                      |
| `gradient_accumulation_steps` | 16 (effective batch = 16)                              |
| `max_seq_len`                 | 7000                                                   |
| `namm_active`                 | false — full KV cache during training                  |
| `eval_interval`               | 14 steps                                               |
| `batch_size_eval`             | 2                                                      |
| `early_stopping_patience`     | 20 (~14 epochs)                                        |
| `lora_rank`                   | 8                                                      |
| `lora_alpha`                  | 16                                                     |
| `lora_dropout`                | 0.1                                                    |
| `lora_target_modules`         | `[q_proj, v_proj]`                                     |
| `warmup_ratio`                | 0.03                                                   |
| `weight_decay`                | 0.01                                                   |
| `max_grad_norm`               | 1.0                                                    |
| Tasks                         | qasper, 2wikimqa, qasper_e, hotpotqa_e, 2wikimqa_e    |
| Splits                        | 0.7/0.15/0.15, seed=42 (306/64/70)                    |
| Filtering                     | cond 4096–6500 tokens, max answer 64                   |
| `seed`                        | 42                                                     |
| Output                        | `experiments/experiment_N/m1_lora_only/m1_r8/`         |

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

| Parameter                     | Value                                           |
| ----------------------------- | ----------------------------------------------- |
| Config                        | `config/config.yaml` (Hydra)                    |
| `run` preset                  | `namm_bam_i1_llama32_1b_5t`                     |
| `pop_size`                    | 8                                               |
| `elite_ratio`                 | 0.5                                             |
| `init_sigma`                  | 0.065                                           |
| `memory_policy_fixed_delay`   | 256                                             |
| `min_conditioning_length`     | 4096                                            |
| `max_conditioning_length`     | 6500                                            |
| `max_memory_length`           | 1024 (4x compression ratio)                     |
| `max_answer_tokens`           | 64                                              |
| `samples_batch_size`          | 8 (prompts per task per step)                   |
| `batch_size`                  | 4 (sequences per GPU forward pass)              |
| `max_iters`                   | 200                                             |
| `eval_interval`               | 5                                               |
| `max_new_tokens`              | 256 (chunk size for generation)                 |
| `save_checkpoint_every`       | null (save every iter)                          |
| Tasks                         | qasper, 2wikimqa, qasper_e, hotpotqa_e, 2wikimqa_e |
| Splits                        | 0.7/0.15/0.15, seed=42 (306/64/70)             |
| `seed`                        | 1337                                            |
| Output                        | `outputs/{date}/{time}/` (Hydra default)        |

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

| Parameter                     | Value                                           |
| ----------------------------- | ----------------------------------------------- |
| Config                        | `scripts/configs/lora_rh_m4_instruct_5t.yaml`  |
| `method`                      | `rh_m4_frozen_5t`                               |
| `sft_mode`                    | true — SFT with chat template                   |
| `namm_active`                 | true + NAMM checkpoint (required)               |
| `learning_rate`               | 5e-5 (matches M1; was 1e-4 in "M3-tuned", §6)  |
| `num_epochs`                  | 150                                             |
| `batch_size`                  | 1                                               |
| `gradient_accumulation_steps` | 16 (effective batch = 16)                       |
| `max_seq_len`                 | 7000                                            |
| `lora_rank`                   | 8                                               |
| `lora_alpha`                  | 16                                              |
| `lora_dropout`                | 0.1 (matches M1; was 0.05 in "M3-tuned")       |
| `lora_target_modules`         | `[q_proj, v_proj]`                              |
| `eval_interval`               | 14 steps                                        |
| `batch_size_eval`             | 2                                               |
| `early_stopping_patience`     | 20                                              |
| `cache_size`                  | 1024                                            |
| Tasks                         | qasper, 2wikimqa, qasper_e, hotpotqa_e, 2wikimqa_e |
| Splits                        | 0.7/0.15/0.15, seed=42 (306/64/70)             |
| Filtering                     | cond 4096–6500 tokens, max answer 64            |
| `seed`                        | 42                                              |

> **What M3 answers:** Does training LoRA under compressed context help, even without co-optimising the NAMM?

---

### M4-LoRA · Joint — simultaneous alternating NAMM + LoRA (primary gradient contribution)

NAMM and LoRA are co-trained in alternating stages: Stage A evolves the NAMM for N CMA-ES iterations; Stage B gradient-updates the LoRA for E epochs. Repeated for K outer loops. **Cold-start only** — no pretrained NAMM checkpoint. Must use the same 5-task dataset, splits, and filtering as M1–M3.

```bash
python scripts/run_joint.py \
    --config scripts/configs/joint_lora_m4_5t.yaml \
    --run_name m4_joint_lora
```

| Parameter                     | Value                                           |
| ----------------------------- | ----------------------------------------------- |
| Config                        | `scripts/configs/joint_lora_m4_5t.yaml`         |
| `adapter_type`                | `lora`                                          |
| `num_outer_loops`             | 2                                               |
| `namm_iterations_per_stage`   | 100 (2x100=200 total, matches M2)               |
| `lora_epochs_per_stage`       | 75 (2x75=150 total, matches M1)                 |
| `lora_rank`                   | 8 (matches M1)                                  |
| `lora_alpha`                  | 16 (matches M1)                                 |
| `learning_rate`               | 5e-5 (matches M1)                               |
| `lora_dropout`                | 0.1 (matches M1)                                |
| `lora_batch_size`             | 1 (matches M1/M3)                               |
| `gradient_accumulation_steps` | 16 (effective batch = 16)                       |
| `max_seq_len`                 | 7000                                            |
| `lora_eval_interval`          | 14                                              |
| `sft_mode`                    | true                                            |
| `cache_size`                  | 1024 (NAMM evicts during training)              |
| `namm_checkpoint`             | null — cold-start                               |
| `eval_after_each_loop`        | true                                            |
| Tasks                         | qasper, 2wikimqa, qasper_e, hotpotqa_e, 2wikimqa_e |
| Splits                        | 0.7/0.15/0.15, seed=42                          |
| Filtering                     | cond 4096–6500 tokens, max answer 64            |
| Output                        | `experiments/.../joint_lora/m4_joint_lora/`     |

> Do not use `joint_default.yaml` for the M4 paper run — it is an ad-hoc preset.

> M4 asymmetries to report in the paper: each LoRA stage runs a fresh 75-epoch warmup+cosine schedule (not one continuous 150-epoch curve), and `early_stopping_patience` is not applied inside joint LoRA stages (each stage runs to its full epoch budget). See `docs/m4_readiness_review.md`.

---

## 4 · Tier 3 — Ablations

These support the discussion sections and answer the research questions. Most require Tier 2 to be complete first.

---

### A1 · LoRA rank sweep (r = 4, 8, 16)

Train M1 at r=4 and r=16 (r=8 already done as the main M1 run). Report all three F1 scores in a table in the ablation section to justify the r=8 choice for all other conditions.

| Parameter       | Value                                                    |
| --------------- | -------------------------------------------------------- |
| Runs needed     | M1-r4, M1-r16 (M1-r8 = main M1 run)                    |
| Metric          | 5-task avg F1 on test split                              |
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

| Parameter          | Value                                                |
| ------------------ | ---------------------------------------------------- |
| Adapter checkpoint | `.../adapter/stage_1/` (0-indexed; final = `stage_1`)|
| NAMM checkpoint    | `.../namm/latest.pt` (most recent stage)             |
| What it answers    | Are joint NAMM+LoRA co-dependent, or is LoRA enough? |
| Expected finding   | M4-NAMM-on > M4-NAMM-off ≈ M1                       |

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
  "num_samples": 70,
  "cache_size": 1024,
  "method": "rh_m1_lora_instruct_5t"
}
```

Joint runs produce a list of entries (one per outer loop), giving per-loop learning curves for free.

### Main results table (FAIR-01 gradient conditions)

All F1 values are micro averages (prompt-count-weighted). Two eval splits: **test** (70 samples, prompts 4096–6500 tokens) and **extended_test** (224 samples, prompts 6500–8192 tokens). See `results/main_table_5t/all_results.json` for full per-task breakdowns.

| Condition                  | Cache              | Test micro (n=70) | Ext test micro (n=224) |
| -------------------------- | ------------------ | -----------------:| ----------------------:|
| B0  Base model, full cache | full               |             22.41 |                  22.30 |
| B1  Base + recency         | 1024               |             11.33 |                   6.93 |
| B1  Base + recency         | 2048               |             11.10 |                   9.30 |
| M1  LoRA only (r=8)        | full               |             31.14 |                  31.84 |
| M2  Standalone NAMM        | 1024               |             10.83 |                  18.79 |
| M2  Standalone NAMM        | 2048               |             15.27 |                  19.19 |
| M3  LoRA + frozen NAMM     | 1024               |             23.52 |                  25.40 |
| M3  LoRA + frozen NAMM     | 2048               |             31.41 |                  23.40 |
| M4  Joint LoRA + NAMM      | —                  |                 — |                      — |
| Trunc/plain                | 1024               |             18.21 |                  17.83 |
| Trunc/plain                | 2048               |             18.26 |                  19.35 |
| Trunc/lora_m1              | 1024               |             26.90 |                  24.24 |
| Trunc/lora_m1              | 2048               |             28.87 |                  27.67 |
| M1_recency                 | 1024               |      0.00 (BROKE) |           0.00 (BROKE) |
| M1_under_NAMM              | 1024               |             26.97 |                  21.83 |
| M1_under_NAMM              | 2048               |             31.71 |                  24.96 |
| A4  M3 LoRA, NAMM off      | full (cs1024 ckpt) |             28.82 |                  25.62 |
| A4  M3 LoRA, NAMM off      | full (cs2048 ckpt) |             38.98 |                  30.61 |

> **Note on M1 eval cache:** M1 trains with full cache and is also evaluated with full cache (no NAMM, no eviction). The FAIR-01 memory-equivalence constraint applies to conditions that use NAMM at eval time; M1 serves as an upper bound showing what LoRA alone can achieve without any cache compression.

> **Note on A4:** A4 was originally specified to use M4 (joint) checkpoints, but M4 has not been run. The A4 results above use M3 (LoRA + frozen NAMM) checkpoints with NAMM disabled at eval time.

> **Note on M1_under_NAMM:** M1 LoRA (trained without NAMM) evaluated with M2 NAMM eviction active. Tests whether the M1 adapter degrades gracefully under eviction it was not trained for.

---

## 6 · Completed Runs & Checkpoints

All runs below use the 5-task LongBench QA subset (qasper, 2wikimqa, qasper_e, hotpotqa_e, 2wikimqa_e) on LLaMA 3.2-1B-Instruct. WandB entity: `SNLP_NAMM`, project: `memory_evolution_hf`.

### M2 — Standalone NAMM (CMA-ES, no LoRA)

These are pure NAMM eviction policy training runs. The long auto-generated names encode hyperparameters: `p8` = pop_size 8, `8qs` = samples_batch_size 8, `256fixDel` = fixed_delay 256.

Run name pattern: `rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs{N}`

GCS base: `gs://statistical-nlp/NAMM_checkpoints/pretrained/`

| Cache | WandB ID                | State    | Iters | GCS subpath                        |
| ----- | ----------------------- | -------- | ----- | ---------------------------------- |
| 1024  | `lenhmfb1`              | finished | 200   | `namm-5t-cs1024-llama32-1b/`      |
| 2048  | `y5fdw0f9` → `ccflnsds` | finished | 200   | `namm-5t-cs2048-llama32-1b/`      |
| 3072  | `quc95irz`              | finished | 200   | **not uploaded**                   |

The cs2048 run was split across two wandb segments: `y5fdw0f9` (iters 0–30, failed on `scaup-l`) was resumed with `scratch=false` as `ccflnsds` (iters 30–200, on `mandarin-l`). Earlier duplicate runs also exist at these names with different IDs (crashed/failed); the IDs above are the definitive finished runs.

#### M2 best validation metrics

| Cache | Best iter | Val mean F1 | Val tasks_aggregate  |
| ----- | --------- | ----------- | -------------------- |
| 1024  | 105       | 27.90       | 0.00279              |
| 2048  | 40        | 27.67       | 0.00277              |
| 3072  | —         | —           | —                    |

### M1 — LoRA only (no NAMM, full context)

Single logical training run split across 3 WandB segments due to crashes/resumes. Best checkpoint is from the final segment (`qfoxxi2m`).

| Segment | WandB run name    | WandB ID   | State   | Epochs | Steps | Best val avg F1      |
| ------- | ----------------- | ---------- | ------- | ------ | ----- | -------------------- |
| 1       | `rh_m1_5t_v2`     | `kz6vqo2o` | crashed | 5      | 125   | 39.15                |
| 2       | `rh_m1_5t_resume` | `x9a4smmf` | failed  | 10     | 250   | 40.60                |
| 3       | `rh_m1_5t_resume` | `qfoxxi2m` | killed  | 28     | 684   | **45.48** (step 336) |

**GCS checkpoint:** `gs://statistical-nlp/NAMM_checkpoints/pretrained/lora-m1-5t-llama32-1b/` (best_ckpt.pt at step 336)

### M3 — LoRA with frozen NAMM at train-time

These runs train LoRA adapters while the frozen M2 NAMM actively evicts KV cache tokens. Named `rh_m4_5t_cs*` in WandB (historical naming from the `rh_m4_frozen` method config).

> **M3 must be re-run.** The runs in the table below used `learning_rate=1e-4` and `lora_dropout=0.05` — both differ from M1's `learning_rate=5e-5` and `lora_dropout=0.1`. This is a confound that invalidates any headline M1-vs-M3 comparison drawn from the tuned runs. The config at `scripts/configs/lora_rh_m4_instruct_5t.yaml` has been corrected (Part C1 of the 2026-04-14 config review) to match M1. See `docs/m3_rerun_plan.md` for the rerun command and WandB naming convention. The runs below are retained as the "M3-tuned" ablation.

#### M3-tuned (pre-fix config — kept as secondary data point)

GCS base: `gs://statistical-nlp/NAMM_checkpoints/pretrained/`

| Cache | WandB ID   | State   | Steps | Best val F1          | GCS subpath                               |
| ----- | ---------- | ------- | ----- | -------------------- | ----------------------------------------- |
| 1024  | `ovosogkj` | crashed | 609   | **45.59** (step 340) | `lora-m4-frozen-5t-cs1024-llama32-1b/`    |
| 2048  | `m4knrhmr` | crashed | 371   | **44.86** (step 244) | `lora-m4-frozen-5t-cs2048-llama32-1b/`    |
| 3072  | `4sgkswa6` | done    | 117   | 33.37                | **not uploaded**                           |

WandB run names: `rh_m4_5t_cs{1024,2048,3072}`.

The cs1024 and cs2048 runs crashed before completing all 150 epochs but best checkpoints were saved before crash. The cs3072 run finished but at only epoch 4 — may need rerunning with more epochs.

#### M3-matched (post-fix config, matching M1 exactly except NAMM)

**Not yet started.** Will use run names `m3_cs{1024,2048,3072}_matched` and WandB group `m3_matched`.

> **Naming note:** WandB names `rh_m4_5t_cs*` refer to M3 (frozen NAMM), not M4 (joint). The `rh_m4` prefix comes from the `rh_m4_frozen` method in the LoRA config. The NAMM checkpoint used by each M3 run is the corresponding M2 run at the same cache size.

> **WARNING — results directory mislabelling:** The `results/main_table_5t/M4/` directory contains M3 (frozen NAMM) eval results, NOT M4 (joint) results. This mislabelling propagated from the `rh_m4_frozen` config name into `scripts/organize_eval_results.py` and all downstream plots/reports. The correct mapping is: `results/main_table_5t/M4/ → experiment spec M3`. Similarly, `A4/` ablations disable the frozen NAMM from M3 checkpoints, not from a joint M4 run. All analysis reports written before this note was added use "M4" to mean M3-frozen-NAMM.

### Maskfix Reruns — Attention Mask Bug Fix

An attention mask bug was discovered in the NAMM split-processing loop (`namm/llms/llama.py` line 445): the attention mask grows with cumulative input length rather than tracking the actual post-eviction cache size. From chunk 9 onward (~2300 tokens into a ~5000-token prompt), attention collapses to perfectly uniform 1/N across all heads and layers. This bug exists in the original Sakana AI codebase and affects all results above. See `scripts/diagnose_attention_mask.py` for the diagnostic.

Maskfix runs retrain M2 and M3 with the corrected attention mask.

#### M2 maskfix — Standalone NAMM (CMA-ES, corrected attention)

| Cache | WandB run name       | WandB ID   | State    | Iters | Best val mean F1     |
| ----- | -------------------- | ---------- | -------- | ----- | -------------------- |
| 1024  | `...-cs1024-maskfix` | `z5bo4n8k` | finished | 200   | **14.90** (iter 170) |
| 2048  | `...-cs2048-maskfix` | `jip3a3dm` | running  | —     | —                    |

Earlier killed/crashed attempts: `upy2r2wr`, `dq14qqr2` (cs1024), `6yfx7846` (cs2048).

M2 maskfix cs1024 val F1 (14.90) is substantially **worse** than buggy M2 (27.90). The NAMM eviction policy, trained with correct attention, performs worse — likely because the CMA-ES hyperparameters (pop_size=8, sigma=0.065) were tuned for the buggy regime. The optimisation landscape is different with correct attention.

#### M3 maskfix — LoRA + frozen NAMM (corrected attention)

| Cache | WandB run name            | WandB ID   | State   | Steps | Best val avg F1              |
| ----- | ------------------------- | ---------- | ------- | ----- | ---------------------------- |
| 1024  | `rh_m4_5t_cs1024_maskfix` | `h0bzg6on` | killed  | 425   | **52.06** (step 260)         |

Earlier killed attempt: `facswin9`.

M3 maskfix cs1024 exceeds both buggy M3 (45.59) and M1 (45.48) by ~6.5 points. Run was killed at step 425; best val F1 at step 260. Multi-hop tasks see the largest gains: HotpotQA-E +14.3, 2WikiMQA-E +19.5, 2WikiMQA +12.0.

#### Maskfix checkpoints

Local: `experiment_artifacts/gcs/M2_cs1024_maskfix/ckpt.pt`, `experiment_artifacts/gcs/M3_cs1024_maskfix/best_ckpt.pt` (downloaded from WandB artifacts).

GCS backup: `gs://statistical-nlp/evo-memory/checkpoints_backup_20260414/` contains:
- `namm_cs1024_maskfix/ckpt.pt` — M2 maskfix NAMM policy
- `lora_m4_cs1024_maskfix/best_ckpt_step260_val52.06.pt` — M3 maskfix LoRA (best so far)
- `namm_fix/llama.py` — the corrected attention mask code
- All original (buggy) checkpoints for reference

### M4 — Joint LoRA + NAMM

**Not yet started.** No runs matching the M4 joint training protocol exist in WandB.

### B0, B1 — Baselines

**Done.** Dedicated eval runs completed for B0 (full cache) and B1 (recency eviction at cs1024 and cs2048). Results in `results/main_table_5t/B0/` and `results/main_table_5t/B1/`. Test micro F1: B0 = 22.41, B1/cs1024 = 11.33, B1/cs2048 = 11.10.

### A1 — LoRA rank sweep

**Not yet run.** Only r=8 has been trained (M1). r=4 and r=16 still needed.

### A4 — NAMM disabled at eval

**Done.** Evaluated M3 (LoRA + frozen NAMM) checkpoints with NAMM disabled at eval time (full KV cache). Originally specified to use M4 (joint) checkpoints, but M4 has not been run; A4 was executed on M3 checkpoints instead. Results in `results/main_table_5t/A4/`. Test micro F1: cs1024_no_namm = 28.82, cs2048_no_namm = 38.98.

### Completion summary

| Step | Experiment           | Status         | Notes                                       |
| ---- | -------------------- | -------------- | ------------------------------------------- |
| 1    | B0 — baseline        | **done**       | test micro 22.41                            |
| 2    | B1 — recency         | **done**       | cs1024: 11.33, cs2048: 11.10                |
| 3    | M1 — LoRA (r=8)      | **done**       | test micro 31.14                            |
| 4    | M2 — standalone NAMM | **done**       | cs1024: 10.83, cs2048: 15.27                |
| 5    | M3 — LoRA+frozen NAMM| **rerun needed**| tuned: 23.52/31.41 (lr/dropout mismatch)    |
| 6    | M4 — joint LoRA+NAMM | **not started**| —                                           |
| 7    | A1 — rank sweep      | **not started**| r=4, r=16 not trained; r=8 = M1             |
| 8    | A4 — NAMM off eval   | **done**       | on M3 ckpts; cs1024: 28.82, cs2048: 38.98   |
| —    | Trunc baselines      | **done**       | plain: 18.21/18.26, lora: 26.90/28.87       |
| —    | M1_recency/cs1024    | **BROKEN**     | all zeros — needs investigation              |
| —    | M2 maskfix cs1024    | **done**       | val F1 14.90 (worse than buggy 27.90)        |
| —    | M2 maskfix cs2048    | **running**    | `jip3a3dm`                                   |
| —    | M3 maskfix cs1024    | **killed@425** | `h0bzg6on`, val F1 52.06 at step 260        |

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
