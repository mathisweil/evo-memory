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

| Step | Experiment                                    | Dependency           |
| ---- | --------------------------------------------- | -------------------- |
| 1    | B0 — baseline eval                            | none                 |
| 2    | B1 — recency eviction eval                    | none                 |
| 3    | B2 — H2O eviction eval                        | none                 |
| 4    | B3 — ScissorHands eviction eval               | none                 |
| 5    | M1-LoRA — rank sweep (r=4, 8, 16)             | none                 |
| 6    | M2 — standalone NAMM training                 | none                 |
| 7    | M3-LoRA — LoRA with frozen NAMM at train-time | M2 checkpoint        |
| 8    | M4-LoRA — joint LoRA + NAMM                   | none                 |
| 9    | A1 — LoRA rank sweep analysis                 | M1-r4, M1-r8, M1-r16 |
| 10   | A4 — NAMM disabled at eval                    | M4-LoRA checkpoint   |

---

## 2 · Tier 1 — Baselines

These require no training. Run first to establish reference scores.

---

### B0 · Base model, full KV cache

Evaluates raw Llama-3.2-1B-Instruct on the 5-task QA subset with no KV cache limit, no eviction, no fine-tuning. This is the absolute performance floor.

```bash
python scripts/run/run_eval.py \
    --run_config full_cache_baseline_llama32_1b \
    --output_dir experiments/experiment_N/baseline
```

| Parameter         | Value                                                       |
| ----------------- | ----------------------------------------------------------- |
| `es_checkpoint`   | null — evaluates base model weights                         |
| `namm_checkpoint` | null — no eviction, full KV cache                           |
| `run_config`      | `full_cache_baseline_llama32_1b` — no-eviction policy       |
| `cache_size`      | null — no limit                                             |
| Tasks             | 5-task QA subset (override task config to `rh_multi_qa_5t`) |
| Splits            | `train_frac=0.7`, `val_frac=0.15`, `split_seed=42`          |
| Output            | `experiments/experiment_N/baseline/results.json`            |

---

### B1 · Base model, recency eviction

Evaluates the base model with a fixed recency policy: keeps the most recently written tokens, evicts oldest first. No learned policy, no training. Establishes what a naive heuristic achieves at `cache_size=1024`.

```bash
python scripts/run/run_eval.py \
    --run_config recency_baseline_llama32_1b \
    --cache_size 1024 \
    --output_dir experiments/experiment_N/es_recency/b1_recency \
    --override "task@_global_=rh_multi_qa_5t"
```

| Parameter    | Value                                                                                 |
| ------------ | ------------------------------------------------------------------------------------- |
| `run_config` | `recency_baseline_llama32_1b` — matches `config/run/recency_baseline_llama32_1b.yaml` |
| `cache_size` | 1024                                                                                  |
| Tasks        | 5-task QA subset (override task config to `rh_multi_qa_5t`)                           |
| Splits       | `train_frac=0.7`, `val_frac=0.15`, `split_seed=42`                                    |
| Output       | `experiments/experiment_N/es_recency/b1_recency/results.json`                         |

---

### B2 · Base model, H2O eviction (Zhang et al., NeurIPS 2023)

Evaluates the base model with the H2O Heavy-Hitter Oracle policy: per-(layer, KV-head) accumulator of post-softmax attention; keeps the top `k_hh = round(B · heavy_hitter_ratio)` heavy hitters and the most recent `k_recent = B − k_hh` tokens. No learned policy, no training. Uses the same FAIR-01 budget `cache_size=1024` as B1, M2, M3, M4.

```bash
python scripts/run/run_eval.py \
    --run_config h2o_baseline_llama32_1b \
    --cache_size 1024 \
    --output_dir experiments/experiment_N/baselines/b2_h2o \
    --override "task@_global_=rh_multi_qa_5t"
```

| Parameter             | Value                                                                |
| --------------------- | -------------------------------------------------------------------- |
| `run_config`          | `h2o_baseline_llama32_1b` — `config/run/h2o_baseline_llama32_1b.yaml` |
| Policy target         | `namm.policy.H2O`                                                    |
| `heavy_hitter_ratio`  | 0.5 (k_hh = 512, k_recent = 512 at cache_size=1024)                  |
| `cache_size`          | 1024                                                                 |
| Tasks                 | 5-task QA subset (override task config to `rh_multi_qa_5t`)          |
| Splits                | `train_frac=0.7`, `val_frac=0.15`, `split_seed=42`                   |
| Output                | `experiments/experiment_N/baselines/b2_h2o/results.json`             |

---

### B3 · Base model, ScissorHands eviction (Liu et al., NeurIPS 2023)

Evaluates the base model with the ScissorHands persistence-of-importance policy: counts how often each token's mean-pooled post-softmax attention falls below `1/t` over a sliding history window; protects the most recent `recent_window` tokens; drops the `drop_count` tokens with the highest unimportance count when `kv_len > cache_size`. No learned policy, no training.

```bash
python scripts/run/run_eval.py \
    --run_config scissorhands_baseline_llama32_1b \
    --cache_size 1024 \
    --output_dir experiments/experiment_N/baselines/b3_scissorhands \
    --override "task@_global_=rh_multi_qa_5t"
```

| Parameter               | Value                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| `run_config`            | `scissorhands_baseline_llama32_1b` — `config/run/scissorhands_baseline_llama32_1b.yaml`     |
| Policy target           | `namm.policy.ScissorHands`                                                                  |
| `history_window_ratio`  | 0.5 (history window = 512 at cache_size=1024)                                               |
| `recent_window_ratio`   | 0.25 (recent window = 256 at cache_size=1024, protected from eviction)                      |
| `drop_ratio`            | 0.5 (drop 512 tokens per compression event)                                                 |
| `cache_size`            | 1024                                                                                        |
| Tasks                   | 5-task QA subset (override task config to `rh_multi_qa_5t`)                                 |
| Splits                  | `train_frac=0.7`, `val_frac=0.15`, `split_seed=42`                                          |
| Output                  | `experiments/experiment_N/baselines/b3_scissorhands/results.json`                           |

---

## 3 · Tier 2 — Four Main Conditions

These fill the primary results table. All four use the same 5-task QA dataset with 70/15/15 splits and must be evaluated with `cache_size=1024` and greedy decoding for a fair comparison.

---

### M1 · LoRA only — SFT fine-tuning, no NAMM

Supervised fine-tuning (SFT) with LoRA on the 5-task QA subset with full KV cache during training. Uses chat-template formatted prompts with answer-only loss. Run three times across the rank sweep. **r=8 is the main table entry**; r=4 and r=16 feed ablation A1.

#### M1-r4

```bash
python scripts/run/run_lora.py \
    --config scripts/configs/m1_lora_5t.yaml \
    --run_name m1_r4 \
    --lora_rank 4 \
    --lora_alpha 8
```

#### M1-r8 (main table)

```bash
python scripts/run/run_lora.py \
    --config scripts/configs/m1_lora_5t.yaml \
    --run_name m1_r8
```

#### M1-r16

```bash
python scripts/run/run_lora.py \
    --config scripts/configs/m1_lora_5t.yaml \
    --run_name m1_r16 \
    --lora_rank 16 \
    --lora_alpha 32
```

| Parameter                     | Value                                                                                  |
| ----------------------------- | -------------------------------------------------------------------------------------- |
| Config                        | `scripts/configs/m1_lora_5t.yaml`                                          |
| `method`                      | `m1_lora_5t` (historical: `rh_m1_lora_instruct_5t`)                                    |
| `sft_mode`                    | true — supervised fine-tuning with chat template                                       |
| `learning_rate`               | 5e-5                                                                                   |
| `num_epochs`                  | 150                                                                                    |
| `batch_size`                  | 1 — matches M3/M4 for controlled comparison (was 4 in earlier drafts)                  |
| `gradient_accumulation_steps` | 16 (effective batch = 16)                                                              |
| `max_seq_len`                 | 7000                                                                                   |
| `namm_active`                 | false — full KV cache during training                                                  |
| `eval_interval`               | 14 steps (was 2 in earlier drafts — kept at 14 for practical wall-clock)               |
| `batch_size_eval`             | 2                                                                                      |
| `early_stopping_patience`     | 20 (≈14 epochs; raised from 5 which was too aggressive for 150-epoch schedule)         |
| `lora_rank`                   | 8                                                                                      |
| `lora_alpha`                  | 16                                                                                     |
| `lora_dropout`                | 0.1                                                                                    |
| `lora_target_modules`         | `[q_proj, v_proj]`                                                                     |
| `warmup_ratio`                | 0.03                                                                                   |
| `weight_decay`                | 0.01                                                                                   |
| `max_grad_norm`               | 1.0                                                                                    |
| Tasks                         | 5-task QA: qasper, 2wikimqa, qasper_e, hotpotqa_e, 2wikimqa_e                          |
| Splits                        | `train_frac=0.7`, `val_frac=0.15`, `split_seed=42` (306 train / 64 val / 70 test)      |
| Filtering                     | `min_conditioning_length=4096`, `max_conditioning_length=6500`, `max_answer_tokens=64` |
| `seed`                        | 42                                                                                     |
| Output                        | `experiments/experiment_N/m1_lora_only/{m1_r4,m1_r8,m1_r16}/`                          |

---

### M2 · Standalone NAMM — frozen LLM, trained from scratch

Trains the NAMM eviction policy via CMA-ES on a frozen Llama-3.2-1B-Instruct using the 5-task QA subset. LLM weights do not change. **No pretrained checkpoint is used** — training starts from random NAMM parameters. This is the learned-eviction-only baseline and the source checkpoint for M3.

```bash
python scripts/run/run_namm.py \
    'run@_global_=namm_bam_i1_llama32_1b_5t' \
    wandb_run_name=m2_namm_standalone \
    wandb_group_name=main_conditions \
    seed=1337
```

| Parameter                   | Value                                                            | Notes                                  |
| --------------------------- | ---------------------------------------------------------------- | -------------------------------------- |
| Config                      | `config/config.yaml` (Hydra)                                     |                                        |
| `run` preset                | `namm_bam_i1_llama32_1b_5t`                                      | BAM policy, i1 architecture, 5-task    |
| `pop_size`                  | 8                                                                | From preset                            |
| `elite_ratio`               | 0.5                                                              | Standard CMA-ES setting                |
| `init_sigma`                | 0.065                                                            | Already validated in prior runs        |
| `memory_policy_fixed_delay` | 256                                                              | From preset                            |
| `min_conditioning_length`   | 4096                                                             | Filters out short prompts              |
| `max_conditioning_length`   | 6500                                                             | Filters which prompts are eligible     |
| `max_memory_length`         | 1024                                                             | KV cache budget (4× compression ratio) |
| `max_answer_tokens`         | 64                                                               |                                        |
| `samples_batch_size`        | 8                                                                | Prompts per task per step              |
| `batch_size`                | 4                                                                | Sequences per GPU forward pass         |
| `max_iters`                 | 200                                                              | Preset default; 200 generations        |
| `eval_interval`             | 5                                                                |                                        |
| `max_new_tokens`            | 256                                                              | Chunk size for generation              |
| `save_checkpoint_every`     | null (save every iter)                                           | Set to N to reduce I/O                 |
| Tasks                       | 5-task QA: qasper, 2wikimqa, qasper_e, hotpotqa_e, 2wikimqa_e    |                                        |
| Splits                      | `train_frac=0.7`, `val_frac=0.15` (306 train / 64 val / 70 test) |                                        |
| `seed`                      | 1337                                                             |                                        |
| Output                      | `outputs/{date}/{time}/` (Hydra default)                         |                                        |

> If training curves are still improving at generation 200, extend to 300+, but do not plan for it upfront.

> **Threshold-only variant:** append `threshold_only=true scoring_initializer=2` to run M2 with the original NAMM paper's eviction rule (score threshold only, no hard top-k cap). The cache size will vary dynamically per step.
> ```bash
> python scripts/run/run_namm.py \
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
python scripts/run/run_lora.py \
    --config scripts/configs/m3_lora_frozen_namm_5t.yaml \
    --run_name m3_lora_frozen_namm \
    --namm_checkpoint <path-to-m2-checkpoint>
```

| Parameter                     | Value                                                                                  |
| ----------------------------- | -------------------------------------------------------------------------------------- |
| Config                        | `scripts/configs/m3_lora_frozen_namm_5t.yaml`                                          |
| `method`                      | `m3_lora_frozen_namm_5t` (historical: `rh_m4_frozen_5t`)                               |
| `sft_mode`                    | true — supervised fine-tuning with chat template                                       |
| `namm_active`                 | true + NAMM checkpoint (required)                                                      |
| `learning_rate`               | 5e-5 — MATCHES M1 (was 1e-4 in the first M3 runs; those are now labelled "M3-tuned", see §6 and `docs/m3_rerun_plan.md`) |
| `num_epochs`                  | 150                                                                                    |
| `batch_size`                  | 1                                                                                      |
| `gradient_accumulation_steps` | 16 (effective batch = 16)                                                              |
| `max_seq_len`                 | 7000                                                                                   |
| `lora_rank`                   | 8                                                                                      |
| `lora_alpha`                  | 16                                                                                     |
| `lora_dropout`                | 0.1 — MATCHES M1 (was 0.05 in the first M3 runs)                                       |
| `lora_target_modules`         | `[q_proj, v_proj]`                                                                     |
| `eval_interval`               | 14 steps                                                                               |
| `batch_size_eval`             | 2                                                                                      |
| `early_stopping_patience`     | 20                                                                                     |
| `cache_size`                  | 1024                                                                                   |
| Tasks                         | 5-task QA: qasper, 2wikimqa, qasper_e, hotpotqa_e, 2wikimqa_e                          |
| Splits                        | `train_frac=0.7`, `val_frac=0.15`, `split_seed=42` (306 train / 64 val / 70 test)      |
| Filtering                     | `min_conditioning_length=4096`, `max_conditioning_length=6500`, `max_answer_tokens=64` |
| `seed`                        | 42                                                                                     |
| What it answers               | Does training LoRA under compressed context help, even without co-optimising the NAMM? |

---

### M4-LoRA · Joint — simultaneous alternating NAMM + LoRA (primary gradient contribution)

NAMM and LoRA are co-trained in alternating stages: Stage A evolves the NAMM for N CMA-ES iterations; Stage B gradient-updates the LoRA for E epochs. Repeated for K outer loops. **Cold-start only** — no pretrained NAMM checkpoint. Must use the same 5-task dataset, splits, and filtering as M1–M3.

> **3-loop schedule rationale:** 3 outer loops × (67 NAMM + 50 LoRA) = 201 NAMM + 150 LoRA, matching M1+M2's compute budget. 3 loops gives 2 co-adaptation cycles instead of 1 (the earlier 2-loop schedule had only one). Each LoRA stage early-stops at ~30 epochs via best-checkpoint selection; 50 allocated epochs gives ~20 epochs headroom. NAMM warm-starts from the previous loop's checkpoint (see `docs/m4_joint_training_analysis.md`).

```bash
python scripts/run/run_joint.py \
    --config scripts/configs/m4_joint_lora_5t.yaml \
    --run_name m4_joint_lora
```

| Parameter                   | Value                                                                                  | Notes                                  |
| --------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------- |
| Config                      | `scripts/configs/m4_joint_lora_5t.yaml` (the M4 FAIR-01 preset)                        | `joint_default.yaml` is an ad-hoc preset — do not use it for the M4 paper run |
| `adapter_type`              | `lora`                                                                                 |                                        |
| `num_outer_loops`           | 3                                                                                      | Final adapter stage is `stage_2` (0-indexed) |
| `namm_iterations_per_stage` | 67                                                                                     | 3 × 67 = 201 total — matches M2       |
| `lora_epochs_per_stage`     | 50                                                                                     | 3 × 50 = 150 total epochs — matches M1 |
| `lora_rank`                 | 8                                                                                      | Matches M1-r8                          |
| `lora_alpha`                | 16                                                                                     | Matches M1                             |
| `learning_rate`             | 5e-5                                                                                   | Matches M1                             |
| `lora_dropout`              | 0.1                                                                                    | Matches M1                             |
| `lora_batch_size`           | 1                                                                                      | Matches M1/M3                          |
| `gradient_accumulation_steps` | 16                                                                                   | Effective batch = 16                   |
| `max_seq_len`               | 7000                                                                                   | Matches M1                             |
| `lora_eval_interval`        | 14                                                                                     | Mid-stage best-checkpointing (was 999999 pre-C3 fix) |
| `sft_mode`                  | true                                                                                   | Matches M1                             |
| `cache_size`                | 1024                                                                                   | NAMM evicts during training            |
| `namm_checkpoint`           | null — cold-start                                                                      | No pretrained checkpoint               |
| `eval_after_each_loop`      | true                                                                                   | Per-loop F1 saved in `results.json`    |
| Tasks                       | 5-task QA: qasper, 2wikimqa, qasper_e, hotpotqa_e, 2wikimqa_e                          |                                        |
| Splits                      | `train_frac=0.7`, `val_frac=0.15`, `split_seed=42`                                     |                                        |
| Filtering                   | `min_conditioning_length=4096`, `max_conditioning_length=6500`, `max_answer_tokens=64` |                                        |
| Output                      | `experiments/experiment_N/joint_lora/m4_joint_lora/`                                   |                                        |

> M4 asymmetries to report in the paper: each LoRA stage runs a fresh 50-epoch warmup+cosine schedule (not one continuous 150-epoch curve), so M4 has three separate warmups vs M1's single warmup. See `docs/m4_joint_training_analysis.md` for the full list of joint-vs-standalone asymmetries and their treatment in the paper.

---

## 4 · Tier 3 — Ablations

These support the discussion sections and answer the research questions. Most require Tier 2 to be complete first.

---

### A1 · LoRA rank sweep (r = 4, 8, 16)

No additional runs needed — results come from M1-r4, M1-r8, and M1-r16. Report all three F1 scores in a table in the ablation section to justify the r=8 choice for all other conditions.

|                 |                                                          |
| --------------- | -------------------------------------------------------- |
| Source runs     | M1-r4, M1-r8, M1-r16                                     |
| Metric          | 5-task avg F1 on test split                              |
| What it answers | Effect of adapter expressivity; justifies rank selection |

---

### A4 · NAMM disabled at eval on M4 checkpoint (modularity test)

Takes the fully trained M4 checkpoint and evaluates it twice: once with NAMM active, once with NAMM disabled (full KV cache). The F1 delta isolates how much of M4's performance comes from the LoRA alone versus the LoRA+NAMM system.

#### M4 — NAMM active (standard)

```bash
python scripts/run/run_eval.py \
    --es_checkpoint experiments/experiment_N/joint_lora/m4_joint_lora/adapter/stage_2/ \
    --namm_checkpoint experiments/experiment_N/joint_lora/m4_joint_lora/namm/latest.pt \
    --cache_size 1024 \
    --output_dir experiments/ablations/a4_modularity/m4_namm_on
```

#### M4 LoRA only — NAMM disabled

```bash
python scripts/run/run_eval.py \
    --es_checkpoint experiments/experiment_N/joint_lora/m4_joint_lora/adapter/stage_2/ \
    --output_dir experiments/ablations/a4_modularity/m4_namm_off
# No --namm_checkpoint → full KV cache, no eviction
```

|                    |                                                                                                                         |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Adapter checkpoint | `joint_lora/m4_joint_lora/adapter/stage_2/` — stages are 0-indexed; final stage with `--num_outer_loops 3` is `stage_2` |
| NAMM checkpoint    | `joint_lora/m4_joint_lora/namm/latest.pt` — written after each NAMM stage; always reflects the most recent stage        |
| What it answers    | RQ4: are jointly-trained NAMM and LoRA co-dependent, or is the LoRA sufficient alone?                                   |
| Expected finding   | M4-NAMM-on > M4-NAMM-off ≈ M1 (LoRA adapts to compressed context; removing NAMM at eval hurts)                          |

---

## 5 · Results Collection

### Generating the cross-experiment report

```bash
python scripts/reporting/generate_report.py \
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
  "method": "m1_lora_5t"
}
```

Joint runs produce a list of entries (one per outer loop), giving per-loop learning curves for free.

### Main results table (FAIR-01 gradient conditions)

All F1 values are test-split micro averages over 70 samples. See `results/main_table_5t/all_results.json` for full per-task breakdowns.

| Condition                                | Run name              | F1 (test micro) | Cache              |
| ---------------------------------------- | --------------------- | --------------- | ------------------ |
| B0  Base model, full cache               | `baseline_eval`       | 22.41           | full               |
| B1  Base model + recency                 | `b1_recency`          | 12.45           | 1024               |
| B1  Base model + recency                 | `b1_recency`          | 13.78           | 2048               |
| B2  Base model + H2O                     | `b2_h2o`              | —               | 1024               |
| B3  Base model + ScissorHands            | `b3_scissorhands`     | —               | 1024               |
| M1-LoRA  LoRA only (r=8)                 | `m1_r8`               | 31.14           | full               |
| M2  Standalone NAMM                      | `m2_namm_standalone`  | 20.30           | 1024               |
| M2  Standalone NAMM                      | `m2_namm_standalone`  | 17.40           | 2048               |
| M3-LoRA  LoRA + frozen NAMM              | `m3_lora_frozen_namm` | 32.28           | 1024               |
| M3-LoRA  LoRA + frozen NAMM              | `m3_lora_frozen_namm` | 31.06           | 2048               |
| M4-LoRA  Joint LoRA + NAMM               | `m4_joint_lora`       | —               | —                  |
| Trunc/plain  Base model, truncated input | —                     | 18.21           | 1024               |
| Trunc/plain  Base model, truncated input | —                     | 18.26           | 2048               |
| Trunc/lora_m1  M1 LoRA, truncated input  | —                     | 26.90           | 1024               |
| Trunc/lora_m1  M1 LoRA, truncated input  | —                     | 28.87           | 2048               |
| M1_recency  M1 LoRA + recency eviction   | —                     | 0.00 (BROKEN)   | 1024               |
| A4  M3 LoRA, NAMM disabled               | —                     | 28.82           | full (cs1024 ckpt) |
| A4  M3 LoRA, NAMM disabled               | —                     | 33.91           | full (cs2048 ckpt) |

> **Note on M1 eval cache:** M1 trains with full cache and is also evaluated with full cache (no NAMM, no eviction). The FAIR-01 memory-equivalence constraint applies to conditions that use NAMM at eval time; M1 serves as an upper bound showing what LoRA alone can achieve without any cache compression.

> **Note on A4:** A4 was originally specified to use M4 (joint) checkpoints, but M4 has not been run. The A4 results above use M3 (LoRA + frozen NAMM) checkpoints with NAMM disabled at eval time.

---

## 6 · Completed Runs & Checkpoints

All runs below use the 5-task LongBench QA subset (qasper, 2wikimqa, qasper_e, hotpotqa_e, 2wikimqa_e) on LLaMA 3.2-1B-Instruct. WandB entity: `SNLP_NAMM`, project: `memory_evolution_hf`.

### M2 — Standalone NAMM (CMA-ES, no LoRA)

These are pure NAMM eviction policy training runs. The long auto-generated names encode hyperparameters: `p8` = pop_size 8, `8qs` = samples_batch_size 8, `256fixDel` = fixed_delay 256.

| Cache | WandB run name                                                                      | WandB ID                | State    | Iters | GCS checkpoint                                                                |
| ----- | ----------------------------------------------------------------------------------- | ----------------------- | -------- | ----- | ----------------------------------------------------------------------------- |
| 1024  | `rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs1024` | `lenhmfb1`              | finished | 200   | `gs://statistical-nlp/NAMM_checkpoints/pretrained/namm-5t-cs1024-llama32-1b/` |
| 2048  | `rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs2048` | `y5fdw0f9` → `ccflnsds` | finished | 200   | `gs://statistical-nlp/NAMM_checkpoints/pretrained/namm-5t-cs2048-llama32-1b/` |
| 3072  | `rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs3072` | `quc95irz`              | finished | 200   | **not uploaded**                                                              |

The cs2048 run was split across two wandb segments: `y5fdw0f9` (iters 0–30, failed on `scaup-l`) was resumed with `scratch=false` as `ccflnsds` (iters 30–200, on `mandarin-l`). Earlier duplicate runs also exist at these names with different IDs (crashed/failed); the IDs above are the definitive finished runs.

#### M2 best validation metrics

| Cache | Best iter | Val mean F1 | Val tasks_aggregate |
| ----- | --------- | ----------- | ------------------- |
| 1024  | 105       | 27.90       | 0.00279             |
| 2048  | 40        | 27.67       | 0.00277             |
| 3072  | —         | —           | —                   |

### M1 — LoRA only (no NAMM, full context)

Single logical training run split across 3 WandB segments due to crashes/resumes. Best checkpoint is from the final segment (`qfoxxi2m`).

| Segment | WandB run name    | WandB ID   | State   | Epochs | Steps | Best val avg F1      |
| ------- | ----------------- | ---------- | ------- | ------ | ----- | -------------------- |
| 1       | `rh_m1_5t_v2`     | `kz6vqo2o` | crashed | 5      | 125   | 39.15                |
| 2       | `rh_m1_5t_resume` | `x9a4smmf` | failed  | 10     | 250   | 40.60                |
| 3       | `rh_m1_5t_resume` | `qfoxxi2m` | killed  | 28     | 684   | **45.48** (step 336) |

**GCS checkpoint:** `gs://statistical-nlp/NAMM_checkpoints/pretrained/lora-m1-5t-llama32-1b/` (best_ckpt.pt at step 336)

### M3 — LoRA with frozen NAMM at train-time

These runs train LoRA adapters while the frozen M2 NAMM actively evicts KV cache tokens. Historical WandB runs are named `rh_m4_5t_cs*` (predates the M-numbering — the file is now `m3_lora_frozen_namm_5t.yaml`).

> **M3 must be re-run.** The runs in the table below used `learning_rate=1e-4` and `lora_dropout=0.05` — both differ from M1's `learning_rate=5e-5` and `lora_dropout=0.1`. This is a confound that invalidates any headline M1-vs-M3 comparison drawn from the tuned runs. The config at `scripts/configs/m3_lora_frozen_namm_5t.yaml` has been corrected (Part C1 of the 2026-04-14 config review) to match M1. See `docs/m3_rerun_plan.md` for the rerun command and WandB naming convention. The runs below are retained as the "M3-tuned" ablation.

#### M3-tuned (pre-fix config — kept as secondary data point)

| Cache | WandB run name    | WandB ID   | State    | Epochs | Steps | Best val avg F1      | GCS checkpoint                                                                          |
| ----- | ----------------- | ---------- | -------- | ------ | ----- | -------------------- | --------------------------------------------------------------------------------------- |
| 1024  | `rh_m4_5t_cs1024` | `ovosogkj` | crashed  | 25     | 609   | **45.59** (step 340) | `gs://statistical-nlp/NAMM_checkpoints/pretrained/lora-m4-frozen-5t-cs1024-llama32-1b/` |
| 2048  | `rh_m4_5t_cs2048` | `m4knrhmr` | crashed  | 15     | 371   | **44.86** (step 244) | `gs://statistical-nlp/NAMM_checkpoints/pretrained/lora-m4-frozen-5t-cs2048-llama32-1b/` |
| 3072  | `rh_m4_5t_cs3072` | `4sgkswa6` | finished | 4      | 117   | 33.37                | **not uploaded**                                                                        |

The cs1024 and cs2048 runs crashed before completing all 150 epochs but best checkpoints were saved before crash. The cs3072 run finished but at only epoch 4 — may need rerunning with more epochs.

#### M3-matched (post-fix config, matching M1 exactly except NAMM)

**Not yet started.** Will use run names `m3_cs{1024,2048,3072}_matched` and WandB group `m3_matched`.

> **Historical naming:** WandB runs and GCS paths use `rh_m4_5t_cs*` for M3 runs and
> `rh_m4_frozen` as the method name. This predates the current M-numbering. Config files
> have been renamed to match the experiment specification (M3 = LoRA + frozen NAMM).
> The `results/main_table_5t/M4/` directory contains M3 results, not M4 joint results.

### M4 — Joint LoRA + NAMM

**Not yet started.** No runs matching the M4 joint training protocol exist in WandB.

### B0, B1 — Baselines

**Done.** Dedicated eval runs completed for B0 (full cache) and B1 (recency eviction at cs1024 and cs2048). Results in `results/main_table_5t/B0/` and `results/main_table_5t/B1/`. Test micro F1: B0 = 22.41, B1/cs1024 = 12.45, B1/cs2048 = 13.78.

### B2, B3 — Training-free heuristic baselines

**Not yet run.** H2O and ScissorHands policies and Hydra configs are implemented (`namm/policy/{h2o,scissorhands}.py`, `config/run/{h2o,scissorhands}_baseline_llama32_1b.yaml`); both pass their unit tests in `tests/test_h2o.py` and `tests/test_scissorhands.py`. Eval runs at `cache_size=1024` are pending.

### A1 — LoRA rank sweep

**Not yet run.** Only r=8 has been trained (M1). r=4 and r=16 still needed.

### A4 — NAMM disabled at eval

**Done.** Evaluated M3 (LoRA + frozen NAMM) checkpoints with NAMM disabled at eval time (full KV cache). Originally specified to use M4 (joint) checkpoints, but M4 has not been run; A4 was executed on M3 checkpoints instead. Results in `results/main_table_5t/A4/`. Test micro F1: cs1024_no_namm = 28.82, cs2048_no_namm = 33.91.

### Completion summary

| Step | Experiment                    | Status                                                                                    |
| ---- | ----------------------------- | ----------------------------------------------------------------------------------------- |
| 1    | B0 — baseline eval            | **done** (test micro 22.41)                                                               |
| 2    | B1 — recency eviction eval    | **done** (cs1024: 12.45, cs2048: 13.78)                                                   |
| 3    | B2 — H2O eviction eval        | **not started** (policy + config implemented, tests pass)                                 |
| 4    | B3 — ScissorHands eval        | **not started** (policy + config implemented, tests pass)                                 |
| 5    | M1-LoRA — rank sweep          | **partial** (r=8 done, test micro 31.14; r=4, r=16 not started)                           |
| 6    | M2 — standalone NAMM          | **done** (cs1024: 20.30, cs2048: 17.40; cs3072 finished but GCS upload pending)           |
| 7    | M3-LoRA — LoRA + frozen NAMM  | **rerun needed**: tuned runs done (cs1024: 32.28, cs2048: 31.06) with lr/dropout mismatch; matched runs not yet started |
| 8    | M4-LoRA — joint LoRA + NAMM   | **not started**                                                                           |
| 9    | A1 — LoRA rank sweep analysis | **not started** (r=4, r=16 not trained)                                                   |
| 10   | A4 — NAMM disabled at eval    | **done** (run on M3 ckpts, not M4; cs1024_no_namm: 28.82, cs2048_no_namm: 33.91)          |
| —    | Trunc baselines               | **done** (plain_1024: 18.21, plain_2048: 18.26, lora_m1_1024: 26.90, lora_m1_2048: 28.87) |
| —    | M1_recency/cs1024             | **BROKEN** (all zeros — likely LoRA+recency incompatibility, needs investigation)         |

---

## 7 · Smoke Tests

Run these before committing to any full experiment to confirm the pipeline works end-to-end.

#### M1 smoke test

```bash
python scripts/run/run_lora.py \
    --config scripts/configs/m1_lora_5t.yaml \
    --run_name smoke_m1 \
    --num_epochs 1 \
    --eval_interval 5 \
    --no-gcs
```

#### M4-LoRA joint smoke test

```bash
python scripts/run/run_joint.py \
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
python scripts/run/run_eval.py \
    --run_config full_cache_baseline_llama32_1b \
    --num_samples 10
```
