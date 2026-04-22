# evo-memory — Reproduction Specification

**Model:** Llama-3.2-1B-Instruct
**Benchmark:** 5-task LongBench QA subset (token-level F1)
**Cache budget at eval:** K = 1024 (EC regime)
**Hardware:** Single NVIDIA GPU

This document describes exactly how to reproduce the results reported in the TACL paper companion. It supersedes the earlier M1 / M2 / M3 / M4 naming. Filenames on disk (e.g. `m1_lora_5t.yaml`, `m3_lora_frozen_namm_5t.yaml`, `results/main_table_5t/M4/`) still carry the old scheme for historical reasons — the mapping is:

| Paper term | Meaning | On-disk artefact |
|---|---|---|
| **Base** | pretrained Llama-3.2-1B, no fine-tuning | — |
| **FTS** | LoRA trained with a full KV cache | `scripts/configs/m1_lora_5t.yaml`, `lora-m1-...` checkpoints |
| **FTE** | LoRA trained with NAMM eviction active | `scripts/configs/m3_lora_frozen_namm_5t.yaml`, `lora-m4-frozen-...` checkpoints, `results/main_table_5t/M4/` |
| **FC** | full cache at inference | — |
| **EC** | NAMM-evicted cache at inference (K = 1024) | — |

Configurations reported in Figure 1 use the form `{variant}-{regime}`: Base-FC, Base-EC, FTS-FC, FTS-EC, FTE-FC, FTE-EC.

---

## 1 · Reproducing the paper

Figures 1–7 in the paper are produced by three training runs plus a grid of evaluations.

| # | Step | Script | Config | Produces |
|---|---|---|---|---|
| 1 | **Train NAMM** | `scripts/run/run_namm.py` | `config/config.yaml` + `run@_global_=namm_bam_i1_llama32_1b_5t` | NAMM `.pt` (feeds Base-EC, FTS-EC, FTE-EC, and FTE training) |
| 2 | **Train FTS** (LoRA, full cache) | `scripts/run/run_lora.py` | `scripts/configs/m1_lora_5t.yaml` | FTS LoRA adapter |
| 3 | **Train FTE** (LoRA, NAMM active) | `scripts/run/run_lora.py` | `scripts/configs/m3_lora_frozen_namm_5t.yaml` + `--namm_checkpoint <step 1>` | FTE LoRA adapter |
| 4 | **Evaluate the six Figure-1 configs** | `scripts/run/eval_namm_splits.py` / `scripts/run/run_eval.py` | — | `results.json` per config |

Analyses (JS divergence, hidden-state drift, attention thirds, Jaccard mask stability, per-layer retention, NAMM score distributions, extended-test OOD) are run from the same checkpoints — see §6.

### 1.1 FAIR-01 constraints

All four Figure-1 LoRA cells (FTS-FC, FTS-EC, FTE-FC, FTE-EC) share:

- Data: QA sequences from Qasper, 2WikiMultihopQA, and HotpotQA in LongBench, plus Qasper and 2WikiMultihopQA from LongBench-E — five sources, each contributing 73–111 sequences. Filtered to `4096 ≤ tokenised length ≤ 6500` (440 sequences in total), stratified 70 / 15 / 15, `split_seed=42`. During training, each example is reweighted by the inverse of its source size.
- Base weights: raw `meta-llama/Llama-3.2-1B-Instruct`, no pretrained adapters.
- Eval cache budget: K = 1024 in the EC regime; unbounded in the FC regime.
- Decoding: greedy, `temperature=0.0`, `max_answer_tokens=64`.
- Checkpoint selection: best validation F1.

Changing any of the above in one cell requires changing it in all of them.

---

## 2 · Training NAMM

```bash
python scripts/run/run_namm.py \
    'run@_global_=namm_bam_i1_llama32_1b_5t' \
    wandb_run_name=namm_5t_cs1024 \
    seed=1337
```

| Parameter | Value | Source |
|---|---|---|
| Scoring network | BAM (hidden 32, 1 head, biased, backward masking, no RoPE) | `namm/policy/deep_scoring_bam.py` |
| Feature extraction | STFT spectrogram, `n_fft=32`, hop 16, Hann(32) periodic window, magnitude output | paper Table 1 |
| Spectrogram reduction | EMA, coefficient 0.99, non-learned | paper Table 1 |
| Cache size K | 1024 | preset |
| Max memory length | 1024 | preset |
| Policy firing period | every 256 tokens | `memory_policy_fixed_delay=256` |
| Processing chunk size | 256 tokens | paper Table 1 |
| CMA-ES `pop_size` | 8 | preset |
| Samples per batch | 8 | paper Table 1 |
| Training batch size | 12 | paper Table 1 |
| `max_iters` | 200 | preset |
| `eval_interval` | 5 | preset |
| Scoring initialiser | 0 | paper Table 1 |
| Tasks | `rh_multi_qa_5t` | preset |
| Filtering | `min_conditioning_length=4096`, `max_conditioning_length=6500` | preset |
| `max_answer_tokens` | 64 | paper Table 1 |
| Output | `outputs/{date}/{time}/` (Hydra default) | |

NAMM trains for 200 CMA-ES generations on the same 306-example train split used for FTS / FTE. The evolving `best_member` determines the eviction policy at every subsequent eval.

**Appendix C deviations from the original NAMM:** (i) eviction uses a cache-size-based top-K cutoff, not a score-threshold cutoff; (ii) an attention-mask bug in the Sakana reference implementation was fixed. See `namm/policy/deep_scoring_bam.py` and `docs/namm_ref_review.md`.

> **Threshold-only variant** (not reported in the paper): append `threshold_only=true scoring_initializer=2` for the original score-threshold eviction rule.

---

## 3 · Training FTS (LoRA, full cache)

```bash
python scripts/run/run_lora.py \
    --config scripts/configs/m1_lora_5t.yaml \
    --run_name fts
```

| Parameter | Value | Notes |
|---|---|---|
| `namm_active` | false | full KV cache at train time |
| `learning_rate` | 1e-4 | paper Table 2 |
| `num_epochs` | 150 | paper Table 2 |
| `batch_size` | 4 | micro-batch (paper Table 2) |
| `gradient_accumulation_steps` | 4 | effective batch = 16 |
| `lora_rank` | 8 | paper Table 2 |
| `lora_alpha` | 16 | paper Table 2 |
| `lora_dropout` | 0.05 | paper Table 2 |
| `lora_target_modules` | `[q_proj, v_proj]` | paper Table 2 |
| `weight_decay` | 0.01 | paper Table 2 |
| `max_grad_norm` | 1.0 | paper Table 2 |
| `warmup_ratio` | 0.03 | paper Table 2 |
| `max_seq_len` | 7000 | paper Table 2 |
| `early_stopping_patience` | 20 | paper Table 2 |
| `bf16` | true | paper Table 2 |
| `eval_interval` | 14 | |
| `sft_mode` | true | chat-template formatted prompt, answer-only loss |
| `split_seed` | 42 | paper Table 2 |
| Output | `experiments/experiment_N/m1_lora_only/fts/` | |

> **Known issue — `num_epochs` in the on-disk YAML.** `scripts/configs/m1_lora_5t.yaml` currently sets `num_epochs=100`. The paper's Table 2 budget (and the schedule used for the reported FTS checkpoint) is 150 epochs. Override on the CLI with `--num_epochs 150` or fix the YAML before re-running.

---

## 4 · Training FTE (LoRA, NAMM active)

```bash
python scripts/run/run_lora.py \
    --config scripts/configs/m3_lora_frozen_namm_5t.yaml \
    --run_name fte \
    --namm_checkpoint <path-to-namm.pt-from-step-1>
```

FTE uses the same LoRA hyperparameters as FTS (see §3). The only differences:

| Parameter | Value | Notes |
|---|---|---|
| `namm_active` | true | NAMM evicts during every gradient step |
| `namm_checkpoint` | required | from step 1; missing this silently falls back to a randomly-initialised NAMM |
| `cache_size` | 1024 | paper Table 2 |
| Output | `experiments/experiment_N/m3_lora_frozen_namm/fte/` | |

FTS and FTE are otherwise identical (same optimiser, rank, dropout, schedule). That matched setup is what makes the FTS-vs-FTE comparison in Figure 1 clean.

---

## 5 · Evaluating the six Figure-1 configurations

Each config is an evaluation at a single `(variant, regime)` combination. Use `scripts/run/eval_namm_splits.py` for EC (needs `--namm_checkpoint`); use `scripts/run/run_eval.py --run_config full_cache_baseline_llama32_1b` for FC.

```bash
# Base-FC — pretrained model, full cache
python scripts/run/run_eval.py \
    --run_config full_cache_baseline_llama32_1b \
    --override "task@_global_=rh_multi_qa_5t" \
    --output_dir experiments/experiment_N/figure1/base_fc

# Base-EC — pretrained model, NAMM eviction (K=1024)
python scripts/run/eval_namm_splits.py \
    --run_config namm_bam_i1_llama32_1b_5t \
    --namm_checkpoint <namm.pt> \
    --cache_size 1024 --splits test

# FTS-FC
python scripts/run/eval_namm_splits.py \
    --run_config full_cache_baseline_llama32_1b \
    --lora_checkpoint <fts best_ckpt.pt> \
    --splits test

# FTS-EC
python scripts/run/eval_namm_splits.py \
    --run_config namm_bam_i1_llama32_1b_5t \
    --lora_checkpoint <fts best_ckpt.pt> \
    --namm_checkpoint <namm.pt> \
    --cache_size 1024 --splits test

# FTE-FC
python scripts/run/eval_namm_splits.py \
    --run_config full_cache_baseline_llama32_1b \
    --lora_checkpoint <fte best_ckpt.pt> \
    --splits test

# FTE-EC
python scripts/run/eval_namm_splits.py \
    --run_config namm_bam_i1_llama32_1b_5t \
    --lora_checkpoint <fte best_ckpt.pt> \
    --namm_checkpoint <namm.pt> \
    --cache_size 1024 --splits test
```

Paper F1 values (Figure 1, test micro):

| Config | F1 |
|---|---|
| Base-FC | 22.0 |
| Base-EC | 12.6 |
| FTS-FC | 32.6 |
| FTS-EC | 19.0 |
| FTE-FC | 29.3 |
| FTE-EC | 28.9 |

Per-task breakdowns are in `results/main_table_5t/<config>/cs1024/results.json`.

---

## 6 · Analyses

All analyses consume the three checkpoints from §§2–4 and produce the figures in the paper. Commands below assume the checkpoints have already been produced.

### 6.1 JS divergence of next-token logits (Figure 2)

Mean Jensen–Shannon divergence of the predicted next-token distribution between the FC and EC regimes, computed over generation across all validation and test prompts, for each of the three variants. Paper values (Figure 2):

| Pair | Mean JS divergence |
|---|---|
| Base-FC ↔ Base-EC | 0.358 ± 0.012 |
| FTS-FC ↔ FTS-EC | 0.301 ± 0.010 |
| FTE-FC ↔ FTE-EC | 0.277 ± 0.010 |

> **Script pending.** No existing script in `scripts/analysis/` produces Figure 2 directly. Reproduce by (a) running a forward pass on the test split under each variant in both regimes, (b) saving the softmaxed next-token logits at each generation step, (c) computing mean JSD across generation positions and prompts.

### 6.2 Per-layer hidden-state ℓ2 drift (Figure 3)

Ratio of FTE's per-layer hidden-state ℓ2 drift under FC → EC to FTS's drift. Measured at the last token of the prompt. Paper Figure 3: the FTE-FC → FTE-EC transition exhibits on average **23.5 % greater** per-layer ℓ2 hidden-state drift than the FTS-FC → FTS-EC transition, with the excess concentrated in the **final three layers**.

```bash
python scripts/analysis/hidden_state_shift_analysis.py \
    --namm_checkpoint <namm.pt> \
    --lora_checkpoint <fte best_ckpt.pt> \
    --cache_size 1024 \
    --splits test \
    --output_dir analysis_out/hidden_drift_fte

python scripts/analysis/hidden_state_shift_analysis.py \
    --namm_checkpoint <namm.pt> \
    --lora_checkpoint <fts best_ckpt.pt> \
    --cache_size 1024 \
    --splits test \
    --output_dir analysis_out/hidden_drift_fts
```

Compute the per-layer ratio `ℓ2(FTE, FC→EC) / ℓ2(FTS, FC→EC)` from the two output directories. Ratio < 1 means FTE's representations are more stable under eviction than FTS's.

### 6.3 Attention mass by prompt third (Figure 4)

Fraction of attention mass directed at the first / middle / last third of the prompt, computed per layer, for FTE-FC minus FTS-FC. Paper Figure 4: averaged over layers and heads, FTE-FC places **+4.2 %** attention on the first third and **−5.0 %** on the last third relative to FTS-FC, with the divergence emerging from **layer 4 onwards**.

```bash
python scripts/analysis/eviction_representation_analysis.py \
    --variant fts --cache_size 1024 \
    --namm_checkpoint <namm.pt> \
    --lora_checkpoint <fts best_ckpt.pt> \
    --splits test \
    --output_dir analysis_out/attn_thirds_fts

python scripts/analysis/eviction_representation_analysis.py \
    --variant fte --cache_size 1024 \
    --namm_checkpoint <namm.pt> \
    --lora_checkpoint <fte best_ckpt.pt> \
    --splits test \
    --output_dir analysis_out/attn_thirds_fte
```

Per-chunk / per-layer / per-head `attn_first_third`, `attn_middle_third`, `attn_last_third` are written to the output directory. Figure 4 plots the FTE − FTS difference per layer.

> The script's `--variant` flag still uses legacy tokens (`plain|m1|m4`). `plain`≡Base, `m1`≡FTS, `m4`≡FTE. Do not rename the flag here — that's a separate refactor.

### 6.4 Jaccard index of retained token sets (Figure 5)

Pairwise mean Jaccard index between the token sets retained by NAMM when the cache is populated by Base, FTS, and FTE activations (same prompts, same NAMM checkpoint, different LoRA adapters). Paper Figure 5:

| Pair | Mean Jaccard |
|---|---|
| Base-EC ↔ FTS-EC | 0.952 |
| Base-EC ↔ FTE-EC | 0.782 |
| FTS-EC ↔ FTE-EC | 0.784 |

FTS preserves the Base eviction mask almost exactly; FTE drifts by ≈ 22 % against either reference.

> **Script pending.** No existing script produces Figure 5 directly. Reproduce by running each variant through the NAMM with `record_eval_stats=true`, saving the per-layer retained-token index sets, and computing pairwise mean Jaccard across variants per layer per prompt.

### 6.5 Per-layer retention fraction (Appendix Figure 6)

Fraction of retained prompt positions (union over heads) per layer for Base, FTS, FTE under the frozen NAMM. Paper Figure 6 caption reports retention-rate-per-layer is **essentially invariant** across the three conditions — the divergence shown in §6.4 arises from *which* positions are retained, not *how many*.

```bash
python scripts/analysis/check_eviction_stats.py \
    --cache_size 1024 \
    --namm_checkpoint <namm.pt> \
    --num_samples 70
```

Run once per variant with the corresponding LoRA checkpoint (or none, for Base). Figure 6 labels the three curves **B0 / M1 / M3** — these correspond to **Base / FTS / FTE** in the terminology above.

### 6.6 NAMM raw score distributions (Appendix Figure 7)

Raw NAMM scores pooled across the test split for Base / FTS / FTE, with pairwise Kolmogorov–Smirnov distances between the three distributions. Paper Figure 7:

| Pair | KS distance |
|---|---|
| Base ↔ FTS | 0.003 |
| Base ↔ FTE | 0.022 |
| FTS ↔ FTE | 0.020 |

All three marginals are essentially coincident (pairwise KS ≤ 0.022), which — together with §6.5 — isolates the FTS / FTE divergence to *ranking within the cache*, not to a shift in overall score magnitude or retention rate.

> **Script pending.** No existing script emits the raw score distributions directly. Reproduce by instrumenting `namm/policy/deep_scoring_bam.py` to log pre-threshold scores during a test-split eval, then computing pairwise KS distances.

### 6.7 Extended test OOD evaluation (154 examples, 6500–8192 tokens)

The paper evaluates all six Figure-1 configurations on a separate extended test set used only for OOD measurement.

```bash
python scripts/run/eval_namm_splits.py \
    --run_config namm_bam_i1_llama32_1b_5t \
    --lora_checkpoint <checkpoint> \
    --namm_checkpoint <namm.pt> \
    --cache_size 1024 \
    --splits extended_test
```

`--splits extended_test` selects the 6500–8192-token split. Swap `--run_config` to `full_cache_baseline_llama32_1b` for the FC regime, and omit `--lora_checkpoint` for the Base variant.

---

## 7 · Smoke tests

Run before launching a full experiment to confirm the pipeline works end-to-end.

```bash
# FTS smoke
python scripts/run/run_lora.py \
    --config scripts/configs/m1_lora_5t.yaml \
    --run_name smoke_fts \
    --num_epochs 1 \
    --eval_interval 5 \
    --no-gcs

# Eval smoke
python scripts/run/run_eval.py \
    --run_config full_cache_baseline_llama32_1b \
    --num_samples 10
```

---

## 8 · Exploratory code (not in the paper)

The repo contains code paths that were explored during research but did not feature in the final paper. They remain in the tree for reproducibility of the broader project, but are **not** needed to reproduce Figures 1–7.

- **Joint NAMM + LoRA training** — `scripts/run/run_joint.py`, `scripts/configs/m4_joint_lora_5t.yaml`. Discussed as future work in Section 6 of the paper.
- **Evolution Strategies fine-tuning** — `scripts/run/run_es.py`, `es_finetuning/`. Paper uses gradient LoRA only.
- **H2O and ScissorHands eviction baselines** — `namm/policy/{h2o,scissorhands}.py`, `config/run/{h2o,scissorhands}_baseline_llama32_1b.yaml`. Not reported.
- **Recency-only eviction baseline.**
- **Truncation baselines (Trunc/plain, Trunc/lora_m1).**
- **LoRA rank sweep (r=4, r=16).** Paper uses r=8 with no ablation.
- **Cache-size sweep (K=2048, K=3072).** Paper reports K=1024 only.
- **Per-prompt case studies** — `scripts/reporting/case_study_{attention,entropy}.py`. Exploratory diagnostics.
- **Paired-Δ / ghost-information / NAMM profiler scripts** — `scripts/analysis/{paired_delta_analysis,ghost_information_analysis,profile_namm}.py`. Utility / diagnostic code, not mapped to paper figures.

---

## 9 · Output layout

```
experiments/
└── experiment_N/
    ├── m1_lora_only/<run_name>/       # FTS runs
    │   ├── config.json
    │   ├── results.json
    │   └── checkpoints/best_ckpt.pt
    └── m3_lora_frozen_namm/<run_name>/ # FTE runs
        ├── config.json
        ├── results.json
        └── checkpoints/best_ckpt.pt
```

`scripts/run/run_namm.py` writes to `outputs/{date}/{time}/`. `scripts/run/run_eval.py` writes `results.json` next to the evaluated checkpoint (or `--output_dir`).
