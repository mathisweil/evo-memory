# TPU Version Instructions

This document explains what changed relative to `es-fine-tuning`, why those changes exist, and how to set up, validate, and run the TPU-enabled ES fine-tuning workflows on a new machine.

It is intended as an operator + engineering handoff guide.

---

## 1) What Changed From `es-fine-tuning`

The TPU branch is not just a hardware switch; it introduces runtime, shape-stability, and operations changes needed for reliable XLA execution.

### 1.1 High-Level Delta

Compared to `es-fine-tuning`, this branch adds or changes:

1. Device abstraction and TPU-safe execution paths.
2. XLA shape-hardening in cache eviction and batching paths.
3. TPU operational tooling (setup, activation, warmup, smoke matrix).
4. Spot/preemption + checkpoint resume workflows.
5. GCS-backed experiment management, with explicit opt-in XLA cache sync.
6. Strict TPU run presets for reproducible fixed-shape runs.
7. GCS-backed pretrained NAMM checkpoint lookup and local caching.

### 1.2 Subsystem-Level Changes

| Area | Change | Why it exists |
|---|---|---|
| Device/runtime | `es_finetuning/device.py` used by entrypoints; `.to(device)` patterns | Removes CUDA-only assumptions (`TPU > CUDA > CPU`) |
| TPU batch guardrails | `es_finetuning/tpu_guardrails.py`; integrated in `scripts/run_es.py`, `scripts/run_eval.py`, `namm/evaluator.py` | Enforces fixed batch constraints and partial-batch padding required by XLA |
| XLA shape hardening | Updates in `namm/policy/base.py`, `namm/policy/deep_selection.py`, `namm/llms/llama.py`, `namm/policy/deep.py` | Prevents dynamic-shape / negative-indexing failures on TPU |
| ES runtime | Vendored `es_finetuning` package and TPU-safe noise handling | Makes ES loop self-contained and TPU-compatible |
| Preemption/reliability | `es_finetuning/preemption.py`, `es_finetuning/trainer.py` resume state logic | Supports spot VM termination + exact resume |
| GCS workflows | `es_finetuning/gcs.py`, run claiming in `scripts/run_es.py`, reporting/archive scripts | Enables cloud manifests + periodic checkpoints |
| Pretrained NAMM checkpoint management | `--namm_checkpoint latest` in `scripts/run_es.py` / `scripts/run_eval.py`, plus `scripts/upload_pretrained.py` | Allows automatic download of latest checkpoint from GCS with local cache |
| TPU setup scripts | `setup/setup_tpu.sh`, `setup/activate_tpu.sh`, `setup/tpu_restart.sh` | Standardizes environment bootstrap and activation |
| Compile warmup | `scripts/warmup_xla_cache.sh` | Precompiles common graphs / catches shape issues early |
| Smoke matrix | `scripts/tpu_smoke_matrix.sh` | One-command train+eval validation across `es_only`, `es_recency`, `es_namm` |
| TPU presets | `cfgs/run/*_tpu.yaml` | Locks known-good fixed batch settings |

### 1.3 New TPU Preset Configs

Added fixed-batch config overlays:

- `cfgs/run/full_cache_es_llama32_1b_tpu.yaml`
- `cfgs/run/recency_es_llama32_1b_tpu.yaml`
- `cfgs/run/namm_bam_i1_llama32_1b_tpu.yaml`

Each forces:

- `batch_size: 18`
- `eval_max_batch_size: 18`

These prevent accidental `"auto"` batching in TPU runs.

---

## 2) Repository Structure and Roles (TPU Additions)

### 2.1 Core Runtime

- `scripts/run_es.py`: Main ES training entrypoint; TPU checks, GCS, resume, optional XLA cache upload, supports `--namm_checkpoint latest`.
- `scripts/run_eval.py`: Full validation-set evaluation for ES checkpoints; supports `--namm_checkpoint latest`.
- `es_finetuning/trainer.py`: ES loop, checkpointing, periodic state save, resume hooks.
- `es_finetuning/tpu_guardrails.py`: TPU-specific invariant checks and partial-batch padding helper.

### 2.2 Policy / XLA Stability Layer

- `namm/policy/base.py`: TPU-safe recency slicing and cache-size guards.
- `namm/policy/deep_selection.py`: validity-mask alignment when KV lengths differ.
- `namm/llms/llama.py`: attention-mask + cache-validity-mask alignment.
- `namm/evaluator.py`: padded final batch behavior in TPU mode.

### 2.3 Operations / Tooling

- `setup/setup_tpu.sh`: full environment bootstrap.
- `setup/activate_tpu.sh`: activates env and exports TPU/GCS vars; optional cache download.
- `scripts/warmup_xla_cache.sh`: compile warmup matrix.
- `scripts/tpu_smoke_matrix.sh`: acceptance smoke matrix runner with machine-readable summary.
- `scripts/upload_pretrained.py`: list/upload pretrained NAMM checkpoints in GCS.

---

## 3) New Machine Setup (Step-by-Step)

Assumes Ubuntu-based TPU VM with internet access.

## 3.1 Clone the Correct Branch

```bash
export BRANCH=<your_tpu_branch_name>   # e.g. tpu or tpuv2
git clone -b "$BRANCH" https://github.com/mathisweil/evo-memory.git
cd evo-memory
```

If repo already exists:

```bash
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH"
```

## 3.2 Run TPU Setup Script

```bash
bash setup/setup_tpu.sh
```

Optional (skip Claude install):

```bash
bash setup/setup_tpu.sh --noclaude
```

If a previous setup attempt installed a broken XLA stack, rerun the script after pulling the latest branch changes:

```bash
bash setup/setup_tpu.sh --noclaude
```

If that shell still imports the wrong wheels, recreate the venv and run setup again:

```bash
rm -rf venv
bash setup/setup_tpu.sh --noclaude
```

The setup script installs:

- Python venv
- pinned TPU runtime pair: `torch==2.9.0` and `torch_xla[tpu]==2.9.0`
- project dependencies
- `google-cloud-storage`
- prompts for Hugging Face and wandb login

The version pin matters. Installing unpinned latest `torch` / `torch_xla` can produce an `_XLAC` undefined-symbol import failure on TPU VMs.

## 3.3 Activate Environment

```bash
source setup/activate_tpu.sh
```

Optional: pull XLA cache from GCS at activation time:

```bash
XLA_CACHE_DOWNLOAD=1 source setup/activate_tpu.sh
```

## 3.4 Quick Environment Sanity Checks

```bash
python3 -c "import torch; print('torch', torch.__version__)"
python3 -c "import torch_xla; print('torch_xla', torch_xla.__version__)"
python3 -c "import torch_xla.core.xla_model as xm; print('device', xm.xla_device())"
python3 -c "from es_finetuning import ESTrainer, ESConfig; print('es_finetuning import OK')"
```

Expected: XLA device like `xla:0`.

## 3.5 Required Runtime Inputs

Recommended (easy path): wrapper scripts auto-resolve a pretrained NAMM checkpoint by using a local cached `.pt` file first, and falling back to GCS only if no local checkpoint exists.

This means `scripts/warmup_xla_cache.sh` and `scripts/tpu_smoke_matrix.sh` do not require a manual `export NAMM_CKPT=...` step anymore unless you want to pin a specific file.

If you want to resolve it explicitly once per shell and reuse the local path across manual commands:

```bash
export NAMM_CKPT="$(python3 scripts/upload_pretrained.py --latest-path)"
test -f "$NAMM_CKPT" && echo "Using NAMM_CKPT=$NAMM_CKPT"
```

Fallback (pin a manual local path):

```bash
export NAMM_CKPT=/abs/path/to/namm_pretrained_romain_v2.pt
test -f "$NAMM_CKPT" && echo "Using NAMM_CKPT=$NAMM_CKPT"
```

Alternative for direct entrypoints (`scripts/run_es.py` and `scripts/run_eval.py`):

- Use `--namm_checkpoint latest` to auto-download newest file from:
  `gs://statistical-nlp/NAMM_checkpoints/pretrained/`
- Downloaded file is cached under `exp_local/pretrained/` and reused if size matches.

List/upload helpers:

```bash
# Resolve a local pretrained checkpoint path (prefer cache, else download latest)
python3 scripts/upload_pretrained.py --latest-path

# List checkpoints in bucket
python3 scripts/upload_pretrained.py --list

# Upload one or more local pretrained checkpoints
python3 scripts/upload_pretrained.py exp_local/pretrained/namm_pretrained_romain_v2.pt
python3 scripts/upload_pretrained.py exp_local/pretrained/*.pt
```

---

## 4) Validate the TPU Implementation

Validation should be done in three layers: static checks, smoke matrix, then stability/reliability.

## 4.1 Static Checks (Fast)

Run from repo root:

```bash
bash -n scripts/tpu_smoke_matrix.sh
bash -n setup/activate_tpu.sh
bash -n scripts/warmup_xla_cache.sh
python3 -m py_compile \
  scripts/run_es.py \
  scripts/run_eval.py \
  scripts/upload_pretrained.py \
  es_finetuning/tpu_guardrails.py \
  es_finetuning/gcs.py
python3 -m unittest tests/test_tpu_guardrails.py tests/test_gcs_pretrained.py
```

## 4.2 Optional Compile Warmup

```bash
source setup/activate_tpu.sh
bash scripts/warmup_xla_cache.sh
```

`scripts/warmup_xla_cache.sh` auto-resolves the latest pretrained NAMM checkpoint unless you already exported `NAMM_CKPT` to pin a specific file.

Default warmup behavior:

- `es_only` full-cache once
- `es_recency` across cache sizes
- `es_namm` across cache sizes

## 4.3 Acceptance Smoke Matrix (Recommended Gate)

```bash
source setup/activate_tpu.sh
bash scripts/tpu_smoke_matrix.sh
```

`scripts/tpu_smoke_matrix.sh` auto-resolves the latest pretrained NAMM checkpoint unless you already exported `NAMM_CKPT` to pin a specific file.

What it runs:

- `es_only` train + eval
- `es_recency` train + eval
- `es_namm` train + eval

Default parameters:

- `NUM_ITERATIONS=2`
- `POP_SIZE=2`
- `BATCH_SIZE=18`
- `CACHE_SIZE=1024`
- `RUN_EVAL=1`
- `GCS_MODE=local` (`--no-gcs`)

Outputs:

- `experiments/smoke_matrix/<timestamp>/summary.tsv`
- `experiments/smoke_matrix/<timestamp>/summary.json`
- per-method logs in same folder

GCS-backed smoke mode:

```bash
GCS_MODE=gcs bash scripts/tpu_smoke_matrix.sh
```

## 4.4 Reliability Validation (Resume/Preemption)

Run a longer GCS-enabled training command and send SIGTERM:

```bash
source setup/activate_tpu.sh

python3 scripts/run_es.py \
  --run_name preempt_test_namm \
  --method es_namm \
  --run_config namm_bam_i1_llama32_1b_tpu \
  --namm_checkpoint latest \
  --num_iterations 50 \
  --population_size 8 \
  --mini_batch_size 18 \
  --batch_size 18 \
  --checkpoint_every 5 \
  --cache_size 1024 \
  --gcs &

PID=$!
sleep 180
kill -TERM "$PID"
wait "$PID" || true
```

Then rerun the same command and verify auto-resume log messages (latest checkpoint detected and resumed).

---

## 5) Running Experiments (Command Cookbook)

Use TPU preset configs for fixed-shape runs.

Shared defaults (recommended):

```bash
source setup/activate_tpu.sh
export FILTER=32768
```

## 5.1 ES-Only (Full Cache)

### Smoke

```bash
python3 scripts/run_es.py \
  --run_name es_only_smoke \
  --method es_only \
  --run_config full_cache_es_llama32_1b_tpu \
  --num_iterations 2 \
  --population_size 2 \
  --mini_batch_size 18 \
  --batch_size 18 \
  --filter_by_length "$FILTER" \
  --checkpoint_every 0 \
  --no-gcs
```

### Main run

```bash
python3 scripts/run_es.py \
  --run_name es_only_i50 \
  --method es_only \
  --run_config full_cache_es_llama32_1b_tpu \
  --num_iterations 50 \
  --population_size 8 \
  --mini_batch_size 18 \
  --batch_size 18 \
  --filter_by_length "$FILTER" \
  --checkpoint_every 10 \
  --gcs
```

## 5.2 ES + Recency

### Smoke

```bash
python3 scripts/run_es.py \
  --run_name es_recency_smoke_c1024 \
  --method es_recency \
  --run_config recency_es_llama32_1b_tpu \
  --cache_size 1024 \
  --num_iterations 2 \
  --population_size 2 \
  --mini_batch_size 18 \
  --batch_size 18 \
  --filter_by_length "$FILTER" \
  --checkpoint_every 0 \
  --no-gcs
```

### Main run

```bash
python3 scripts/run_es.py \
  --run_name es_recency_i50_c1024 \
  --method es_recency \
  --run_config recency_es_llama32_1b_tpu \
  --cache_size 1024 \
  --num_iterations 50 \
  --population_size 8 \
  --mini_batch_size 18 \
  --batch_size 18 \
  --filter_by_length "$FILTER" \
  --checkpoint_every 10 \
  --gcs
```

## 5.3 ES + NAMM (Core)

### Smoke

```bash
python3 scripts/run_es.py \
  --run_name es_namm_smoke_c1024 \
  --method es_namm \
  --run_config namm_bam_i1_llama32_1b_tpu \
  --namm_checkpoint latest \
  --cache_size 1024 \
  --num_iterations 2 \
  --population_size 2 \
  --mini_batch_size 18 \
  --batch_size 18 \
  --filter_by_length "$FILTER" \
  --checkpoint_every 0 \
  --no-gcs
```

### Main run

```bash
python3 scripts/run_es.py \
  --run_name es_namm_i50_c1024 \
  --method es_namm \
  --run_config namm_bam_i1_llama32_1b_tpu \
  --namm_checkpoint latest \
  --cache_size 1024 \
  --num_iterations 50 \
  --population_size 8 \
  --mini_batch_size 18 \
  --batch_size 18 \
  --filter_by_length "$FILTER" \
  --checkpoint_every 10 \
  --gcs
```

### Optional XLA cache upload on exit

Add `--sync-xla-cache` to `run_es.py` commands (requires `--gcs` and TPU env):

```bash
python3 scripts/run_es.py ... --gcs --sync-xla-cache
```

---

## 6) Evaluating Checkpoints

Use `scripts/run_eval.py` on final checkpoint files:

- checkpoint path pattern:
  `experiments/experiment_<N>/<method>/<run_name>/checkpoints/es_checkpoint_final.pt`

## 6.1 Evaluate ES-only checkpoint

```bash
python3 scripts/run_eval.py \
  --es_checkpoint experiments/experiment_N/es_only/es_only_i50/checkpoints/es_checkpoint_final.pt \
  --run_config full_cache_es_llama32_1b_tpu \
  --batch_size 18
```

## 6.2 Evaluate ES-recency checkpoint

```bash
python3 scripts/run_eval.py \
  --es_checkpoint experiments/experiment_N/es_recency/es_recency_i50_c1024/checkpoints/es_checkpoint_final.pt \
  --run_config recency_es_llama32_1b_tpu \
  --cache_size 1024 \
  --batch_size 18
```

## 6.3 Evaluate ES+NAMM checkpoint

```bash
python3 scripts/run_eval.py \
  --es_checkpoint experiments/experiment_N/es_namm/es_namm_i50_c1024/checkpoints/es_checkpoint_final.pt \
  --namm_checkpoint latest \
  --run_config namm_bam_i1_llama32_1b_tpu \
  --cache_size 1024 \
  --batch_size 18
```

---

## 7) Expected Outputs and Where to Look

Training outputs live under:

```text
experiments/experiment_<N>/<method>/<run_name>/
  config.json
  results.json
  examples.json
  checkpoints/es_checkpoint_final.pt
```

Smoke matrix outputs:

```text
experiments/smoke_matrix/<timestamp>/
  summary.tsv
  summary.json
  es_only_train.log
  es_only_eval.log
  es_recency_train.log
  ...
```

Pretrained checkpoint cache (when using `--namm_checkpoint latest`):

```text
exp_local/pretrained/*.pt
```

Useful checks:

```bash
cat experiments/smoke_matrix/<timestamp>/summary.tsv
cat experiments/experiment_<N>/es_namm/<run_name>/results.json
```

---

## 8) Troubleshooting

1. `TPU requires a fixed integer batch size`:
- Use TPU preset configs or pass `--batch_size 18 --mini_batch_size 18`.

2. `NAMM_CKPT` problems in warmup/smoke scripts:
- `scripts/warmup_xla_cache.sh` and `scripts/tpu_smoke_matrix.sh` auto-resolve the latest pretrained checkpoint if `NAMM_CKPT` is unset or set to `latest`.
- To pin a specific file, export `NAMM_CKPT=/abs/path/to/checkpoint.pt` and verify it exists.

3. `No pretrained NAMM checkpoints found in gs://.../NAMM_checkpoints/pretrained/`:
- Upload one first via `python3 scripts/upload_pretrained.py <checkpoint.pt>`, or use a local `NAMM_CKPT` path.

4. `_XLAC ... undefined symbol ...` when importing `torch_xla`:
- Cause: mismatched `torch` / `torch_xla` wheels.
- Fix: rerun `bash setup/setup_tpu.sh --noclaude`.
- If needed, recreate the environment with `rm -rf venv && bash setup/setup_tpu.sh --noclaude`.

5. XLA cache upload does not run:
- `--sync-xla-cache` is opt-in and requires TPU + `--gcs` + `gsutil` + `GCS_BUCKET`.

6. Slow first run:
- Expected due XLA compilation; run warmup and/or smoke first.

7. Resume not happening:
- Resume auto-detection is GCS-backed; ensure same `--run_name` and `--gcs`.

---

## 9) Recommended Execution Order on a Fresh TPU VM

1. Setup + activate environment.
2. Run quick sanity checks (`import torch_xla`, `xm.xla_device()`).
3. Run static checks (`py_compile`, unit test).
4. Optional but useful: run `python3 scripts/upload_pretrained.py --latest-path` once to confirm checkpoint resolution works.
5. Run `scripts/tpu_smoke_matrix.sh` in local mode.
6. Run one 50-iteration `es_namm` job in GCS mode.
7. Run preemption/resume validation.
8. Run full experiment matrix (`es_only`, `es_recency`, `es_namm`) and evaluate all final checkpoints.
