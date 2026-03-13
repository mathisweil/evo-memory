# TPU Version Instructions

This document explains how to set up, validate, and run the TPU-enabled ES fine-tuning workflows on the current TPU branch.

It replaces the older `tpuv2` guidance in one important way:

- the branch now supports a real multichip TPU training backend via `--execution-backend tpu_multichip_exact`
- that backend does not use `xmp.spawn`; it launches one Python worker per local TPU chip and initializes PJRT manually
- the smoke / warmup wrapper scripts are still useful, but they remain single-process validation tools by default

## 1) What Changed Relative To The Older TPU Branch

The current branch adds a second TPU execution mode:

- `single_process`
  - the existing default
  - used by wrapper scripts like `scripts/tpu_smoke_matrix.sh` and `scripts/warmup_xla_cache.sh`
  - best for compile validation, smoke checks, and quick debugging
- `tpu_multichip_exact`
  - opt-in
  - intended for longer training runs on a full TPU host
  - launches one worker per local TPU chip
  - currently validated for single-host TPU VMs only
  - currently requires using all local TPU chips on the host

Other relevant updates:

- `scripts/run_es.py` now contains the manual multichip launcher and hidden worker entrypoint.
- distributed worker logs are written under `<run_dir>/worker_logs/worker_<rank>.log`.
- the TPU NAMM spectrogram path was fixed so XLA no longer keeps the magnitude tensor complex.
- smoke / warmup defaults were reduced to be more realistic for TPU debugging:
  - `FILTER_BY_LENGTH=6500`
  - `RUN_EVAL=0`
  - `python3 -u` / unbuffered output

## 2) Setup On A New TPU VM

### 2.1 Clone The Correct Branch

```bash
export BRANCH=tpu-multi
git clone -b "$BRANCH" https://github.com/mathisweil/evo-memory.git
cd evo-memory
```

If the repo already exists:

```bash
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH"
```

### 2.2 Run TPU Setup

```bash
bash setup/setup_tpu.sh
```

Optional:

```bash
bash setup/setup_tpu.sh --noclaude
```

If the environment is broken and imports the wrong XLA wheels:

```bash
rm -rf venv
bash setup/setup_tpu.sh --noclaude
```

The setup script pins the TPU runtime pair:

- `torch==2.9.0`
- `torch_xla[tpu]==2.9.0`

Do not swap these for unpinned latest wheels on a TPU VM.

### 2.3 Activate The TPU Environment

```bash
source setup/activate_tpu.sh
```

Optional XLA cache pull at activation:

```bash
XLA_CACHE_DOWNLOAD=1 source setup/activate_tpu.sh
```

### 2.4 Quick Sanity Checks

```bash
python3 -c "import torch; print('torch', torch.__version__)"
python3 -c "import torch_xla; print('torch_xla', torch_xla.__version__)"
python3 -c "import torch_xla.core.xla_model as xm; print('device', xm.xla_device())"
python3 -c "from es_finetuning import ESTrainer, ESConfig; print('es_finetuning import OK')"
```

Expected TPU device output:

- `xla:0`

### 2.5 Resolve A Pretrained NAMM Checkpoint

Wrapper scripts auto-resolve this if `NAMM_CKPT` is unset or set to `latest`.

If you want to resolve it explicitly once:

```bash
export NAMM_CKPT="$(python3 scripts/upload_pretrained.py --latest-path)"
test -f "$NAMM_CKPT" && echo "Using NAMM_CKPT=$NAMM_CKPT"
```

For direct entrypoints, `--namm_checkpoint latest` is also supported.

## 3) Validation Gates

### 3.1 Static Checks

```bash
bash -n scripts/tpu_smoke_matrix.sh
bash -n scripts/warmup_xla_cache.sh
bash -n setup/activate_tpu.sh

python3 -m py_compile \
  scripts/run_es.py \
  scripts/run_eval.py \
  scripts/upload_pretrained.py \
  es_finetuning/gcs.py \
  es_finetuning/tpu_guardrails.py \
  namm/policy/deep_embedding_spectogram.py

python3 -m unittest \
  tests/test_es_bucketing.py \
  tests/test_es_population.py \
  tests/test_tpu_guardrails.py \
  tests/test_gcs_pretrained.py
```

### 3.2 Optional Compile Warmup

```bash
source setup/activate_tpu.sh
bash scripts/warmup_xla_cache.sh
```

Current warmup defaults:

- `NUM_ITERATIONS=1`
- `POP_SIZE=2`
- `BATCH_SIZE=18`
- `FILTER_BY_LENGTH=6500`

This script is still single-process and is meant for graph validation, not throughput benchmarking.

### 3.3 Acceptance Smoke Matrix

```bash
source setup/activate_tpu.sh
bash scripts/tpu_smoke_matrix.sh
```

Current smoke defaults:

- `NUM_ITERATIONS=2`
- `POP_SIZE=2`
- `BATCH_SIZE=18`
- `CACHE_SIZE=1024`
- `FILTER_BY_LENGTH=6500`
- `RUN_EVAL=0`
- `GCS_MODE=local`

If you want the extra `run_eval.py` pass too:

```bash
RUN_EVAL=1 bash scripts/tpu_smoke_matrix.sh
```

Important:

- the smoke matrix is still a single-process validation wrapper
- it is intentionally lighter than a real multichip training run

## 4) TPU Execution Model

### 4.1 Single-Process TPU Mode

This is the default when you do not pass `--execution-backend`.

Use it for:

- smoke tests
- compile warmup
- quick debugging
- verifying one method in isolation

### 4.2 Multichip TPU Mode

Use:

```bash
--execution-backend tpu_multichip_exact
```

Behavior:

- one Python worker per local TPU chip
- PJRT multiprocess initialization is done explicitly inside each worker
- rank 0 logs to the main stdout
- nonzero ranks log to `<run_dir>/worker_logs/worker_<rank>.log`

Current constraints:

- single-host TPU VMs only
- use all local TPU chips on the host
- on a `v6e-8` host, use `--worker-count 8`

If you request a different worker count than the local TPU topology expects, `run_es.py` now fails fast instead of trying to run a broken partial topology.

### 4.3 Recommended Pattern For Longer TPU Runs

For long multichip training runs, use:

- `--benchmark-mode train_only`
- separate `scripts/run_eval.py` after training

That avoids paying full validation inside the training entrypoint.

## 5) Command Cookbook

Recommended shared shell defaults:

```bash
source setup/activate_tpu.sh
export FILTER=6500
export WORKERS=8   # v6e-8 host
```

### 5.1 ES-Only

Smoke:

```bash
python3 -u scripts/run_es.py \
  --run_name es_only_smoke \
  --method es_only \
  --run_config full_cache_es_llama32_1b_tpu \
  --num_iterations 2 \
  --population_size 2 \
  --mini_batch_size 18 \
  --batch_size 18 \
  --filter_by_length "$FILTER" \
  --checkpoint_every 0 \
  --skip-full-eval \
  --skip-examples \
  --no-gcs
```

Long multichip training run:

```bash
python3 -u scripts/run_es.py \
  --run_name es_only_i50_w8 \
  --method es_only \
  --run_config full_cache_es_llama32_1b_tpu \
  --execution-backend tpu_multichip_exact \
  --worker-count "$WORKERS" \
  --benchmark-mode train_only \
  --num_iterations 50 \
  --population_size 8 \
  --mini_batch_size 18 \
  --batch_size 18 \
  --filter_by_length "$FILTER" \
  --checkpoint_every 10 \
  --gcs
```

### 5.2 ES + Recency

Smoke:

```bash
python3 -u scripts/run_es.py \
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
  --skip-full-eval \
  --skip-examples \
  --no-gcs
```

Long multichip training run:

```bash
python3 -u scripts/run_es.py \
  --run_name es_recency_i50_c1024_w8 \
  --method es_recency \
  --run_config recency_es_llama32_1b_tpu \
  --cache_size 1024 \
  --execution-backend tpu_multichip_exact \
  --worker-count "$WORKERS" \
  --benchmark-mode train_only \
  --num_iterations 50 \
  --population_size 8 \
  --mini_batch_size 18 \
  --batch_size 18 \
  --filter_by_length "$FILTER" \
  --checkpoint_every 10 \
  --gcs
```

### 5.3 ES + NAMM

Smoke:

```bash
python3 -u scripts/run_es.py \
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
  --skip-full-eval \
  --skip-examples \
  --no-gcs
```

Long multichip training run:

```bash
python3 -u scripts/run_es.py \
  --run_name es_namm_i50_c1024_w8 \
  --method es_namm \
  --run_config namm_bam_i1_llama32_1b_tpu \
  --namm_checkpoint latest \
  --cache_size 1024 \
  --execution-backend tpu_multichip_exact \
  --worker-count "$WORKERS" \
  --benchmark-mode train_only \
  --num_iterations 50 \
  --population_size 8 \
  --mini_batch_size 18 \
  --batch_size 18 \
  --filter_by_length "$FILTER" \
  --checkpoint_every 10 \
  --gcs
```

### 5.4 Optional XLA Cache Upload On Exit

Requires TPU + `--gcs`:

```bash
python3 -u scripts/run_es.py ... --gcs --sync-xla-cache
```

## 6) Monitoring Long Runs

Recommended way to launch a long job:

```bash
tmux new -s tpu_long
source setup/activate_tpu.sh
python3 -u scripts/run_es.py ...
```

Recommended monitoring commands from another shell:

```bash
tmux ls
tmux attach -t tpu_long
```

If the run uses `tpu_multichip_exact`, monitor worker logs too:

```bash
tail -f experiments/experiment_<N>/<method>/<run_name>/worker_logs/worker_1.log
tail -f experiments/experiment_<N>/<method>/<run_name>/worker_logs/worker_7.log
```

Check the live process tree:

```bash
ps -eo pid,ppid,cmd --forest | grep -A20 -B5 '<run_name>'
```

If you just want to see whether all workers are still active:

```bash
pgrep -af '_tpu-worker-payload'
```

Important behavior:

- rank 0 prints the top-level run progress
- nonzero workers write to `worker_logs`
- long periods of silence during compile warmup are expected
- `Failed to deserialize executable: UNIMPLEMENTED` warnings are expected on this runtime and are not fatal

## 7) Evaluating A Checkpoint

Use `scripts/run_eval.py` after training finishes.

Example:

```bash
python3 -u scripts/run_eval.py \
  --es_checkpoint experiments/experiment_N/es_namm/es_namm_i50_c1024_w8/checkpoints/es_checkpoint_final.pt \
  --namm_checkpoint latest \
  --run_config namm_bam_i1_llama32_1b_tpu \
  --cache_size 1024 \
  --batch_size 18
```

## 8) Output Layout

Training output:

```text
experiments/experiment_<N>/<method>/<run_name>/
  config.json
  results.json
  examples.json
  checkpoints/
  worker_logs/              # multichip runs only
    worker_1.log
    worker_2.log
    ...
```

Smoke matrix output:

```text
experiments/smoke_matrix/<timestamp>/
  summary.tsv
  summary.json
  es_only_train.log
  es_recency_train.log
  es_namm_train.log
```

## 9) Troubleshooting

1. `TPU requires a fixed integer batch size`
- Use the TPU preset configs and pass `--batch_size 18 --mini_batch_size 18`.

2. `--worker-count` rejected for multichip mode
- On the current runtime, multichip mode must use all local TPU chips.
- On `v6e-8`, use `--worker-count 8`.

3. `Device or resource busy` / `/dev/vfio/...`
- Cause: stale TPU worker processes are still holding chips.
- Fix:

```bash
pgrep -af '_tpu-worker-payload|spawn_main|scripts/run_es.py'
pkill -f '_tpu-worker-payload'
pkill -f 'spawn_main'
```

4. Nonzero worker failed but rank 0 did not show much
- Check:

```bash
ls experiments/experiment_<N>/<method>/<run_name>/worker_logs
tail -n 100 experiments/experiment_<N>/<method>/<run_name>/worker_logs/worker_<rank>.log
```

5. `Failed to deserialize executable: UNIMPLEMENTED`
- Expected on this TPU runtime.
- It means cross-process executable reuse is not available.

6. First run looks slow or silent
- Expected during compile warmup.
- Use `python3 -u`.
- For multichip runs, inspect both rank 0 stdout and `worker_logs/`.

7. `filter_by_length=32768` makes smoke runs look hung
- Do not use that for smoke by default.
- Use `6500` unless you intentionally need the larger shape envelope.

8. `torch_xla` import failures / `_XLAC` symbol issues
- Recreate the environment with the pinned setup script:

```bash
rm -rf venv
bash setup/setup_tpu.sh --noclaude
```

## 10) Recommended Fresh-VM Order

1. Run `bash setup/setup_tpu.sh --noclaude`
2. Run `source setup/activate_tpu.sh`
3. Run the quick sanity checks
4. Run the static checks
5. Run `bash scripts/tpu_smoke_matrix.sh`
6. Run one longer multichip `es_only` or `es_namm` job with `--execution-backend tpu_multichip_exact`
7. Run `scripts/run_eval.py` separately on the final checkpoint

