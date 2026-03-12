# Experiment Examples

All commands assume you've activated the environment and are in the repo root:

```bash
source setup/activate_tpu.sh   # TPU VM
# or: source setup/activate.sh  # GPU
```

---

## 1. NAMM Only (Train Scoring Network)

### 1a. Smoke test (2 iterations, ~5 min)

```bash
torchrun --standalone --nproc_per_node=1 scripts/run_namm.py \
    run@_global_=namm_bam_i1_llama32_1b.yaml \
    max_iters=2 \
    pop_size=2
```

### 1b. Full training (200 iterations, ~44h)

```bash
torchrun --standalone --nproc_per_node=1 scripts/run_namm.py \
    run@_global_=namm_bam_i1_llama32_1b.yaml
```

Default config: `pop_size=8`, `cache_size=1024`, `batch_size=1`, `max_iters=200`.
Checkpoint saved to `experiments/namm_only_runs/.../ckpt.pt`.

### 1b-alt. Training with a different length filter

By default, `filter_by_length=6500` drops samples longer than ~5000 words (6500 / 1.3). `max_position_embeddings` and `max_position_id` are automatically tied to `filter_by_length`. To use a different threshold:

```bash
torchrun --standalone --nproc_per_node=1 scripts/run_namm.py \
    run@_global_=namm_bam_i1_llama32_1b.yaml \
    filter_by_length=4096
```

At `filter_by_length=6500` (default), this keeps 181/200 Qasper samples (90.5%).

### 1c. Evaluate a trained NAMM checkpoint

```bash
# Single cache size
python scripts/run_namm.py \
    'run@_global_=namm_bam_eval_llama32_1b.yaml' \
    init_from=experiments/namm_only_runs/.../ckpt.pt \
    cache_size=1024

# Sweep cache sizes
for CS in 128 256 512 1024; do
    python scripts/run_namm.py \
        'run@_global_=namm_bam_eval_llama32_1b.yaml' \
        init_from=experiments/namm_only_runs/.../ckpt.pt \
        cache_size=$CS
done
```

### 1d. Baselines (no NAMM checkpoint needed)

```bash
# Full cache (upper bound, no eviction)
python scripts/run_namm.py 'run@_global_=full_cache_baseline_llama32_1b.yaml'

# Recency eviction
python scripts/run_namm.py 'run@_global_=recency_baseline_llama32_1b.yaml' cache_size=1024
```

---

## 2. ES Fine-Tuning Only (No NAMM)

### 2a. Smoke test (2 iterations, ~2 min)

```bash
python scripts/run_es.py \
    --run_name smoke_test \
    --num_iterations 2 \
    --population_size 2 \
    --mini_batch_size 2
```

### 2b. Full run (50 iterations)

```bash
python scripts/run_es.py \
    --run_name full_no_namm \
    --num_iterations 50 \
    --population_size 8 \
    --mini_batch_size 16 \
    --sigma 0.001 \
    --alpha 0.0005 \
    --noise_mode correlated
```

Results saved to `experiments/experiment_N/es_only/full_no_namm/` and `gs://statistical-nlp/experiments/` (GCS enabled by default).

### 2b-alt. With dataset length filtering

```bash
python scripts/run_es.py \
    --run_name no_namm_filtered \
    --num_iterations 50 \
    --population_size 8 \
    --mini_batch_size 16 \
    --filter_by_length 6500
```

### 2c. Evaluate the ES-fine-tuned model

```bash
CKPT=experiments/experiment_N/es_only/full_no_namm/checkpoints/es_checkpoint_final.pt

# Full cache
python scripts/run_namm.py 'run@_global_=full_cache_baseline_llama32_1b.yaml' init_from=$CKPT

# NAMM eviction (requires a NAMM checkpoint)
python scripts/run_namm.py \
    'run@_global_=namm_bam_eval_llama32_1b.yaml' \
    init_from=$CKPT \
    cache_size=1024

# Recency eviction
python scripts/run_namm.py 'run@_global_=recency_baseline_llama32_1b.yaml' init_from=$CKPT cache_size=1024
```

---

## 3. ES Fine-Tuning With NAMM (Core Experiment)

### 3a. Smoke test (2 iterations, ~2 min)

```bash
python scripts/run_es.py \
    --run_name smoke_test_namm \
    --namm_checkpoint exp_local/pretrained/namm_pretrained_romain.pt \
    --num_iterations 2 \
    --population_size 2 \
    --mini_batch_size 2
```

### 3b. Full run (50 iterations)

```bash
python scripts/run_es.py \
    --run_name full_with_namm \
    --namm_checkpoint exp_local/pretrained/namm_pretrained_romain.pt \
    --num_iterations 50 \
    --population_size 8 \
    --mini_batch_size 16 \
    --sigma 0.001 \
    --alpha 0.0005 \
    --noise_mode correlated
```

The frozen NAMM scoring network runs inside every forward pass, scoring and evicting KV-cache tokens at every 256-token boundary.

Results saved to `experiments/experiment_N/es_namm/full_with_namm/`.

### 3c. Evaluate the combined model

```bash
CKPT=experiments/experiment_N/es_namm/full_with_namm/checkpoints/es_checkpoint_final.pt
NAMM_CKPT=exp_local/pretrained/namm_pretrained_romain.pt

# Under NAMM (same policy used during training)
python scripts/run_eval.py \
    --es_checkpoint $CKPT \
    --namm_checkpoint $NAMM_CKPT

# Under full cache (does fine-tuning under NAMM hurt full-cache perf?)
python scripts/run_namm.py 'run@_global_=full_cache_baseline_llama32_1b.yaml' init_from=$CKPT

# Under recency (does it generalise to a different eviction policy?)
python scripts/run_namm.py 'run@_global_=recency_baseline_llama32_1b.yaml' init_from=$CKPT cache_size=1024
```

---

## 4. Full Evaluation (scripts/run_eval.py)

`scripts/run_eval.py` evaluates an ES-fine-tuned checkpoint on the full Qasper validation set (not just a mini-batch). Use this to get final numbers for the results table. Logs are automatically saved alongside the checkpoint.

### 4a. ES checkpoint only (no NAMM)

```bash
python scripts/run_eval.py \
    --es_checkpoint experiments/experiment_N/es_only/run_name/checkpoints/es_checkpoint_final.pt
```

### 4b. ES checkpoint with NAMM eviction active

```bash
python scripts/run_eval.py \
    --es_checkpoint experiments/experiment_N/es_namm/run_name/checkpoints/es_checkpoint_final.pt \
    --namm_checkpoint exp_local/pretrained/namm_pretrained_romain.pt
```

### 4c. With a different batch size

```bash
python scripts/run_eval.py \
    --es_checkpoint experiments/experiment_N/es_namm/run_name/checkpoints/es_checkpoint_final.pt \
    --namm_checkpoint exp_local/pretrained/namm_pretrained_romain.pt \
    --batch_size 8
```

---

## 5. Utility Scripts

### 5a. Generate a report from experiment results

```bash
python scripts/generate_report.py
```

### 5b. Archive a completed experiment

```bash
python scripts/archive_experiment.py
```

### 5c. Run the TPU smoke matrix

```bash
export NAMM_CKPT=/abs/path/to/namm_pretrained_romain_v2.pt
bash scripts/tpu_smoke_matrix.sh

# Optional: use GCS-backed runs
GCS_MODE=gcs bash scripts/tpu_smoke_matrix.sh
```

---

## Monitoring

```bash
# Check experiment results
cat experiments/experiment_N/es_namm/run_name/results.json
cat experiments/experiment_N/es_only/run_name/results.json

# View captured Q/A examples
cat experiments/experiment_N/es_namm/run_name/examples.json

# wandb for NAMM training/eval
# Results logged to project=memory_evolution_hf
```

---

## Tips

- Use `tmux` for long runs: `tmux new -s train`, then detach with `Ctrl+b d`.
- GCS checkpointing is on by default (`--gcs`). Checkpoints sync to `gs://statistical-nlp/experiments/` every `--checkpoint_every` iterations (default 10). Disable with `--no-gcs`.
- Auto-resume: if training is interrupted, re-run with the same `--run_name` and it will resume from the latest GCS checkpoint automatically.
- Preemption-safe: SIGTERM handler triggers an immediate checkpoint upload before exit.
- XLA cache sync is opt-in:
  - upload on run exit: pass `--sync-xla-cache` to `scripts/run_es.py` (with `--gcs`)
  - startup download: `XLA_CACHE_DOWNLOAD=1 source setup/activate_tpu.sh`
- Baseline and final full evaluations are run automatically; no periodic validation during training.
- Q/A examples are captured during the final evaluation (controlled by `--n_examples`).
- All NAMM configs live in `cfgs/run/*_llama32_1b.yaml`.
