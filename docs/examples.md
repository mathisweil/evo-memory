# Experiment Examples

All commands assume you've activated the environment and are in the repo root:

```bash
source scripts/activate.sh   # also done automatically by setup.sh
```

---

## 1. NAMM Only (Train Scoring Network)

### 1a. Smoke test (2 iterations, ~5 min)

```bash
torchrun --standalone --nproc_per_node=1 run_namm_training.py \
    run@_global_=namm_bam_i1_llama32_1b.yaml \
    max_iters=2 \
    pop_size=2
```

### 1b. Full training (200 iterations, ~44h)

```bash
torchrun --standalone --nproc_per_node=1 run_namm_training.py \
    run@_global_=namm_bam_i1_llama32_1b.yaml
```

Default config: `pop_size=8`, `cache_size=1024`, `batch_size=1`, `max_iters=200`.
Checkpoint saved to `experiments/namm_only_runs/.../ckpt.pt`.

### 1b-alt. Training with a different length filter

By default, `filter_by_length=6500` drops samples longer than ~5000 words (6500 / 1.3). `max_position_embeddings` and `max_position_id` are automatically tied to `filter_by_length`. To use a different threshold:

```bash
torchrun --standalone --nproc_per_node=1 run_namm_training.py \
    run@_global_=namm_bam_i1_llama32_1b.yaml \
    filter_by_length=4096
```

At `filter_by_length=6500` (default), this keeps 181/200 Qasper samples (90.5%).

### 1c. Evaluate a trained NAMM checkpoint

```bash
# Single cache size
python run_namm_training.py \
    'run@_global_=namm_bam_eval_llama32_1b.yaml' \
    init_from=experiments/namm_only_runs/.../ckpt.pt \
    cache_size=1024

# Sweep cache sizes
for CS in 128 256 512 1024; do
    python run_namm_training.py \
        'run@_global_=namm_bam_eval_llama32_1b.yaml' \
        init_from=experiments/namm_only_runs/.../ckpt.pt \
        cache_size=$CS
done
```

### 1d. Baselines (no NAMM checkpoint needed)

```bash
# Full cache (upper bound, no eviction)
python run_namm_training.py 'run@_global_=full_cache_baseline_llama32_1b.yaml'

# Recency eviction
python run_namm_training.py 'run@_global_=recency_baseline_llama32_1b.yaml' cache_size=1024
```

---

## 2. ES Fine-Tuning Only (No NAMM)

### 2a. Smoke test (2 iterations, ~2 min)

```bash
python run_es_finetuning.py \
    --num_iterations 2 \
    --population_size 2 \
    --mini_batch_size 2
```

### 2b. Full run (150 iterations, ~7.5h)

```bash
python run_es_finetuning.py \
    --num_iterations 150 \
    --population_size 8 \
    --mini_batch_size 4 \
    --sigma 0.001 \
    --alpha 0.0005 \
    --noise_mode correlated \
    --log_dir experiments/es_only_runs/no_namm
```

### 2b-alt. With dataset length filtering

```bash
python run_es_finetuning.py \
    --num_iterations 150 \
    --population_size 8 \
    --mini_batch_size 4 \
    --filter_by_length 6500 \
    --log_dir experiments/es_only_runs/no_namm_filtered
```

### 2c. Evaluate the ES-fine-tuned model

```bash
CKPT=experiments/es_only_runs/no_namm/checkpoints/es_checkpoint_final.pt

# Full cache
python run_namm_training.py 'run@_global_=full_cache_baseline_llama32_1b.yaml' init_from=$CKPT

# NAMM eviction (requires a NAMM checkpoint)
python run_namm_training.py \
    'run@_global_=namm_bam_eval_llama32_1b.yaml' \
    init_from=$CKPT \
    cache_size=1024

# Recency eviction
python run_namm_training.py 'run@_global_=recency_baseline_llama32_1b.yaml' init_from=$CKPT cache_size=1024
```

---

## 3. ES Fine-Tuning With NAMM (Core Experiment)

### 3a. Smoke test (2 iterations, ~2 min)

```bash
python run_es_finetuning.py \
    --namm_checkpoint exp_local/pretrained/namm_pretrained_romain.pt \
    --num_iterations 2 \
    --population_size 2 \
    --mini_batch_size 2
```

### 3b. Full run (150 iterations, ~8-10h)

```bash
python run_es_finetuning.py \
    --namm_checkpoint exp_local/pretrained/namm_pretrained_romain.pt \
    --num_iterations 150 \
    --population_size 8 \
    --mini_batch_size 4 \
    --sigma 0.001 \
    --alpha 0.0005 \
    --noise_mode correlated \
    --log_dir experiments/es_namm_runs/with_namm
```

The frozen NAMM scoring network runs inside every forward pass, scoring and evicting KV-cache tokens at every 256-token boundary.

### 3c. Evaluate the combined model

```bash
CKPT=experiments/es_namm_runs/with_namm/checkpoints/es_checkpoint_final.pt
NAMM_CKPT=exp_local/pretrained/namm_pretrained_romain.pt

# Under NAMM (same policy used during training)
python run_eval.py \
    --es_checkpoint $CKPT \
    --namm_checkpoint $NAMM_CKPT

# Under full cache (does fine-tuning under NAMM hurt full-cache perf?)
python run_namm_training.py 'run@_global_=full_cache_baseline_llama32_1b.yaml' init_from=$CKPT

# Under recency (does it generalise to a different eviction policy?)
python run_namm_training.py 'run@_global_=recency_baseline_llama32_1b.yaml' init_from=$CKPT cache_size=1024
```

---

## 4. Full Evaluation (run_eval.py)

`run_eval.py` evaluates an ES-fine-tuned checkpoint on the full Qasper validation set (not just a mini-batch). Use this to get final numbers for the results table. Logs are automatically saved alongside the checkpoint.

### 4a. ES checkpoint only (no NAMM)

```bash
python run_eval.py \
    --es_checkpoint experiments/es_only_runs/no_namm/checkpoints/es_checkpoint_final.pt
```

### 4b. ES checkpoint with NAMM eviction active

```bash
python run_eval.py \
    --es_checkpoint experiments/es_namm_runs/with_namm/checkpoints/es_checkpoint_final.pt \
    --namm_checkpoint exp_local/pretrained/namm_pretrained_romain.pt
```

### 4c. With a different batch size (default is 4)

```bash
python run_eval.py \
    --es_checkpoint experiments/es_namm_runs/with_namm/checkpoints/es_checkpoint_final.pt \
    --namm_checkpoint exp_local/pretrained/namm_pretrained_romain.pt \
    --eval_batch_size 8
```

### 4d. Evaluate intermediate checkpoints (policy staleness check)

```bash
NAMM_CKPT=exp_local/pretrained/namm_pretrained_romain.pt

for ITER in 25 50 75 100 125 150; do
    echo "=== Iteration $ITER ==="
    python run_eval.py \
        --es_checkpoint experiments/es_namm_runs/with_namm/checkpoints/es_checkpoint_iter${ITER}.pt \
        --namm_checkpoint $NAMM_CKPT
done
```

If F1 degrades at later checkpoints, the frozen NAMM policy is going stale as LLM weights drift.

---

## Monitoring

```bash
# TensorBoard for ES runs
tensorboard --logdir experiments/es_namm_runs/
tensorboard --logdir experiments/es_only_runs/

# wandb for NAMM training/eval
# Results logged to project=memory_evolution_hf
```

---

## Tips

- Use `tmux` for long runs: `tmux new -s train`, then detach with `Ctrl+b d`.
- ES checkpoints are saved every 25 iterations by default (`--checkpoint_every`).
- Validation runs every 25 iterations by default (`--eval_every`).
- To resume, point `--log_dir` at the existing run directory.
- All NAMM configs live in `cfgs/run/*_llama32_1b.yaml`.
