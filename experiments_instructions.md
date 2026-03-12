# Experiments Instructions (es-fine-tuning branch)

This guide is a cleaned, code-aligned version of the current setup/experiment workflow in this repo.

## 1. Repo Structure (High-Level)

Core entry scripts:

- `run_namm_training.py`: NAMM training/evaluation pipeline (Hydra config driven).
- `run_es_finetuning.py`: ES fine-tuning of base LLM weights; can optionally run with frozen NAMM eviction.
- `run_eval.py`: evaluate base or ES-fine-tuned checkpoints on full validation set for the selected `run_config`.

Important folders:

- `cfgs/`: Hydra config graph (model, task, policy, evolution, run presets).
- `scripts/`: environment setup/activation and helper run scripts.
- `namm/`: core implementation (trainer, evaluator, policy, evolution, model wrappers).
- `docs/`: existing notes/examples (useful but partially redundant/inconsistent).
- `data/longbench/`: prompt/metadata files for LongBench tasks.

## 2. How Experiments Are Wired

`run_namm_training.py` always composes `cfgs/config.yaml` plus a run override:

- Example: `run@_global_=namm_bam_i1_llama32_1b.yaml`.
- That run file then selects model/task/policy/evolution presets.

Main run presets in this branch:

- `cfgs/run/namm_bam_i1_llama32_1b.yaml`:
  - Stage-1 NAMM training via CMA-ES.
  - Task: `lb/qasper`.
  - Defaults: `pop_size=8`, `samples_batch_size=16`, `max_iters=200`, `cache_size=1024`.
- `cfgs/run/namm_bam_eval_llama32_1b.yaml`:
  - Eval-only NAMM checkpoint comparison on 3 tasks:
  - `lb/passage_retrieval_en`, `lb/qasper`, `lb/narrativeqa`.
  - Requires `init_from=<NAMM ckpt.pt>`.
- `cfgs/run/full_cache_baseline_llama32_1b.yaml`:
  - Eval-only baseline with no eviction (`cache_size=max_position_id`).
- `cfgs/run/recency_baseline_llama32_1b.yaml`:
  - Eval-only baseline with recency eviction.

## 3. New Machine Setup (Upstream GitHub Repo)

This section is specifically for running `github.com/mathisweil/evo-memory` on a fresh machine.

### 3.1 Prerequisites

- Linux machine with NVIDIA GPU and CUDA drivers.
- `git`, `python3` (3.9+), `pip`, and internet access.
- HuggingFace account with Llama 3.2 model access.
- wandb account/API key.

### 3.2 Recommended Setup Command (use this first)

```bash
curl -fsSL https://raw.githubusercontent.com/mathisweil/evo-memory/es-fine-tuning/scripts/setup_cmd.sh -o /tmp/setup_cmd.sh
bash /tmp/setup_cmd.sh --dir ~/ft-namm
```

Why this is the right default on a new machine:

- it clones the correct upstream repo/branch into `~/ft-namm/evo-memory`
- it clones `~/ft-namm/es-fine-tuning-paper`
- it creates `~/ft-namm/venv` and installs all dependencies
- it prompts for HuggingFace + wandb login
- it sets up shared cache under `~/ft-namm/.hf_cache`

Optional flags you can add:

- `--gpu 0` to pin GPU.
- `--user <name>` if your environment needs explicit user path logic.
- `--noclaude` to skip Claude Code install.

### 3.3 Enter Repo + Verify Branch + Activate

```bash
cd ~/ft-namm/evo-memory
git branch --show-current
# expected: es-fine-tuning
source scripts/activate.sh
```

### 3.4 Alternative (if you already cloned upstream manually)

```bash
mkdir -p ~/ft-namm
cd ~/ft-namm
git clone -b es-fine-tuning https://github.com/mathisweil/evo-memory.git
cd evo-memory
bash scripts/setup.sh --dir ~/ft-namm
source scripts/activate.sh
```

### 3.5 Critical Model Path Check

`cfgs/model/wrapped_llm/llama32-1b.yaml` hardcodes:

```yaml
pretrained_llm_name: .hf_cache/models--meta-llama--Llama-3.2-1B/snapshots/4e20de...
```

If that exact snapshot path does not exist, runs will fail before model load.

Fix once by editing `pretrained_llm_name` to your local downloaded LLaMA 3.2-1B directory (must contain `config.json`), e.g.:

```yaml
pretrained_llm_name: .hf_cache/llama32_1b_local
```

Then place model files there (gated model access required).

## 4. Fast Sanity Checks

From `~/ft-namm/evo-memory` after `source scripts/activate.sh`:

```bash
# 1) Verify Python imports from both repos
python -c "from es_finetuning import ESTrainer, ESConfig; import namm; print('ok')"

# 2) Short NAMM smoke run
torchrun --standalone --nproc_per_node=1 run_namm_training.py \
  run@_global_=namm_bam_i1_llama32_1b.yaml \
  max_iters=2 \
  pop_size=2 \
  samples_batch_size=2

# 3) Short ES smoke run
python run_es_finetuning.py \
  --num_iterations 2 \
  --population_size 2 \
  --mini_batch_size 2
```

## 5. Experiment Workflows

### A) Train NAMM (Stage 1)

Command:

```bash
torchrun --standalone --nproc_per_node=1 run_namm_training.py \
  run@_global_=namm_bam_i1_llama32_1b.yaml
```

What this does:

- Freezes base LLaMA weights.
- Optimizes NAMM scoring-policy parameters with CMA-ES.
- Fitness is task performance (`lb/qasper`) under cache budget constraints.

Outputs:

- Main checkpoint: `ckpt.pt` (contains `evolution_state` with `best_member`, CMA state, stored buffers).
- Also writes `latest.pt`, optional numbered checkpoints, and eval json logs.
- Run directory follows Hydra out path:
  - `experiments/<wandb_project>/<wandb_group_name>/<wandb_run_name>/<seed>/`

### B) Evaluate NAMM checkpoint + baselines (Stage 1 evaluation)

Set an absolute checkpoint path:

```bash
NAMM_CKPT=/absolute/path/to/ckpt.pt
```

NAMM checkpoint evaluation (3-task subset):

```bash
for CS in 256 512 1024; do
  torchrun --standalone --nproc_per_node=1 run_namm_training.py \
    run@_global_=namm_bam_eval_llama32_1b.yaml \
    init_from="$NAMM_CKPT" \
    cache_size="$CS"
done
```

Recency baseline:

```bash
for CS in 256 512 1024; do
  torchrun --standalone --nproc_per_node=1 run_namm_training.py \
    run@_global_=recency_baseline_llama32_1b.yaml \
    cache_size="$CS"
done
```

Full-cache baseline:

```bash
torchrun --standalone --nproc_per_node=1 run_namm_training.py \
  run@_global_=full_cache_baseline_llama32_1b.yaml
```

### C) ES fine-tuning of base LLM weights (Stage 2)

No NAMM (control):

```bash
python run_es_finetuning.py \
  --num_iterations 150 \
  --population_size 8 \
  --mini_batch_size 4 \
  --sigma 0.001 \
  --alpha 0.0005 \
  --noise_mode correlated \
  --log_dir experiments/es_runs/no_namm
```

With frozen NAMM policy:

```bash
python run_es_finetuning.py \
  --namm_checkpoint /absolute/path/to/ckpt.pt \
  --num_iterations 150 \
  --population_size 8 \
  --mini_batch_size 4 \
  --sigma 0.001 \
  --alpha 0.0005 \
  --noise_mode correlated \
  --log_dir experiments/es_runs/with_namm
```

What this does:

- Uses ES perturbations/updates on base model parameters only.
- If `--namm_checkpoint` is set, loads NAMM `best_member` + buffers and uses fixed NAMM eviction during scoring.
- Reward comes from `TaskSampler.evaluate(...)` on `lb/qasper` for the selected run config.

Notes from code:

- `--run_config` defaults to `namm_bam_i1_llama32_1b` (Qasper setup).
- `--train_samples` currently goes into metadata only; it is not used to slice the dataset in `run_es_finetuning.py`.

### D) Evaluate ES checkpoints (recommended via `run_eval.py`)

Set paths:

```bash
ES_CKPT=/absolute/path/to/es_checkpoint_final.pt
NAMM_CKPT=/absolute/path/to/ckpt.pt
```

1) ES checkpoint under NAMM policy:

```bash
python run_eval.py \
  --run_config namm_bam_eval_llama32_1b \
  --es_checkpoint "$ES_CKPT" \
  --namm_checkpoint "$NAMM_CKPT" \
  --eval_batch_size 4
```

2) ES checkpoint under recency:

```bash
python run_eval.py \
  --run_config recency_baseline_llama32_1b \
  --es_checkpoint "$ES_CKPT" \
  --eval_batch_size 4
```

3) ES checkpoint under full-cache:

```bash
python run_eval.py \
  --run_config full_cache_baseline_llama32_1b \
  --es_checkpoint "$ES_CKPT" \
  --eval_batch_size 4
```

4) Baseline (no ES checkpoint; base weights):

```bash
python run_eval.py \
  --run_config namm_bam_eval_llama32_1b \
  --namm_checkpoint "$NAMM_CKPT"
```

Important checkpoint compatibility:

- `init_from=` in `run_namm_training.py` expects NAMM/CMA checkpoint format (`evolution_state`).
- ES checkpoints should be evaluated with `run_eval.py` (`--es_checkpoint`), not via `init_from`.

## 6. Suggested End-to-End Order

1. Setup + activation + model path fix.
2. NAMM smoke test.
3. Full NAMM training (Stage 1).
4. NAMM + recency + full-cache evaluation on 3-task subset.
5. ES run without NAMM (control).
6. ES run with frozen NAMM.
7. Evaluate ES checkpoints under all three eviction settings via `run_eval.py`.

## 7. Useful Helpers Already in Repo

- `scripts/run_namm_experiment.sh`: runs Stage-1 NAMM training.
- `scripts/run_eval_comparison.sh`: sweep template for eval runs (edit `NAMM_CKPT`; currently only NAMM loop is active, recency/full sections are commented out).
- `docs/tmux_cmd.md`: keep long jobs alive.

## 8. Common Failure Modes

- `ModuleNotFoundError: es_finetuning`: `es-fine-tuning-paper` not cloned/installed in expected workspace layout.
- `config.json` not found for LLaMA path: update `pretrained_llm_name` in `cfgs/model/wrapped_llm/llama32-1b.yaml`.
- HuggingFace 401/403: token missing or model access not granted.
- CUDA OOM: lower `batch_size`/`eval_batch_size`, reduce `pop_size` or sample counts.

## 9. Expanded Load/Write Map By Script

### 9.1 `run_namm_training.py`

Load behavior:

- Loads base LLaMA from `pretrained_llm_name` in the active Hydra run config (`cfgs/model/wrapped_llm/llama32-1b.yaml` for the 1B setup).
- NAMM checkpoint loading precedence in trainer startup:
  - if `<out_dir>/latest.pt` exists and `scratch=false`, it resumes from `latest.pt`
  - otherwise, if `init_from=/path/to/ckpt.pt` is provided, it loads that file
  - otherwise it starts from fresh initialization

Write behavior:

- Output root comes from `cfgs/config.yaml`:
  - `out_folder=./experiments`
  - `out_dir=${out_folder}/${wandb_project}/${wandb_group_name}/${wandb_run_name}/${seed}`
- Files written in `out_dir`:
  - `ckpt.pt`: best-validation NAMM/CMA checkpoint
  - `latest.pt`: rolling latest checkpoint for resume
  - `iter_<N>.pt`: periodic snapshot if `keep_past_epoch_checkpoints_every` is enabled
  - `eval_<N>.json`: stored evaluation logs when local eval storage is enabled
  - `rng_ckpt.pt`, `rng_latest.pt`, `rng_iter_<N>.pt`: RNG state companions

Checkpoint content notes:

- NAMM checkpoints (`ckpt.pt`, `latest.pt`, `iter_<N>.pt`) contain an `evolution_state` dict, including CMA state and NAMM policy parameters (for example `best_member`).
- These files are the ones expected by `init_from=` and by `--namm_checkpoint` in the other scripts.

### 9.2 `run_es_finetuning.py`

Load behavior:

- Loads base LLaMA through Hydra compose (`run@_global_=...`) using the same model config system as `run_namm_training.py`.
- If `--namm_checkpoint` is given:
  - reads `evolution_state['best_member']` from that NAMM checkpoint
  - applies it to memory-policy params
  - restores stored normalization buffers from `evolution_state` keys prefixed with `stored_buffers_to_save.`

Write behavior:

- Writes ES outputs under `--log_dir` (default `experiments/es_runs`).
- ES checkpoints are written under a `checkpoints/` subdirectory in that log dir by the ES trainer library.
- These ES checkpoint files store fine-tuned base-model parameter tensors (not NAMM CMA state).

Important compatibility:

- ES checkpoints are not compatible with `init_from=` in `run_namm_training.py`.
- Use ES checkpoints with `run_eval.py --es_checkpoint ...`.

### 9.3 `run_eval.py`

Load behavior:

- Loads base LLaMA from the selected Hydra run config (`--run_config`).
- Optionally loads NAMM policy weights from `--namm_checkpoint` (same NAMM `evolution_state` format as above).
- Optionally loads ES-tuned base weights from `--es_checkpoint` by copying tensors into matching model parameter names.

Write behavior:

- This script is evaluation-only; it does not save NAMM training checkpoints (`ckpt.pt` / `latest.pt` / `iter_<N>.pt`).
- Primary output is printed metrics in stdout (for example `lb/qasper` and other task scores for the chosen run config).

### 9.4 Quick Path Checks

From `~/ft-namm/evo-memory`, useful commands:

```bash
# Find NAMM checkpoints written by run_namm_training.py
find experiments -type f \( -name "ckpt.pt" -o -name "latest.pt" -o -name "iter_*.pt" \) | sort

# Find ES checkpoints written by run_es_finetuning.py
find experiments/es_runs -type f -name "es_checkpoint*.pt" | sort

# Show active LLaMA source path from config
grep -n "pretrained_llm_name" cfgs/model/wrapped_llm/llama32-1b.yaml
```
