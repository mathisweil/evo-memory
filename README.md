# evo-memory

Fine-tuning LLaMA 3.2-1B-Instruct via evolutionary strategies (ES) or LoRA while [NAMM](https://arxiv.org/abs/2410.13166) manages the KV cache.

---

## Project Structure

```
evo-memory/
├── Makefile                      # build orchestration (setup, logins, TPU lifecycle)
├── pyproject.toml                # Python metadata + all dependencies (single source of truth)
├── requirements.lock             # pinned dependency versions (committed to git)
├── .env.example                  # environment variable template
│
├── config/                       # Hydra configuration (used by run_namm.py)
│   ├── config.yaml               #   main config with composable defaults
│   ├── model/                    #   LLM / evaluator configs
│   ├── policy/                   #   memory eviction policies (BAM, MLP, attention)
│   ├── evolution/                #   CMA-ES and dummy evolution configs
│   ├── task/                     #   dataset / sampler configs
│   ├── run/                      #   experiment-specific run presets
│   ├── trainer/                  #   training and eval trainer configs
│   └── typing/                   #   precision / attention configs
│
├── scripts/                      # main entry points
│   ├── run_es.py                 #   ES fine-tuning
│   ├── run_lora.py               #   LoRA gradient-based fine-tuning
│   ├── run_namm.py               #   NAMM policy training (Hydra)
│   ├── run_joint.py              #   joint alternating NAMM + adapter training
│   ├── run_eval.py               #   evaluation runner
│   ├── generate_report.py        #   cross-experiment comparison reports
│   ├── upload_pretrained.py      #   GCS checkpoint management
│   ├── archive_experiment.py     #   experiment archival to GCS
│   ├── es_default.yaml           #   ES hyperparameter defaults
│   ├── lora_default.yaml         #   LoRA base defaults
│   ├── lora_m1_only.yaml         #   m1 condition (LoRA only, no NAMM)
│   ├── lora_rh_m1_instruct.yaml  #   m1 multi-task variant
│   ├── lora_rh_m4_instruct.yaml  #   m4 condition (LoRA + frozen NAMM)
│   ├── joint_default.yaml        #   joint training defaults
│   └── eval_default.yaml         #   evaluation configuration
│
├── es_finetuning/                # ES optimizer module
│   ├── config.py                 #   ESConfig dataclass
│   ├── trainer.py                #   ESTrainer (main ES loop)
│   ├── device.py                 #   device abstraction (TPU/GPU/CPU)
│   ├── noise.py                  #   weight perturbation logic
│   ├── gcs.py                    #   Google Cloud Storage utilities
│   ├── preemption.py             #   TPU spot VM preemption handling
│   └── utils.py                  #   memory cleanup utilities
│
├── grad_lora_finetuning/         # LoRA gradient-based fine-tuning module
│   ├── trainer.py                #   LoRAGradTrainer
│   └── datasets.py               #   NTP and SFT dataset classes
│
├── namm/                         # Neural Adaptive Memory Management module
│   ├── trainer.py                #   NAMM training loop (DDP support)
│   ├── tasks.py                  #   task definitions (Qasper, LongBench)
│   ├── run_utils.py              #   setup utilities
│   ├── evaluation/               #   evaluator, metrics, LongBench scoring
│   ├── evolution/                #   CMA-ES implementation
│   ├── llms/                     #   LLM wrappers (base + LLaMA)
│   ├── modules/                  #   neural network components for policies
│   └── policy/                   #   eviction policies (recency, deep scoring)
│
├── utils/                        # shared utilities
│   ├── helpers.py                #   model loading, tokenization, data processing
│   └── hydra_helpers.py          #   Hydra configuration utilities
│
├── tests/                        # unit tests
├── data/                         # dataset storage (e.g. LongBench)
└── docs/                         # additional guides (ES, LoRA, NAMM, TPU notes)
```

---

## Prerequisites

| Requirement | Purpose |
|---|---|
| **[uv](https://docs.astral.sh/uv/)** | Fast Rust-based Python package manager (creates venv, installs deps) |
| **make** | Orchestrates hardware-specific setup, logins, and TPU lifecycle |

### `uv` Setup on UCL GPU Machines

Home directories on UCL GPU machines have strict storage limits. Configure `uv` to use the quota directory for its cache.

Install `uv` if not already present:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Add the following to your `~/.cshrc` file to update your path and set the cache directory permanently:

```csh
set path = ( $HOME/.local/bin $path )
setenv UV_CACHE_DIR $QUOTA_DIR/uv_cache
```

Apply the changes to your current session:

```csh
source ~/.cshrc
```

**Note:** `make` is pre-installed on macOS and most Linux distributions. On Ubuntu/Debian, install with `sudo apt install build-essential`.

---

## Setup & Installation

### 1. Clone and configure

```bash
git clone https://github.com/mathisweil/evo-memory.git
cd evo-memory
cp .env.example .env              # then edit .env (see Configuration below)
```

### 2. Run the setup target for your hardware

| Target | When to use |
|---|---|
| `make setup-local` | Laptops, workstations, or any machine with optional CUDA. Installs PyTorch from the default PyPI index. |
| `make setup-gpu GPU=N` | Dedicated CUDA GPU server. Pulls PyTorch CUDA 12.1 wheels. Pass `GPU=N` to pin a device (e.g. `make setup-gpu GPU=2`). |
| `make setup-tpu` | Google Cloud TPU VM. Installs PyTorch + XLA from Google's TPU index, syncs the XLA compilation cache from GCS. |
| `make setup-ucl-gpu` | UCL CSH cluster. Loads the `cuda` environment module, installs CUDA 12.1 wheels, generates `activate.csh` for tcsh. |

```bash
make setup-local              # example: local development
```

Each target creates a `uv`-managed virtualenv (Python 3.10), installs platform-specific PyTorch wheels, then installs all remaining dependencies from `pyproject.toml` via `uv pip install -e "."`. Activation scripts (`activate.sh`, `activate.csh`) are generated automatically.

### 3. Log in to required services

```bash
make hf-login                 # required: gated LLaMA 3.2 access
```

Optional logins (run individually or all at once with `make logins`):

```bash
make wandb-login              # only needed when wandb_log=true
make gcs-auth                 # GCS access (installs gcloud CLI if missing)
make install-claude           # Claude Code CLI (installs Node.js via nvm if needed)
```

### 4. Activate in subsequent shells

```bash
source activate.sh            # bash / zsh — auto-detects TPU vs GPU
source activate.csh           # csh / tcsh (UCL machines)
```

### Other useful targets

```bash
make lock                     # pin dependency versions to requirements.lock
make smoke                    # quick sanity check (ES, 2 iterations, no GCS)
make clean                    # remove venv, caches, stamps, activation scripts
make clean-cache              # remove HF + XLA caches only (keeps venv)
make help                     # list all available targets
```

---

## Configuration

Copy the template and edit:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_MODEL_PATH` | **Yes** | — | HuggingFace model ID or local path (e.g. `meta-llama/Llama-3.2-1B-Instruct`) |
| `HF_CACHE_DIR` | No | `<repo>/.hf_cache` | HuggingFace cache directory |
| `CUDA_VISIBLE_DEVICES` | No | `0` | GPU index (also settable via `make setup-gpu GPU=N`) |
| `NAMM_CKPT` | Eval only | — | Path to a trained NAMM checkpoint |
| `WANDB_API_KEY` | No | — | Only needed when `wandb_log=true` |
| `GCS_BUCKET` | TPU/cloud | `statistical-nlp` | GCS bucket name |
| `GCS_PROJECT` | TPU/cloud | `statistical-nlp` | GCS project ID |

---

## Execution

All scripts accept `--config <yaml>` to load defaults; CLI flags override the config. Pass `--no-gcs` to disable cloud syncing.

### Experiments

| Experiment | Script | Config | Required Args | Key Optional Args |
|---|---|---|---|---|
| **Train NAMM** | `scripts/run_namm.py` | `config/config.yaml` (Hydra) | — | Hydra overrides: `key=value` |
| **ES — no NAMM** | `scripts/run_es.py` | `scripts/es_default.yaml` | `--run_name` | `--num_iterations`, `--population_size`, `--sigma`, `--alpha` |
| **ES + frozen NAMM** | `scripts/run_es.py` | `scripts/es_default.yaml` | `--run_name`, `--namm_checkpoint` | `--cache_size`, `--num_iterations` |
| **LoRA only** (m1) | `scripts/run_lora.py` | `scripts/lora_m1_only.yaml` | `--run_name` | `--num_epochs`, `--learning_rate`, `--lora_rank` |
| **LoRA multi-task** (rh-m1) | `scripts/run_lora.py` | `scripts/lora_rh_m1_instruct.yaml` | `--run_name` | `--num_epochs`, `--eval_interval` |
| **LoRA + frozen NAMM** (rh-m4) | `scripts/run_lora.py` | `scripts/lora_rh_m4_instruct.yaml` | `--run_name`, `--namm_checkpoint` | `--cache_size`, `--eval_interval` |
| **Joint NAMM + ES** | `scripts/run_joint.py` | `scripts/joint_default.yaml` | `--run_name`, `--adapter_type es` | `--num_outer_loops`, `--namm_iterations_per_stage`, `--adapter_iterations_per_stage` |
| **Joint NAMM + LoRA** | `scripts/run_joint.py` | `scripts/joint_default.yaml` | `--run_name`, `--adapter_type lora` | `--num_outer_loops`, `--namm_iterations_per_stage`, `--lora_epochs_per_stage` |
| **Evaluate** | `scripts/run_eval.py` | `scripts/eval_default.yaml` | — | `--es_checkpoint`, `--namm_checkpoint`, `--cache_size`, `--num_samples` |

### Example commands

```bash
# Smoke test (ES, no NAMM) — also: make smoke
python scripts/run_es.py --run_name smoke \
    --num_iterations 2 --population_size 2 --mini_batch_size 2 --no-gcs

# ES + frozen NAMM
python scripts/run_es.py --config scripts/es_default.yaml --run_name es_namm_run \
    --namm_checkpoint exp_local/pretrained/namm.pt --cache_size 1024

# LoRA only (m1)
python scripts/run_lora.py --config scripts/lora_m1_only.yaml --run_name m1_run

# LoRA multi-task (rh-m1)
python scripts/run_lora.py --config scripts/lora_rh_m1_instruct.yaml --run_name rh_m1_run

# LoRA + frozen NAMM (rh-m4)
python scripts/run_lora.py --config scripts/lora_rh_m4_instruct.yaml --run_name rh_m4_run \
    --namm_checkpoint exp_local/pretrained/namm.pt

# Evaluate ES checkpoint
python scripts/run_eval.py \
    --es_checkpoint experiments/experiment_1/es_namm/my_run/checkpoints/es_checkpoint_final.pt \
    --namm_checkpoint exp_local/pretrained/namm.pt

# Evaluate baseline (no fine-tuning, no NAMM)
python scripts/run_eval.py

# Joint NAMM + ES (alternating)
python scripts/run_joint.py --config scripts/joint_default.yaml \
    --run_name joint_es_run --adapter_type es \
    --num_outer_loops 5 --namm_iterations_per_stage 50 \
    --adapter_iterations_per_stage 25

# Joint NAMM + LoRA (alternating)
python scripts/run_joint.py --config scripts/joint_default.yaml \
    --run_name joint_lora_run --adapter_type lora \
    --num_outer_loops 5 --namm_iterations_per_stage 50 \
    --lora_epochs_per_stage 1

# Joint with pre-trained NAMM warm-start
python scripts/run_joint.py --config scripts/joint_default.yaml \
    --run_name joint_warm --adapter_type es \
    --namm_checkpoint exp_local/pretrained/namm.pt

# Joint smoke test
python scripts/run_joint.py --run_name test --adapter_type es \
    --num_outer_loops 2 --namm_iterations_per_stage 3 \
    --adapter_iterations_per_stage 2 --population_size 2 \
    --mini_batch_size 2 --no-gcs
```

---

## Output Layout

```
experiments/
└── experiment_N/
    ├── {es_namm,es_only,es_recency,lora_grad,m1_lora_only,...}/
    │   └── {run_name}/
    │       ├── config.json          # full config snapshot
    │       ├── results.json         # final eval scores (Qasper F1 + per-task)
    │       ├── examples.json        # captured Q/A examples
    │       └── checkpoints/
    │           ├── es_checkpoint_iter_*.pt   # rolling (last 2 kept)
    │           └── es_checkpoint_final.pt
    │
    └── joint_{es,lora}/                     # joint alternating training
        └── {run_name}/
            ├── config.json                  # joint config snapshot
            ├── results.json                 # per-loop eval history
            ├── namm/                        # NAMM checkpoints
            │   ├── latest.pt                #   latest CMA-ES state
            │   └── namm_stage_K.pt          #   per-stage snapshots
            └── adapter/                     # adapter checkpoints
                └── stage_K/                 #   per-stage ES/LoRA output
```

`run_eval.py` writes `eval_{baseline,es}_{timestamp}.log` and `results.json` into the checkpoint's parent directory (or `--output_dir`).

`run_namm.py` (Hydra) writes to `outputs/{date}/{time}/` relative to the repo root.

---

## Dependencies

All dependencies are declared in `pyproject.toml`. Key version pins:

```
torch==2.3.1          # cu121; TPU: torch_xla matching this version
transformers==4.41.2  # CRITICAL: 4.45+ breaks DynamicCache API
peft==0.11.1
numpy<2
```

Requires GLIBC >= 2.28 (Ubuntu 20.04+).

---

## TPU (Google Cloud)

Use `make setup-tpu` for first-time setup (see [Setup & Installation](#setup--installation)).

**Restart a preempted/stopped spot VM:**

```bash
make tpu-restart-v6e          # spot v6e-8 in europe-west4-a
make tpu-restart-v4           # on-demand v4-8 in us-central2-b
```

First XLA run takes ~20 min for compilation. The cache is synced to/from GCS automatically. Spot VM preemption triggers a SIGTERM for emergency checkpoint upload; re-run with the same `--run_name` to resume.

---

## Utility Scripts

| Script | Purpose |
|---|---|
| `scripts/generate_report.py` | Compare results across experiments |
| `scripts/upload_pretrained.py` | Upload / list pretrained NAMM checkpoints in GCS |
| `scripts/archive_experiment.py` | Archive completed experiment dirs to GCS |
