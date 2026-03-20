# evo-memory

Fine-tuning LLaMA 3.2-1B-Instruct via evolutionary strategies (ES) or LoRA while [NAMM](https://arxiv.org/abs/2410.13166) manages the KV cache.

---

## Getting Started

### Prerequisites

**Python 3.10+** and **[uv](https://docs.astral.sh/uv/)** (a fast Rust-based Python package manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install

```bash
git clone https://github.com/mathisweil/evo-memory.git
cd evo-memory
cp .env.example .env          # edit: LLM_MODEL_PATH, HF_CACHE_DIR, CUDA_VISIBLE_DEVICES
make setup-local              # or the target that matches your hardware
make hf-login                 # one-time: required for gated LLaMA 3.2
```

Pick the `make` target that matches your hardware:

| Target | When to use |
|---|---|
| `make setup-local` | Laptops, workstations, or any machine with optional CUDA. Installs PyTorch from the default PyPI index. |
| `make setup-gpu` | Dedicated CUDA GPU server. Pulls PyTorch wheels built for CUDA 12.1. Pass `GPU=N` to pin a device (e.g. `make setup-gpu GPU=2`). |
| `make setup-tpu` | Google Cloud TPU VM. Installs PyTorch + XLA from Google's TPU index, syncs the XLA compilation cache from GCS. |
| `make setup-ucl-gpu` | UCL CSH cluster. Loads the `cuda` environment module, installs CUDA 12.1 wheels, generates `activate.csh` for tcsh. Skips GCS and wandb. |

Run `make help` to see every available target.

### Activate (subsequent shells)

```bash
source activate.sh            # bash / zsh — auto-detects TPU vs GPU
source activate.csh           # csh / tcsh (UCL machines)
```

### Optional service logins

Run these individually, or all at once with `make logins`:

| Target | Service | Notes |
|---|---|---|
| `make hf-login` | HuggingFace | Required for gated LLaMA 3.2 access. |
| `make wandb-login` | Weights & Biases | Only needed when `wandb_log=true`. |
| `make gcs-auth` | Google Cloud Storage | Installs `gcloud` CLI if missing, then authenticates. |
| `make install-claude` | Claude Code | Installs Node.js via nvm if needed. |
| `make logins` | All three | Runs `hf-login` + `wandb-login` + `gcs-auth`. |

### Project structure (setup files)

```
evo-memory/
├── pyproject.toml        # Python metadata + all dependencies (single source of truth)
├── requirements.lock      # pinned dependency versions (committed to git)
├── Makefile              # orchestration: hardware-specific torch, logins, TPU lifecycle
├── activate.sh           # generated — bash/zsh activation
└── activate.csh          # generated — csh/tcsh activation (UCL)
```

`pyproject.toml` declares every dependency. PyTorch is listed under optional extras (`[gpu]`, `[tpu]`, `[cpu]`) because it requires a different index URL per hardware platform. The Makefile handles that routing, then installs the rest of the project via `uv pip install -e "."`.

### csh / tcsh — UCL GPU machines

> `activate.sh` is bash-only and will error in csh. Use `activate.csh` exclusively on UCL machines.

```csh
git clone https://github.com/mathisweil/evo-memory.git
cd evo-memory
make setup-ucl-gpu
# Add to ~/.cshrc (persists across shells):
setenv LLM_MODEL_PATH  meta-llama/Llama-3.2-1B-Instruct
setenv HF_CACHE_DIR    /path/to/hf/cache
setenv CUDA_VISIBLE_DEVICES 0
source activate.csh
huggingface-cli login
```

### Reproducible installs (CI or new collaborators)

```bash
make lock                # pin dependency versions
git add requirements.lock  # commit so others get identical resolution
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_MODEL_PATH` | Yes | — | HuggingFace model ID or local path to LLaMA 3.2-1B-Instruct |
| `HF_CACHE_DIR` | No | `<repo>/.hf_cache` | HuggingFace cache directory |
| `CUDA_VISIBLE_DEVICES` | No | `0` | GPU index |
| `NAMM_CKPT` | Eval only | — | Path to a trained NAMM checkpoint |
| `WANDB_API_KEY` | Optional | — | Only needed when `wandb_log=true` |
| `GCS_BUCKET` | TPU/cloud | `statistical-nlp` | GCS bucket name |
| `GCS_PROJECT` | TPU/cloud | `statistical-nlp` | GCS project ID |

---

## Experiments

All scripts accept `--config <yaml>` to load defaults from a YAML file; CLI flags override the config. Pass `--no-gcs` to disable cloud syncing (works for both ES and LoRA).

| Experiment | Script | Config | Required Args | Key Optional Args |
|---|---|---|---|---|
| **Train NAMM** | `scripts/run_namm.py` | `config/config.yaml` (Hydra only) | — | Hydra overrides via CLI: `key=value` |
| **ES — no NAMM** (baseline) | `scripts/run_es.py` | `scripts/es_default.yaml` | `--run_name` | `--num_iterations`, `--population_size`, `--mini_batch_size`, `--sigma`, `--alpha` |
| **ES + frozen NAMM** | `scripts/run_es.py` | `scripts/es_default.yaml` | `--run_name`, `--namm_checkpoint` | `--cache_size`, `--num_iterations` |
| **LoRA only** (m1) | `scripts/run_lora.py` | `scripts/lora_m1_only.yaml` | `--run_name` | `--num_epochs`, `--learning_rate`, `--lora_rank` |
| **LoRA multi-task** (rh-m1) | `scripts/run_lora.py` | `scripts/lora_rh_m1_instruct.yaml` | `--run_name` | `--num_epochs`, `--eval_interval` |
| **LoRA + frozen NAMM** (rh-m4) | `scripts/run_lora.py` | `scripts/lora_rh_m4_instruct.yaml` | `--run_name`, `--namm_checkpoint` | `--cache_size`, `--eval_interval` |
| **Evaluate checkpoint** | `scripts/run_eval.py` | `scripts/eval_default.yaml` | — | `--es_checkpoint`, `--namm_checkpoint`, `--cache_size`, `--num_samples` |

### Example commands

```bash
# Smoke test (ES, no NAMM) — also: make smoke
python scripts/run_es.py --run_name smoke --num_iterations 2 --population_size 2 --mini_batch_size 2 --no-gcs

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
```

---

## Output Layout

```
experiments/
└── experiment_N/
    └── {es_namm,es_only,es_recency,lora_grad,m1_lora_only,rh_m1_lora_instruct,rh_m4_frozen}/
        └── {run_name}/
            ├── config.json          # full config snapshot
            ├── results.json         # final eval scores (Qasper F1 + per-task)
            ├── examples.json        # captured Q/A examples
            └── checkpoints/
                ├── es_checkpoint_iter_*.pt   # rolling (last 2 kept)
                └── es_checkpoint_final.pt
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

Use `make setup-tpu` for first-time setup (see [Getting Started](#getting-started)).

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
