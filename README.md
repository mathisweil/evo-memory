# evo-memory

Fine-tuning Llama-3.2-1B-Instruct with LoRA while [NAMM](https://arxiv.org/abs/2410.13166) manages the KV cache.

---

## Paper

Companion code for the TACL submission (UCL SNLP 2026). The paper studies three model variants — **Base** (pretrained Llama-3.2-1B), **FTS** (LoRA fine-tuned with a full KV cache), and **FTE** (LoRA fine-tuned with NAMM eviction active) — each evaluated under two inference regimes: **FC** (full cache at eval) and **EC** (evicted cache at eval, `K=1024`). The headline result (Figure 1) is that fine-tuning under eviction (FTE-EC) closes most of the gap between FTS-FC and FTS-EC without changing the eval-time cache budget.

See `experiment_specification.md` for the full reproduction recipe, datasets, hyperparameters, checkpoint locations, and per-figure analysis commands.

> **Terminology note.** Code and configs still use the earlier M-scheme on disk: `m1_lora_5t.yaml` ≡ FTS, `m3_lora_frozen_namm_5t.yaml` ≡ FTE. YAML and GCS paths have not been renamed; only the documentation uses Base/FTS/FTE.

---

## Project Structure

```
evo-memory/
├── pyproject.toml                # Python metadata + all dependencies (single source of truth)
├── uv.lock                       # uv-managed lockfile (committed to git)
├── .env.example                  # environment variable template
│
├── config/                       # Hydra configuration (used by run_namm.py)
│   ├── config.yaml               #   main config with composable defaults
│   ├── model/                    #   LLM / evaluator configs
│   ├── policy/                   #   memory eviction policies (BAM is used in the paper; others exploratory)
│   ├── evolution/                #   CMA-ES and dummy evolution configs
│   ├── task/                     #   dataset / sampler configs
│   ├── run/                      #   experiment-specific run presets
│   ├── trainer/                  #   training and eval trainer configs
│   └── typing/                   #   precision / attention configs
│
├── scripts/                      # main entry points
│   ├── run/                      #   training + evaluation drivers
│   │   ├── run_es.py             #     ES fine-tuning
│   │   ├── run_lora.py           #     LoRA gradient-based fine-tuning
│   │   ├── run_namm.py           #     NAMM policy training (Hydra)
│   │   ├── run_joint.py          #     joint alternating NAMM + adapter training
│   │   ├── run_eval.py           #     evaluation runner
│   │   └── eval_namm_splits.py   #     multi-split NAMM/LoRA evaluator
│   ├── analysis/                 #   diagnostic / probing tools
│   │   ├── check_eviction_stats.py
│   │   ├── eviction_representation_analysis.py
│   │   ├── ghost_information_analysis.py
│   │   ├── hidden_state_shift_analysis.py
│   │   ├── paired_delta_analysis.py
│   │   └── profile_namm.py
│   ├── reporting/                #   figures, tables, exploratory case studies
│   │   ├── generate_paper_figures.py
│   │   ├── generate_report.py    #     cross-experiment comparison reports
│   │   ├── plot_main_table.py
│   │   ├── case_study_attention.py  # EXPLORATORY: per-prompt attention heatmaps (not in paper)
│   │   └── case_study_entropy.py    # EXPLORATORY: attention-entropy trajectories (not in paper)
│   ├── infra/                    #   GCS / experiment lifecycle
│   │   ├── upload_pretrained.py  #     GCS checkpoint management
│   │   ├── archive_experiment.py #     experiment archival to GCS
│   │   ├── download_artifacts.py
│   │   └── organize_eval_results.py
│   └── configs/                  #   YAML hyperparameter presets
│       ├── m1_lora_5t.yaml              #   FTS — LoRA, full cache (paper)
│       ├── m3_lora_frozen_namm_5t.yaml  #   FTE — LoRA, NAMM active (paper)
│       ├── m4_joint_lora_5t.yaml        #   joint LoRA + NAMM (future work, not in paper)
│       ├── joint_default.yaml           #   joint training defaults
│       ├── eval_default.yaml            #   evaluation configuration
│       ├── eval_main_table.yaml         #   main table evaluation config
│       └── deprecated/                  #   pre-FAIR-01 configs (reproducibility only)
│
├── es_finetuning/                # ES optimizer module — EXPLORATORY (not used in paper)
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
│   └── policy/                   #   eviction policies (BAM deep scoring used in paper; recency, H2O, ScissorHands are exploratory)
│
├── utils/                        # shared utilities
│   ├── experiment.py             #   manifest, config loading, eval dispatch
│   ├── helpers.py                #   model loading, tokenization, data processing
│   └── hydra_helpers.py          #   Hydra configuration utilities
│
├── tests/                        # unit tests
├── data/                         # dataset storage (e.g. LongBench)
└── docs/                         # additional guides (ES, LoRA, NAMM, TPU notes)
```

---

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
# GPU (CUDA 12.1)
uv sync --extra gpu

# CPU only
uv sync --extra cpu

# TPU
uv sync --extra tpu

# Dev tools (add to any of the above)
uv sync --extra gpu --extra dev
```

`uv sync` creates `.venv/`, resolves against `uv.lock`, and installs the project in editable mode. Activate with `source .venv/bin/activate`, or prefix commands with `uv run` to skip activation.

On UCL GPU machines, home directories have strict storage quotas. Point `uv`'s cache at the quota directory before syncing:

```csh
setenv UV_CACHE_DIR $QUOTA_DIR/uv_cache
```

## Service logins

```bash
# HuggingFace (required for gated LLaMA 3.2)
huggingface-cli login

# Weights & Biases
wandb login
```

For GCS access, authenticate with `gcloud auth application-default login`.

Copy `.env.example` to `.env` and fill in the values listed under [Configuration](#configuration).

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
| `CUDA_VISIBLE_DEVICES` | No | `0` | GPU index |
| `NAMM_CKPT` | Eval only | — | Path to a trained NAMM checkpoint |
| `WANDB_API_KEY` | No | — | Only needed when `wandb_log=true` |
| `GCS_BUCKET` | TPU/cloud | `statistical-nlp` | GCS bucket name |
| `GCS_PROJECT` | TPU/cloud | `statistical-nlp` | GCS project ID |

---

## Execution

All scripts accept `--config <yaml>` to load defaults; CLI flags override the config. Pass `--no-gcs` to disable cloud syncing.

### Paper experiments

The six configurations plotted in Figure 1 of the paper are produced by two training runs and a grid of evaluations.

| Paper config | Meaning | Source |
|---|---|---|
| **Base-FC** | pretrained Llama-3.2-1B, full cache at eval | eval only |
| **Base-EC** | pretrained Llama-3.2-1B, NAMM cache (K=1024) at eval | eval only (needs trained NAMM) |
| **FTS-FC** | LoRA trained with full cache, evaluated with full cache | train + eval |
| **FTS-EC** | LoRA trained with full cache, evaluated under NAMM | same train checkpoint, NAMM-on eval |
| **FTE-FC** | LoRA trained under NAMM, evaluated with full cache | same train checkpoint, NAMM-off eval |
| **FTE-EC** | LoRA trained under NAMM, evaluated under NAMM | train + eval |

| Experiment | Script | Config | Required Args |
|---|---|---|---|
| **Train NAMM** | `scripts/run/run_namm.py` | `config/config.yaml` (Hydra) | `'run@_global_=namm_bam_i1_llama32_1b_5t'` |
| **FTS — LoRA, full cache** | `scripts/run/run_lora.py` | `scripts/configs/m1_lora_5t.yaml` | `--run_name` |
| **FTE — LoRA, NAMM active** | `scripts/run/run_lora.py` | `scripts/configs/m3_lora_frozen_namm_5t.yaml` | `--run_name`, `--namm_checkpoint` |
| **Evaluate (any variant × regime)** | `scripts/run/eval_namm_splits.py` | — | `--run_config`, `--lora_checkpoint` (optional), `--namm_checkpoint` (optional) |

> **Joint NAMM + LoRA (M4) is future work and not a paper result.** `scripts/run/run_joint.py` and `scripts/configs/m4_joint_lora_5t.yaml` remain in the repo as exploratory code but are not needed to reproduce Figures 1–7.

### NAMM eviction modes

`run_namm.py` supports two eviction modes, selectable via a Hydra override:

| Mode | Flag | Behaviour |
|---|---|---|
| **Top-k (default)** | `threshold_only=false` | Keeps the `cache_size` highest-scoring tokens — hard budget enforced every step. |
| **Threshold-only** | `threshold_only=true` | Evicts all tokens with score `s_i < 0`; no hard cap. Cache size varies per step, matching the original NAMM paper (Cetin et al., ICLR 2025). |

```bash
# Top-k mode (default, cache_size=1024)
python scripts/run/run_namm.py 'run@_global_=namm_bam_i1_llama32_1b'

# Threshold-only mode — eviction driven purely by learned score threshold
python scripts/run/run_namm.py \
    'run@_global_=namm_bam_i1_llama32_1b' \
    threshold_only=true \
    scoring_initializer=2
```

> **`run@_global_=` syntax required.** The run config is mounted at `@_global_` scope in `config.yaml`, so the override key must match. Plain `run=` raises a Hydra error.

> **`scoring_initializer=2` required for threshold mode.** With the default `scoring_initializer=0` the CMA-ES mean starts at the eviction boundary (score=0). A small perturbation collapses all scores below zero and evicts every token. Starting at 2 gives CMA-ES room to learn selective eviction before the threshold is first reached.

In threshold mode, `max_memory_length` (internal buffer sizing) is unchanged; only the top-k cutoff and the evaluator's physical KV truncation are lifted. Use `scripts/analysis/check_eviction_stats.py --cache_size 0` to diagnose token retention for a threshold-mode checkpoint.

#### Checkpoint frequency

By default NAMM saves `latest.pt` on every iteration. To save only every N steps:

```bash
python scripts/run/run_namm.py \
    'run@_global_=namm_bam_i1_llama32_1b' \
    save_checkpoint_every=10
```

Set `save_checkpoint_every:` (null) in a run config to restore the save-every-step default.

#### Data split vs. buffer size

`max_conditioning_length` sets the model's KV buffer size. A separate key `split_max_conditioning_length` controls which prompts are eligible for the train/val/test split. If only `max_conditioning_length` is overridden, the split filter uses that same value, which can silently empty the training set for long-context tasks. Override independently when needed:

```bash
# Reduce buffer size without filtering out long training examples
python scripts/run/run_namm.py \
    'run@_global_=namm_bam_i1_llama32_1b' \
    max_conditioning_length=2048 \
    split_max_conditioning_length=6500
```

5-task LongBench QA split (prompts in [4096, 6500] tokens), `split_seed=42`:

| Task | Train | Val | Test |
|---|---|---|---|
| `lb/qasper` | 60 | 13 | 14 |
| `lb/2wikimqa` | 56 | 12 | 12 |
| `lb/qasper_e` | 77 | 16 | 17 |
| `lb/hotpotqa_e` | 51 | 10 | 12 |
| `lb/2wikimqa_e` | 62 | 13 | 14 |
| **Total** | **306** | **64** | **69** |

The paper additionally uses an **extended test set** (154 examples, context length 6500–8192 tokens) for OOD evaluation only — it is not part of training/validation.

#### Resuming interrupted runs

`scratch=true` (default) starts fresh. To resume from `latest.pt` in the run's output directory, append `scratch=false`. The best checkpoint (`ckpt.pt`) is only overwritten when `val_tasks_aggregate` improves.

### Example commands

```bash
# FTS — LoRA trained with full cache
python scripts/run/run_lora.py \
    --config scripts/configs/m1_lora_5t.yaml --run_name fts

# FTE — LoRA trained with NAMM active (requires a trained NAMM checkpoint)
python scripts/run/run_lora.py \
    --config scripts/configs/m3_lora_frozen_namm_5t.yaml --run_name fte \
    --namm_checkpoint path/to/namm.pt

# Evaluate a LoRA checkpoint under both FC and EC on test + extended_test
python scripts/run/eval_namm_splits.py \
    --run_config namm_bam_i1_llama32_1b_5t \
    --lora_checkpoint path/to/best_ckpt.pt \
    --namm_checkpoint path/to/namm.pt \
    --cache_size 1024 --splits test extended_test

# Base-FC — pretrained model, full cache, no fine-tuning
python scripts/run/run_eval.py --run_config full_cache_baseline_llama32_1b
```

---

## Output Layout

```
experiments/
└── experiment_N/
    ├── m1_lora_only/<run_name>/       # FTS runs (LoRA, full cache)
    │   ├── config.json
    │   ├── results.json
    │   └── checkpoints/best_ckpt.pt
    └── m3_lora_frozen_namm/<run_name>/ # FTE runs (LoRA, NAMM active)
        ├── config.json
        ├── results.json
        └── checkpoints/best_ckpt.pt
```

`scripts/run/run_eval.py` writes `eval_{baseline,es}_{timestamp}.log` and `results.json` into the checkpoint's parent directory (or `--output_dir`).

`scripts/run/run_namm.py` (Hydra) writes to `outputs/{date}/{time}/` relative to the repo root.

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

Install with `uv sync --extra tpu` on the TPU VM (see [Setup](#setup)).

First XLA run takes ~20 min for compilation. The cache is synced to/from GCS automatically. Spot VM preemption triggers a SIGTERM for emergency checkpoint upload; re-run with the same `--run_name` to resume.

---

## Utility Scripts

| Script | Purpose |
|---|---|
| `scripts/reporting/generate_report.py` | Compare results across experiments |
| `scripts/infra/upload_pretrained.py` | Upload / list pretrained NAMM checkpoints in GCS |
| `scripts/infra/archive_experiment.py` | Archive completed experiment dirs to GCS |
