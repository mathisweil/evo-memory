# evo-memory — NAMM + ES / LoRA Fine-Tuning

Fine-tuning LLaMA 3.2-1B-Instruct weights via evolutionary strategies (ES) or
gradient-based LoRA while [NAMM](https://arxiv.org/abs/2410.13166)'s trained
eviction policy manages the KV cache.

Based on the [ES fine-tuning paper](https://arxiv.org/abs/2509.24372) and the
[NAMM paper](https://arxiv.org/abs/2410.13166).

---

## Quick Start (any machine with a CUDA 12.1+ GPU)

```bash
# 1. Clone
git clone https://github.com/mathisweil/evo-memory.git
cd evo-memory

# 2. Configure environment
cp .env.example .env
# Edit .env — set LLM_MODEL_PATH, HF_CACHE_DIR, CUDA_VISIBLE_DEVICES
# GCS_BUCKET / GCS_PROJECT only needed for TPU or cloud experiments

# 3. Create venv and install dependencies (~2 min on first run)
source setup/activate.sh

# 4. Log in to HuggingFace (required for gated LLaMA 3.2 model)
huggingface-cli login

# 5. (Optional) Log in to Weights & Biases
wandb login
```

**Smoke test** — verifies the stack end-to-end in under a minute:
```bash
python scripts/run_es.py \
    --run_name smoke \
    --num_iterations 2 \
    --population_size 2 \
    --mini_batch_size 2 \
    --no-gcs
```

**In subsequent shells:**
```bash
source setup/activate.sh
```

---

## TPU (Google Cloud)

```bash
git clone https://github.com/mathisweil/evo-memory.git
cd evo-memory
cp .env.example .env   # set GCS_BUCKET and GCS_PROJECT
bash setup/setup_tpu.sh
source setup/activate_tpu.sh
```

The TPU setup additionally installs `torch_xla[tpu]` and `google-cloud-storage`.
`activate_tpu.sh` downloads the XLA compilation cache from GCS on startup;
`run_es.py` re-uploads it on exit.

---

## Running Experiments

Three main entry points:

| Script | Purpose |
|---|---|
| `scripts/run_namm.py` | Train NAMM scoring network (CMA-ES) or run eval baselines |
| `scripts/run_es.py` | ES fine-tune LLM weights (with or without frozen NAMM) |
| `scripts/run_lora.py` | LoRA gradient fine-tuning |
| `scripts/run_eval.py` | Evaluate a checkpoint on the full validation set |

Utility scripts:

| Script | Purpose |
|---|---|
| `scripts/generate_report.py` | Generate a comparison report from experiment results |
| `scripts/upload_pretrained.py` | Upload or list pretrained NAMM checkpoints in GCS |
| `scripts/archive_experiment.py` | Archive completed experiments to GCS |

See [docs/examples.md](docs/examples.md) for copy-paste commands covering smoke tests and full runs for each pipeline.

### Experiment output layout

```
experiments/experiment_N/{es_namm,es_only,es_recency}/run_name/
    config.json      # full configuration snapshot
    results.json     # final eval scores
    examples.json    # captured Q/A examples from final eval
    checkpoints/
        es_checkpoint_final.pt
        saved/       # permanent saves (--save_every)
```

`experiments/manifest.json` tracks all experiments locally; GCS mirrors it when `--gcs` is active.

---

## Dependencies

Pin these versions exactly — newer versions break at runtime.

```
torch==2.3.1          # cu121 build; TPU uses torch_xla matching this version
transformers==4.41.2  # CRITICAL: 4.45+ breaks DynamicCache API used by custom LlamaModel patches
peft==0.11.1
numpy<2               # numpy 2.x breaks many downstream packages
```

Full list: [`requirements.txt`](requirements.txt).

**System requirement:** GLIBC >= 2.28 (Ubuntu 20.04+, RHEL 8/9).

**TPU additional dependencies:**
```
torch_xla[tpu]        # matching torch version; from Google's libtpu releases
google-cloud-storage  # for GCS experiment management and XLA cache syncing
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in values before running:

| Variable | Required | Description |
|---|---|---|
| `LLM_MODEL_PATH` | Yes | HuggingFace model ID or local path to LLaMA 3.2-1B-Instruct |
| `HF_CACHE_DIR` | No | HuggingFace cache dir (default: `~/.cache/huggingface`) |
| `CUDA_VISIBLE_DEVICES` | No | GPU index to use (default: `0`) |
| `NAMM_CKPT` | Eval only | Path to a trained NAMM checkpoint |
| `WANDB_API_KEY` | Optional | Only needed when `wandb_log=true` |
| `GCS_BUCKET` | TPU/cloud | GCS bucket name (default: `statistical-nlp`) |
| `GCS_PROJECT` | TPU/cloud | GCS project ID (default: `statistical-nlp`) |

---

## TPU Notes

- **XLA compilation**: First run is slow (~20 min) as XLA compiles graphs.
- **XLA cache syncing**: `activate_tpu.sh` auto-downloads the XLA cache from GCS on startup; `run_es.py` auto-uploads it on exit.
- **Spot VM preemption**: SIGTERM triggers an emergency checkpoint upload. Re-running with the same `--run_name` auto-resumes from the latest GCS checkpoint.
- **Fixed-size tensors**: XLA requires fixed shapes. NAMM pads KV caches to a fixed size on TPU to avoid recompilation.

---

## Documentation

- [docs/examples.md](docs/examples.md) — experiment commands and smoke tests
- [docs/es-ft-namm-guide.md](docs/es-ft-namm-guide.md) — how ES + NAMM interact, parameters, forward pass details
- [docs/namm-guide.md](docs/namm-guide.md) — NAMM scoring network and CMA-ES training
- [docs/es-ft-guide.md](docs/es-ft-guide.md) — standalone ES fine-tuning
- [docs/lora-grad-ft.md](docs/lora-grad-ft.md) — LoRA gradient fine-tuning
