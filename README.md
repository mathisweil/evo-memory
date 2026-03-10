# ES Fine-Tuning with NAMM — LLaMA 3.2-1B-Instruct

Fine-tuning LLaMA 3.2-1B-Instruct weights via evolutionary strategies (ES) while [NAMM](https://arxiv.org/abs/2410.13166)'s trained eviction policy manages the KV cache. Based on the [ES fine-tuning paper](https://arxiv.org/abs/2509.24372) and the [NAMM paper](https://arxiv.org/abs/2410.13166).

The core question: can the base model learn to cooperate with its eviction policy?

---

## Quick Start

**On a UCL VM:**
```bash
git clone -b es-fine-tuning https://github.com/mathisweil/evo-memory.git
bash evo-memory/setup/setup.sh
```

**On any other machine:**
```bash
git clone -b es-fine-tuning https://github.com/mathisweil/evo-memory.git
bash evo-memory/setup/setup.sh --dir ~/ft-namm
```

This clones the repo, creates a venv, installs all dependencies, and prompts for HuggingFace + wandb login.

See `setup/setup.sh --help` for options (`--user`, `--gpu`, `--noclaude`, `--dir`).

**In subsequent shells:**
```bash
source setup/activate.sh
```

---

## Running Experiments

Three entry points:

| Script | Purpose |
|---|---|
| `scripts/run_namm.py` | Train NAMM scoring network (CMA-ES) or run eval baselines |
| `scripts/run_es.py` | ES fine-tune LLM weights (with or without frozen NAMM) |
| `scripts/run_eval.py` | Evaluate a checkpoint on the full validation set |

Utility scripts:

| Script | Purpose |
|---|---|
| `scripts/generate_report.py` | Generate a comparison report from experiment results |
| `scripts/cleanup_checkpoints.py` | Clean up intermediate checkpoint files |

See [docs/examples.md](docs/examples.md) for copy-paste commands covering smoke tests and full runs for each pipeline.

### Experiment Hierarchy

Results are organised under:
```
experiments/experiment_N/{es_namm,es_only}/run_name/
    config.json      # full configuration snapshot
    results.json     # final eval scores
    examples.json    # captured Q/A examples from final eval
    checkpoints/     # final checkpoint only
```

A `manifest.json` in `experiments/` tracks all experiments.

---

## Dependencies

Pin these versions exactly — newer versions break at runtime.

```
torch==2.3.1          (cu121 build for GPU; TPU uses torch_xla matching version)
transformers==4.45.2  (supports Llama 3.2 rope_type)
peft==0.11.1
numpy<2               (numpy 2.x breaks many downstream packages)
```

System requirement: GLIBC >= 2.28 (RHEL 8/9, Ubuntu 20.04+).

HuggingFace access required for gated LLaMA 3.2-1B model: `huggingface-cli login`.

---

## Documentation

- [docs/examples.md](docs/examples.md) — experiment commands and smoke tests
- [docs/es-ft-namm-guide.md](docs/es-ft-namm-guide.md) — how ES + NAMM interact, parameters, forward pass details
- [docs/namm-guide.md](docs/namm-guide.md) — NAMM scoring network and CMA-ES training
- [docs/es-ft-guide.md](docs/es-ft-guide.md) — standalone ES fine-tuning
- [docs/to_do_list.md](docs/to_do_list.md) — phased experiment plan
