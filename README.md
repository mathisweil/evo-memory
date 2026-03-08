# ES Fine-Tuning with NAMM — LLaMA 3.2-1B-Instruct

Fine-tuning LLaMA 3.2-1B-Instruct weights via evolutionary strategies (ES) while [NAMM](https://arxiv.org/abs/2410.13166)'s trained eviction policy manages the KV cache. Based on the [ES fine-tuning paper](https://arxiv.org/abs/2509.24372) and the [NAMM paper](https://arxiv.org/abs/2410.13166).

The core question: can the base model learn to cooperate with its eviction policy?

---

## Quick Start

**On a UCL VM:**
```bash
git clone -b es-fine-tuning https://github.com/mathisweil/evo-memory.git
bash evo-memory/scripts/setup.sh
```

**On any other machine:**
```bash
git clone -b es-fine-tuning https://github.com/mathisweil/evo-memory.git
bash evo-memory/scripts/setup.sh --dir ~/ft-namm
```

This clones the repo, creates a venv, installs all dependencies, and prompts for HuggingFace + wandb login.

See `scripts/setup.sh --help` for options (`--user`, `--gpu`, `--noclaude`, `--dir`).

**In subsequent shells:**
```bash
source scripts/activate.sh
```

---

## Running Experiments

Three entry points:

| Script | Purpose |
|---|---|
| `run_namm_training.py` | Train NAMM scoring network (CMA-ES) or run eval baselines |
| `run_es_finetuning.py` | ES fine-tune LLM weights (with or without frozen NAMM) |
| `run_eval.py` | Evaluate a checkpoint on the full validation set |

See [docs/examples.md](docs/examples.md) for copy-paste commands covering smoke tests and full runs for each pipeline.

---

## Dependencies

Pin these versions exactly — newer versions break at runtime.

```
torch==2.3.1          (cu121 build)
transformers==4.41.2  (4.45+ breaks DynamicCache API)
peft==0.11.1          (newer versions depend on transformers 4.45+)
numpy<2               (numpy 2.x breaks many downstream packages)
```

System requirement: GLIBC ≥ 2.28 (RHEL 8/9, Ubuntu 20.04+).

HuggingFace access required for gated LLaMA 3.2-1B model: `huggingface-cli login`.

---

## Documentation

- [docs/examples.md](docs/examples.md) — experiment commands and smoke tests
- [docs/es-ft-namm-guide.md](docs/es-ft-namm-guide.md) — how ES + NAMM interact, parameters, forward pass details
- [docs/namm-guide.md](docs/namm-guide.md) — NAMM scoring network and CMA-ES training
- [docs/es-ft-guide.md](docs/es-ft-guide.md) — standalone ES fine-tuning
- [docs/to_do_list.md](docs/to_do_list.md) — phased experiment plan
