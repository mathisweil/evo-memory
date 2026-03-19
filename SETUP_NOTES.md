# Setup Guide — evo-memory

Minimal, step-by-step instructions to run the project from a fresh clone.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10 |
| CUDA | 12.8 (cu128) |
| Conda | any recent |

---

## Step 1 — Clone the repo

```bash
git clone -b romain_implementation https://github.com/mathisweil/evo-memory.git
cd evo-memory
```

---

## Step 2 — Create the Conda environment

```bash
conda env create -f environment.yaml -n evo-memory
conda activate evo-memory
```

> **On clusters with limited home-dir quota**, redirect caches before creating
> the env:
> ```csh
> setenv PIP_CACHE_DIR /your/project/volume/.pip_cache
> conda env create -f environment.yaml -p /your/project/volume/envs/evo-memory
> conda activate /your/project/volume/envs/evo-memory
> ```

---

## Step 3 — Install PyTorch (CUDA build)

PyTorch must be installed separately from the PyTorch index.

```bash
pip install torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

---

## Step 4 — Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

| Variable | Description |
|----------|-------------|
| `LLM_MODEL_PATH` | Local path or HuggingFace model ID for Llama-3.2-1B-Instruct |
| `HF_CACHE_DIR` | Where HuggingFace datasets/models are cached (optional) |

**Option A — local snapshot** (no internet required at runtime):

```bash
# Download once:
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
    --local-dir /data/models/llama-1b-instruct-hf

# Set in .env:
LLM_MODEL_PATH=/data/models/llama-1b-instruct-hf
```

**Option B — HuggingFace Hub** (requires internet and a gated-model token):

```bash
# Set in .env:
LLM_MODEL_PATH=meta-llama/Llama-3.2-1B-Instruct
HUGGING_FACE_HUB_TOKEN=hf_...
```

Load `.env` into your shell before running experiments:

```bash
# bash/zsh
export $(grep -v '^#' .env | xargs)

# csh (UCL cluster default)
foreach line (`grep -v '^#' .env`)
  setenv `echo $line | sed 's/=/ /'`
end
```

---

## Step 5 — Verify the setup

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
```

Expected output:
```
2.7.0+cu128  True
4.41.2
```

---

## Running experiments

All commands must be run from the repo root with the conda env active.
Set `HYDRA_FULL_ERROR=1` for full tracebacks.

### NAMM training (CMA-ES)

```bash
python main.py +run=rh_instruct_namm_train seed=42
```

### NAMM evaluation (after training)

Update `init_from` in `cfgs/run/rh_instruct_namm_eval.yaml` to your checkpoint,
or pass it on the command line:

```bash
python main.py +run=rh_instruct_namm_eval init_from=/path/to/ckpt.pt seed=1337
```

Or use the eval script:

```bash
NAMM_CKPT=/path/to/ckpt.pt bash run_namm_instruct_eval.sh
```

### Recency baseline

```bash
python main.py +run=rh_instruct_recency cache_size=1024 seed=1337
```

### LoRA m1 (no NAMM)

```bash
python main.py +run=rh_m1_lora_instruct seed=1337
```

### LoRA m4 (frozen NAMM)

Update `init_from` in `cfgs/run/rh_m4_lora_instruct.yaml` to the trained NAMM
checkpoint, then:

```bash
python main.py +run=rh_m4_lora_instruct seed=1337
```

---

## Critical dependency notes

### `transformers==4.41.2` is mandatory

The code deeply patches `LlamaModel` internals. Version 4.45+ changes the
`DynamicCache` API, `cache_position` handling, and causal mask construction,
causing `RuntimeError` during both training and evaluation.

`environment.yaml` already pins this. If you install dependencies manually,
verify with:

```bash
pip show transformers | grep Version
# Version: 4.41.2
```

### `numpy<2`

`numpy 2.x` breaks several downstream packages (`evaluate`, `lm-eval`). The
pin `numpy==1.26.4` in `environment.yaml` handles this automatically.

---

## Shell notes (UCL cluster — csh)

The UCL cluster default shell is `csh`. Use `setenv` instead of `export`:

| bash | csh |
|------|-----|
| `export VAR=value` | `setenv VAR value` |
| `VAR=value command` | `setenv VAR value; command` |
| `HYDRA_FULL_ERROR=1 cmd` | `setenv HYDRA_FULL_ERROR 1` then `cmd` |

---

## Known issues

**Attention mask off-by-one** — After NAMM eviction the causal mask can be
1 token shorter than the KV cache, causing:
```
RuntimeError: The size of tensor a (N) must match the size of tensor b (N-1)
```
This is a known issue under investigation in `memory_llms/llama.py`.

**CMA-ES slow convergence** — With `pop_size=8` and `samples_batch_size=2`,
the fitness signal is noisy. `sample_D_mean` barely changes in the first 20
steps. This is expected; 200 iterations are typically needed.
