# Setup & Troubleshooting Notes — evo-memory-giacommo

This document summarises the steps taken to clone, set up, and begin running
the `dev/joint-namm-lora-es` branch from Mathis Weil's fork, on the UCL CS
cluster under user `rhautier`.

---

## 1. Cloning the fork branch

The repo lives at a GitHub fork. We extracted the branch name and URL from the
full GitHub tree URL:

```bash
git clone -b dev/joint-namm-lora-es https://github.com/mathisweil/evo-memory.git .
```

(Run from inside the empty target directory
`/cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo`.)

---

## 2. Conda environment setup

### 2.1 Pointing the env to the project volume

The original `environment.yaml` had `prefix` pointing to gmaralla's home. We
changed two things:

| Field    | Before (gmaralla)                                         | After (rhautier)                                          |
|----------|-----------------------------------------------------------|-----------------------------------------------------------|
| `name`   | `th2`                                                     | `th2` (kept short — cannot contain path separators)       |
| `prefix` | `/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/th2` | `/cs/student/project_msc/2025/csml/rhautier/envs/th2` |

### 2.2 Removing PyTorch & NVIDIA packages from the yaml

`torch==2.7.0+cu128` (and `torchaudio`, `torchvision`) are CUDA-specific builds
hosted on PyTorch's own index, not on PyPI. The pinned `nvidia-*` packages also
conflict with whatever torch version pip resolves. We removed all of them from
the `pip:` section (plus `triton`) so the rest of the env can install cleanly:

```yaml
# Lines removed from pip section:
# - torch==2.7.0+cu128
# - torchaudio==2.7.0+cu128
# - torchvision==0.22.0+cu128
# - nvidia-cublas-cu12==...  (and all other nvidia-* packages)
# - triton==3.3.0
```

### 2.3 Creating the environment

**Important:** On the UCL cluster the default shell is `csh`, so use `setenv`
instead of `export`.

```csh
# Redirect pip cache to the project volume (avoids filling the 10 GB home quota)
setenv PIP_CACHE_DIR /cs/student/project_msc/2025/csml/rhautier/.pip_cache

# Create the env (use -p to force the absolute prefix path)
conda env create -f environment.yaml -p /cs/student/project_msc/2025/csml/rhautier/envs/th2
```

### 2.4 Installing PyTorch with CUDA separately

After the env is created, activate it and install torch from the PyTorch index:

```csh
conda activate /cs/student/project_msc/2025/csml/rhautier/envs/th2
pip install torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

### 2.5 Downgrading transformers

The codebase deeply patches HuggingFace `LlamaModel` internals and was written
for transformers **4.41.x**. Version 4.45+ changes the `DynamicCache` API,
`cache_position` handling, and causal mask construction, causing multiple runtime
errors. Downgrade:

```bash
pip install transformers==4.41.2
```

---

## 3. Storage / quota issues

The home directory (`/cs/student/msc/csml/2025/rhautier/`) has a **10 GB quota**.
The project volume (`/cs/student/project_msc/2025/csml/rhautier/`) has ~3 TB free.

### Problems encountered

1. **pip cache in home dir** (~2 GB) — cleared with `rm -rf ~/.cache/pip`.
   Prevented by setting `PIP_CACHE_DIR` (see above).

2. **Conda env created inside home dir** — if conda interprets `prefix` as
   relative, the env lands at `~/cs/student/.../envs/th2` instead of the
   absolute path. Fix: always use `-p <absolute_path>` on the command line.
   The misplaced copy was removed with `rm -rf ~/cs/`.

---

## 4. Code fixes applied

### 4.1 Model path

The config pointed to gmaralla's HF cache. Updated to rhautier's local copy:

**File:** `cfgs/model/wrapped_llm/llama32-1b-instruct.yaml` (line 10)

```yaml
# Before:
pretrained_llm_name: /cs/student/project_msc/2025/csml/gmaralla/.hf_cache/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6

# After:
pretrained_llm_name: /cs/student/project_msc/2025/csml/rhautier/evo-memory/llama-1b-instruct-hf
```

### 4.2 Dataset trust_remote_code

LongBench requires `trust_remote_code=True`.

**File:** `task_sampler.py` (line 89)

```python
# Before:
dataset = load_dataset('THUDM/LongBench', task_name, split='test')

# After:
dataset = load_dataset('THUDM/LongBench', task_name, split='test', trust_remote_code=True)
```

---

## 5. Running NAMM training with CMA-ES

```csh
setenv HYDRA_FULL_ERROR 1
torchrun --standalone --nproc_per_node=1 main.py \
    'run@_global_=namm_bam_i1_llama32_1b_instruct'
```

### Key config: `cfgs/run/namm_bam_i1_llama32_1b_instruct.yaml`

| Parameter             | Value  |
|-----------------------|--------|
| `max_iters`           | 50     |
| `pop_size`            | 8      |
| `cache_size`          | 1024   |
| `samples_batch_size`  | 16     |
| Evolution algorithm   | CMA-ES |
| Tasks                 | qasper, passage_retrieval_en, narrativeqa |

---

## 6. Status / remaining issues

- Environment is created and PyTorch with CUDA 12.8 is installed.
- Model loads successfully, datasets download correctly.
- **Pending:** confirm training runs end-to-end after downgrading to
  `transformers==4.41.2`. The 4.45.2 version caused multiple incompatibilities
  with the custom LlamaModel code (DynamicCache indexing, cache_position
  splitting, causal mask shapes).

---

## Quick reference: csh gotchas

| bash                        | csh equivalent              |
|-----------------------------|-----------------------------|
| `export VAR=value`          | `setenv VAR value`          |
| `VAR=value command`         | `setenv VAR value; command` |
| `HYDRA_FULL_ERROR=1 cmd`   | `setenv HYDRA_FULL_ERROR 1` then `cmd` |
