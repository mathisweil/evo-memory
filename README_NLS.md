# evo-memory Setup Guide
## Getting the Repo Running with TinyLlama 1.1B

This guide documents the steps required to run the SakanaAI evo-memory NAMM experiments
using TinyLlama 1.1B as the base model, on a single GPU (tested on RTX 4070 Ti, 12GB VRAM).

---

## 1. Clone the Repo

Make sure to clone the repo when you are in your main storage directory.

```bash
git clone https://github.com/SakanaAI/evo-memory.git
cd evo-memory
```

---

## 2. Create the Conda Environment

First, set the conda environment and package cache directories to the main storage
area so you don't hit your home directory quota:

```bash
export CONDA_ENVS_PATH=/cs/student/project_msc/2025/csml/<your_cs_username>/envs
export CONDA_PKGS_DIRS=/cs/student/project_msc/2025/csml/<your_cs_username>/conda_pkgs
mkdir -p $CONDA_ENVS_PATH $CONDA_PKGS_DIRS
```

Add these to your `~/.bashrc` to make them permanent:

```bash
echo 'export CONDA_ENVS_PATH=/cs/student/project_msc/2025/csml/<your_cs_username>/envs' >> ~/.bashrc
echo 'export CONDA_PKGS_DIRS=/cs/student/project_msc/2025/csml/<your_cs_username>/conda_pkgs' >> ~/.bashrc
source ~/.bashrc
```

Then create and activate the environment:

```bash
conda env create --file=env_minimal_th3.yaml
conda activate th3
```

### Critical version overrides

The repo's `env_minimal_th3.yaml` should have the correct dependencies within. However if you do not have the `transformers==4.41.2` which is required, and the `peft==0.10.0`, 
after creating the environment, downgrade/pin these packages explicitly:

```bash
pip install transformers==4.41.2
pip install peft==0.10.0
```

> **Why:** transformers 4.45+ introduced breaking changes to the `DynamicCache` API
> that are incompatible with the evo-memory codebase. peft versions above 0.10.0
> depend on `EncoderDecoderCache` which was only added in transformers 4.45.

---

## 3. Configure HuggingFace Cache Location

By default HuggingFace downloads models to `~/.cache/huggingface` which may hit
your home directory quota. Redirect it to your project directory:

```bash
export HF_HOME=/path/to/your/project/evo-memory/.hf_cache
mkdir -p $HF_HOME
```

Add this to your `~/.bashrc` to make it permanent:

```bash
echo 'export HF_HOME=/path/to/your/project/evo-memory/.hf_cache' >> ~/.bashrc
```

---

## 4. Configure the Model

The repo is designed for Llama models. We use TinyLlama 1.1B as it is
architecturally identical to Llama and compatible with transformers 4.41.2.

Edit `cfgs/model/wrapped_llm/llama-3.2-1b.yaml`: --> this should already be done in the `tiny_llama_implementation` branch.

```yaml
pretrained_llm_name: TinyLlama/TinyLlama-1.1B-Chat-v1.0
llm_log_name: TinyLlama-1.1B
max_position_id: 32768
```

> TinyLlama will be downloaded automatically from HuggingFace on first run (~2.2GB).
> It is not a gated model so no HuggingFace login is required.

---

## 5. Configure wandb (Optional but Recommended)

The repo logs training metrics to wandb by default. Either log in:

```bash
wandb login
```

Or disable it by adding `wandb_log=false` to your run command.

---

## 6. Run a Sanity Check

Before running a full experiment, verify the pipeline works end to end with
minimal settings. This should complete in ~30 minutes on a 12GB GPU:

```bash
torchrun --standalone --nproc_per_node=1 main.py "run@_global_=namm_bam_i1.yaml" \
    score_normalization_reference=null \
    pop_size=4 \
    samples_batch_size=8 \
    max_iters=20 \
    memory_policy_fixed_delay=128 \
    max_new_tokens=32 \
    eval_max_batch_size=1 \
    max_conditioning_length=1024 \
    max_memory_length=512
```

You should see output like:
```
POP STATS - step 0 | time xxxms | pop/tasks_aggregate_mean:x.xx | ...
saving checkpoint to ./exp_local/...
```

---

## 7. Run a Proper Stage 1 Experiment

Once the sanity check passes, run a proper experiment. Adjust parameters
based on available GPU memory:

```bash
torchrun --standalone --nproc_per_node=1 main.py "run@_global_=namm_bam_i1.yaml" \
    score_normalization_reference=null \
    pop_size=8 \
    samples_batch_size=16 \
    max_iters=200 \
    memory_policy_fixed_delay=256 \
    max_new_tokens=64 \
    eval_max_batch_size=1 \
    max_conditioning_length=2048 \
    max_memory_length=1024
```

Expected runtime: ~12-16 hours on an RTX 4070 Ti.

---

## 8. Run Stages 2 and 3

After stage 1 completes, find the checkpoint:

```bash
ls exp_local/memory_evolution_hf/TinyLlama-1.1B/.../latest.pt
```

Then run stage 2, passing the stage 1 checkpoint:

```bash
torchrun --standalone --nproc_per_node=1 main.py "run@_global_=namm_bam_i2.yaml" \
    score_normalization_reference=null \
    eval_max_batch_size=1 \
    init_from='path/to/stage1/latest.pt'
```

And stage 3 from stage 2:

```bash
torchrun --standalone --nproc_per_node=1 main.py "run@_global_=namm_bam_i3.yaml" \
    score_normalization_reference=null \
    eval_max_batch_size=1 \
    init_from='path/to/stage2/latest.pt'
```

---

## Key Parameters Reference

| Parameter | What it controls | Recommendation |
|---|---|---|
| `pop_size` | CMA-ES population size | 8-16 for single GPU |
| `samples_batch_size` | Sequences evaluated per candidate | 16-32 |
| `max_iters` | Total CMA-ES update steps | 200-500 for real experiments |
| `memory_policy_fixed_delay` | NAMM eviction frequency (tokens) | 128-256 |
| `max_conditioning_length` | Max input context tokens | 2048-4096 |
| `max_memory_length` | Max KV cache size (tokens) | 512-1024 |
| `max_new_tokens` | Max response generation length | 64-128 |
| `eval_max_batch_size` | Parallel sequences during eval | 1 for 12GB GPU |
| `score_normalization_reference` | Reference scores for normalization | Set to null for TinyLlama |

---

## Monitoring Training

Training metrics are logged to wandb. Key metrics to watch:

- `pop/tasks_aggregate_mean` — mean fitness across population, should trend upward
- `pop/tasks_aggregate_std` — spread across population, decreases as CMA-ES converges
- `pop/tasks_aggregate_best` — best candidate score per iteration

Checkpoints are saved to `exp_local/memory_evolution_hf/TinyLlama-1.1B/...` after
every iteration. To resume a run from the latest checkpoint:

```bash
torchrun --standalone --nproc_per_node=1 main.py "run@_global_=namm_bam_i1.yaml" \
    [your other args] \
    init_from='path/to/latest.pt'
```

---

## Common Issues

**OOM during evaluation:**
Reduce `max_conditioning_length`, `max_memory_length`, and set `eval_max_batch_size=1`.

**Disk quota exceeded during model download:**
Set `HF_HOME` to a directory with sufficient space (see Step 3).

**`huggingface-cli: Command not found`:**
```bash
pip install huggingface_hub
```

**`cannot import name 'EncoderDecoderCache'`:**
```bash
pip install peft==0.10.0
```

**Very slow first iteration (>1 hour):**
Reduce `max_conditioning_length` and `samples_batch_size`. At default settings
each iteration takes 1.5+ hours on a 4070 Ti.