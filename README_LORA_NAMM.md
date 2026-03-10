# LoRA + NAMM Experiments on Llama 3.2 1B Instruct

Extension of the [NAMM codebase](https://github.com/SakanaAI/universal-transformer-memory/) to study the interaction between LoRA fine-tuning and NAMM (Neural Attention Memory Model) eviction policies.

## Setup

### Environment

```bash
conda env create --file=environment_conda_freeze.yaml
conda activate th3
```

This installs all dependencies including `peft`. Alternatively, for a minimal install:

```bash
conda env create --file=env_minimal.yaml
conda activate th2
pip install peft>=0.10.0
```

### Model Weights

Download Llama 3.2 1B Instruct from HuggingFace (requires access):

```bash
huggingface-cli login
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = 'meta-llama/Llama-3.2-1B-Instruct'
AutoModelForCausalLM.from_pretrained(model_id).save_pretrained('./llama-1b-instruct-hf')
AutoTokenizer.from_pretrained(model_id).save_pretrained('./llama-1b-instruct-hf')
"
```

For the base (non-instruct) model, do the same with `meta-llama/Llama-3.2-1B` into `./llama-1b-hf`.

### wandb (optional but recommended)

```bash
wandb login
```

All configs have `wandb_log: true` by default. Set `wandb_log=false` on the CLI to disable.

---

## Repo Structure

| File | Purpose |
|------|---------|
| `main.py` | Entry point for NAMM training (CMA-ES) and eval |
| `lora_finetune.py` | Entry point for LoRA SFT training and eval |
| `task_sampler.py` | Task loading, prompt sampling, train/test splitting, evaluation |
| `memory_llms/llama.py` | Llama model wrapper with KV cache + memory policy hooks |
| `memory_policy/deep.py` | Deep memory policy (NAMM/BAM) — scoring, selection, eviction |
| `memory_trainer.py` | CMA-ES training loop for NAMM |
| `memory_evaluator.py` | Generation-based evaluation (F1 scoring) |
| `cfgs/config.yaml` | Root Hydra config — all runs compose from this |
| `cfgs/run/` | Per-experiment run configs (override defaults) |
| `cfgs/lora/default.yaml` | LoRA hyperparameter defaults |
| `cfgs/run/base_memory_policy/deep/bam/base_bam.yaml` | BAM/NAMM architecture defaults |

---

## Experimental Configurations

Four configurations studying LoRA + NAMM interaction:

| Config | Pipeline | Description |
|--------|----------|-------------|
| **m1** | base LLM → LoRA | LoRA fine-tuning only, no NAMM |
| **m2** | base LLM → NAMM | NAMM training only (existing pipeline) |
| **m3** | base LLM → LoRA → NAMM | LoRA first, then train NAMM on the fine-tuned model |
| **m4** | base LLM → NAMM → LoRA | Train NAMM first, then LoRA with NAMM active (frozen) |

---

## Running Experiments

All commands use single-GPU. Hydra config composition: `cfgs/config.yaml` + `+run=<run_config>`.

### 1. NAMM Training (m2)

Train NAMM on the instruct model with multi-task QA (qasper, multifieldqa_en, hotpotqa, 2wikimqa). Uses CMA-ES with pop_size=8, 200 iterations. Train/test split: 75/25 on qasper.

```bash
python main.py --config-name=config '+run=llama32_instruct_multi_qa_namm'
```

Output: `exp_local/memory_evolution_hf/Llama-3.2-1B/llama32_instruct_namm_multi_qa/1337/ckpt.pt`

### 2. LoRA Fine-tuning — m1 (no NAMM)

SFT on the same multi-task QA prompts. Answer-only loss (context tokens masked). Uses `peft` with `inject_adapter_in_model` (in-place, preserves module paths).

```bash
python lora_finetune.py --config-name=config '+run=llama32_qasper_lora_m1'
```

Key config overrides you may want:
- `lora.lr=1e-5` — learning rate
- `lora.num_epochs=15` — number of epochs
- `lora.max_seq_len=6600` — max tokenized sequence length (keep >=6600 to avoid dropping prompts filtered by max_conditioning_length=6500 words)
- `lora.val_samples=50` — validation set size
- `lora.eval_every=25` — eval frequency in steps
- `lora.save_dir=exp_local/lora/my_run` — output directory

Output: `<save_dir>/lora_best.pt` (best checkpoint by val loss)

### 3. LoRA Fine-tuning — m4 (NAMM active, frozen)

Requires a trained NAMM checkpoint. NAMM evicts tokens from the KV cache during training; LoRA learns to predict well given the compressed context. Uses truncated BPTT (gradients only on the final chunk after NAMM-compressed cache).

```bash
python lora_finetune.py --config-name=config '+run=llama32_qasper_lora_m4'
```

The m4 config points to the NAMM checkpoint at `lora.namm_ckpt`. Update this path if your checkpoint is elsewhere.

### 4. LoRA Fine-tuning — m3 (LoRA then NAMM)

Run m1 first, then train NAMM on the LoRA-finetuned model. Not yet a single config — requires manually merging LoRA weights and running NAMM training on the result.

---

## Evaluation

### Baseline Evals (instruct model, no fine-tuning)

Full cache (no eviction):
```bash
python main.py --config-name=config '+run=llama32_instruct_qasper_fullcache'
```

Recency eviction at different cache sizes:
```bash
python main.py --config-name=config '+run=llama32_instruct_qasper_recency' cache_size=1024
python main.py --config-name=config '+run=llama32_instruct_qasper_recency' cache_size=2048
python main.py --config-name=config '+run=llama32_instruct_qasper_recency' cache_size=4096
```

### NAMM-only Eval (instruct model + NAMM, no LoRA)

```bash
python main.py --config-name=config '+run=llama32_instruct_qasper_namm_eval' cache_size=1024
python main.py --config-name=config '+run=llama32_instruct_qasper_namm_eval' cache_size=2048
python main.py --config-name=config '+run=llama32_instruct_qasper_namm_eval' cache_size=4096
```

The NAMM checkpoint path is set in the config (`init_from`). Override on CLI if needed:
```bash
init_from='path/to/ckpt.pt'
```

### LoRA Eval (with or without NAMM)

The `lora_finetune.py` script handles eval when `lora.num_epochs=0`:

LoRA-only fullcache:
```bash
python lora_finetune.py --config-name=config '+run=llama32_qasper_lora_m1' \
  lora.num_epochs=0 lora.lora_ckpt=path/to/lora_best.pt
```

LoRA + recency at different cache sizes — use the m1 config with policy overrides or create a dedicated eval config.

LoRA + NAMM at different cache sizes:
```bash
python lora_finetune.py --config-name=config '+run=llama32_qasper_lora_m4' \
  lora.num_epochs=0 lora.lora_ckpt=path/to/lora_best.pt cache_size=1024
```

---

## Train/Test Split

All experiments use a deterministic 75/25 split (seed=1337) on overlapping tasks. For qasper: 135 train / 46 test samples (after filtering by `max_conditioning_length=6500` words).

Controlled by `train_split_ratio: 0.75` in run configs. The split is applied in `lora_finetune.py` via `split_task_samples()` and in `main.py` when `train_split_ratio` is set.

To ensure `same_test_train_tasks=False` (so the trainer evaluates the test split), eval configs include a second training task (multifieldqa_en) even when only testing on qasper.

---

## Key Architecture Details

- **NAMM/BAM**: A small MLP scoring network that scores each token in the KV cache. Tokens below a learned threshold are evicted. `cache_size` acts as a hard upper bound via TopK. Trained with CMA-ES (evolutionary strategy) using F1 as fitness.
- **BinarySelection**: Threshold-based eviction — tokens scoring < 0 are evicted regardless of cache occupancy. This means the actual cache can be smaller than `cache_size`.
- **`memory_policy_fixed_delay: 256`**: Eviction runs every 256 new tokens, not after every token. The cache temporarily overshoots `cache_size` by up to 256 between eviction steps.
- **LoRA injection**: Uses `peft.inject_adapter_in_model` (not `get_peft_model`) to preserve module paths. Critical because `memory_llms/llama.py` uses `isinstance(layer, LlamaAttention)` checks and direct attribute access.
- **Debug logging**: `memory_policy/deep.py` prints `[NAMM] layer 0: X -> Y tokens` on each eviction step for layer 0. Grep for `[NAMM]` in output to monitor cache behavior.

---

## Configs Reference

| Config file | What it does |
|-------------|-------------|
| `llama32_instruct_multi_qa_namm` | NAMM training on instruct model (multi-task QA) |
| `llama32_qasper_lora_m1` | m1: LoRA-only SFT |
| `llama32_qasper_lora_m4` | m4: LoRA SFT with frozen NAMM |
| `llama32_instruct_qasper_fullcache` | Baseline eval: instruct model, no eviction |
| `llama32_instruct_qasper_recency` | Baseline eval: instruct model, recency eviction |
| `llama32_instruct_qasper_namm_eval` | NAMM-only eval: instruct model + NAMM |
| `qa_multi_train_qasper_eval` | Task config: multi-task QA training, qasper-only test |

All run configs are in `cfgs/run/` and are composed via `+run=<name>` on the CLI.
