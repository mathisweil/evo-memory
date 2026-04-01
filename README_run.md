# NAMM Training Run — LLaMA 3.2-1B, 5-Task QA, CMA-ES

## Run Environment

| Field | Value |
|---|---|
| **Host** | `mallard-l.cs.ucl.ac.uk` |
| **OS** | Linux 5.14.0-611.41.1.el9_7.x86_64 |
| **Python** | CPython 3.10.19 |
| **Python executable** | `/cs/student/project_msc/2025/csml/rhautier/envs/th2/bin/python` |
| **Repository** | `https://github.com/mathisweil/evo-memory.git` |
| **Commit** | `8156cf3fac245304a85f16bce5d8aedf70513c08` |
| **Branch** | `rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-16qs-256fixDel-llama32-1b-5t-cs1024` |

## Launch Command

```bash
/cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo/scripts/run_namm.py \
    run@_global_=namm_bam_i1_llama32_1b_5t \
    filter_by_length=8192 \
    cache_size=1024 \
    max_memory_length=1024 \
    run_name_suffix=llama32-1b-5t-cs1024 \
    wandb_project=memory_evolution_hf \
    wandb_group_name=namm-training
```

## Full Resolved Configuration

### Model
- **Base model**: `meta-llama/Llama-3.2-1B-Instruct`
- **Wrapper**: `WrappedLlamaForCausalLM` (custom `LlamaCompatModel` with patched `rope_scaling`)
- **Device**: CUDA

### Task — `rh_multi_qa_5t`
5-task LongBench QA subset (dropped `multifieldqa_en` due to insufficient samples above 4096 tokens).
Includes base tasks plus `_e` (extended) variants for data augmentation:

| # | Task |
|---|------|
| 1 | `lb/qasper` |
| 2 | `lb/2wikimqa` |
| 3 | `lb/qasper_e` |
| 4 | `lb/hotpotqa_e` |
| 5 | `lb/2wikimqa_e` |

Metric: `perf`

### Memory Policy — BAM (deep)
- **Evolution strategy**: CMA-ES
- **Memory policy**: Deep (attention-based spectrogram norm embedding, BAM scoring, binary selection)

| Parameter | Value |
|---|---|
| `scoring_attn_hidden_dim` | 32 |
| `scoring_attn_output_dim` | (default) |
| `scoring_attn_num_heads` | 1 |
| `scoring_attn_bias` | true |
| `scoring_attn_use_rope` | false |
| `scoring_attn_masking_strategy` | backward |
| `scoring_initializer` | 0 |

### Spectrogram (STFT) Parameters
| Parameter | Value |
|---|---|
| `n_fft` | 32 |
| `hop_length` | 16 |
| `window_fn` | Hann (periodic) |
| `pad_mode` | constant |
| `output_magnitudes` | true |

### Embedding Reduction
| Parameter | Value |
|---|---|
| `scoring_reduction_mode` | (null — default) |
| `embedding_reduction_mode` | `ema` |
| `embedding_ema_coeff` | 0.99 |
| `embedding_reduction_learned` | false |

### KV Cache & Token Lengths
| Parameter | Config | CLI Override | Effective |
|---|---|---|---|
| `cache_size` | 1024 | 1024 | **1024** |
| `max_memory_length` | 1024 | 1024 | **1024** |
| `max_new_tokens` | 64 | — | **64** |
| `filter_by_length` | (model default) | 8192 | **8192** |
| `max_conditioning_length` | 6500 | — | **6500** |
| `min_conditioning_length` | 4096 | — | **4096** |
| `memory_policy_fixed_delay` | 256 | — | **256** |

### Evolution & Training
| Parameter | Value |
|---|---|
| `pop_size` | 8 |
| `samples_batch_size` | 16 |
| `batch_size` | 4 |
| `eval_max_batch_size` | 4 |
| `max_iters` | 200 |
| `always_save_checkpoint` | true |
| `keep_past_epoch_checkpoints_every` | 25 |

### Data Splits
| Parameter | Value |
|---|---|
| `train_frac` | 0.7 |
| `val_frac` | 0.15 |
| (test_frac implied) | 0.15 |

### Logging
| Parameter | Value |
|---|---|
| `wandb_log` | true |
| `wandb_project` | `memory_evolution_hf` (CLI) |
| `wandb_group_name` | `namm-training` (CLI) |
| `run_name_suffix` | `llama32-1b-5t-cs1024` (CLI) |
| `add_bos_token` | true |

---

## What Changed in Commit `8156cf3`

**Commit message**: `feat: add _e variant tasks, min_conditioning_length, max_answer_tokens filtering`

This commit builds on the previous work (merged PR #5 — threshold training from `mathisweil/namm_threshold_training`) with the following additions:

### 1. New `_e` (Extended) LongBench Task Variants
- Added `qasper_e`, `hotpotqa_e`, `2wikimqa_e` to LongBench prompt and max-length mappings (`data/longbench/dataset2prompt.json`, `dataset2maxlen.json`).
- Created new task config `rh_multi_qa_5t` using 5 tasks (2 base + 3 extended) to increase the training data pool after aggressive token-length filtering.
- Dropped `multifieldqa_en` entirely because too few of its prompts exceed 4096 tokens.

### 2. Minimum Conditioning Length Filter (`min_conditioning_length`)
- **NAMM pipeline** (`namm/tasks.py`): `apply_train_val_test_split()` now accepts `min_conditioning_length` and filters out prompts shorter than this threshold *before* splitting, so train/val/test partitions are consistently filtered.
- **LoRA pipeline** (`grad_lora_finetuning/datasets.py`): `SFTDataset` also gained `min_conditioning_length`, discarding prompts below the threshold during dataset construction.
- **`run_namm.py`**: Reads `min_conditioning_length` from config and passes it through to the split function.
- This ensures only prompts in the range `[min_conditioning_length, max_conditioning_length]` (i.e., `[4096, 6500]`) are used, targeting the "sweet spot" of sufficiently long but not excessively long contexts.

### 3. Shortest-Answer Token Filtering (`max_answer_tokens`)
- **LoRA pipeline** (`SFTDataset`): New `max_answer_tokens` parameter. For each item, all candidate answers are tokenized; if even the shortest answer exceeds `max_answer_tokens`, the item is discarded. Otherwise, the shortest-fitting answer is selected.
- **NAMM pipeline** (`run_namm.py`): Before the train/val/test split, `filter_answers_by_token_count()` is called with `max_new_tokens` (default 64) to remove items whose answers are too long to generate.
- **Bug fix** in `tasks.py` `filter_answers_by_token_count()`: items with no answers were previously kept (`keep.append(i)`) — now they are correctly skipped.

### 4. Aligned Token Counting Across Pipelines
- Both NAMM (`tasks.py`) and LoRA (`datasets.py`) now count prompt tokens via `tokenizer.apply_chat_template()` minus the BOS token, rather than raw `tokenizer.encode()`. This ensures identical eligible sets and split indices between the two training pipelines.

### 5. New Config Files
- `config/run/namm_bam_i1_llama32_1b_5t.yaml` — The NAMM run config used in this run.
- `scripts/lora_rh_m1_instruct_5t.yaml` — LoRA fine-tuning config for the 5-task setup (model 1).
- `scripts/lora_rh_m4_instruct_5t.yaml` — LoRA fine-tuning config for the 5-task setup (model 4).

### Summary of Changes vs. Previous State (post-PR #5 merge)
| Area | Before (PR #5 merge) | After (commit `8156cf3`) |
|---|---|---|
| Tasks | 6-task set including `multifieldqa_en` | 5-task set with `_e` variants, `multifieldqa_en` dropped |
| Min prompt length | No minimum filter | `min_conditioning_length=4096` filters short prompts |
| Answer length | No answer-token filter | `max_answer_tokens` (= `max_new_tokens`) discards long answers |
| Token counting | Raw `tokenizer.encode()` | `apply_chat_template()` minus BOS for consistency |
| Config files | Single multi-QA task config | Separate 5-task configs for NAMM and LoRA |
