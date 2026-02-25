# Architecture

**Analysis Date:** 2026-02-25

## Pattern Overview

**Overall:** Evolutionary Algorithm with Neural Architecture Search (NAS) for Memory Management

This codebase implements NAMM (Neural Attention Memory Mechanism), a framework for evolving memory policies in large language models using CMA-ES (Covariance Matrix Adaptation Evolution Strategy). The system optimizes parameterized memory selection policies across LLM layers while evaluating performance on long-context benchmarks.

**Key Characteristics:**
- **Evolutionary optimization** of memory policy parameters via CMA-ES
- **Task-driven evaluation** using LongBench and ChouBun datasets with distributed inference
- **Stateless parallel modules** for efficient parameter transformation across population members
- **Hierarchical memory abstraction** separating policy definition from model integration
- **Distributed training** with DDP support for population-based optimization across multiple GPUs

## Layers

**Entry Point & Configuration:**
- Purpose: Initialize distributed training, load models, samplers, and evolution algorithms
- Location: `main.py`
- Contains: DDP setup, Hydra configuration loading, factory functions
- Depends on: All module imports, Hydra, PyTorch
- Used by: Command-line execution

**Trainer (Orchestration):**
- Purpose: Coordinate the evolutionary loop, manage checkpoints, log metrics
- Location: `memory_trainer.py` (1400+ lines)
- Contains: `MemoryTrainer` class with training loop, evaluation scheduling, checkpoint management
- Depends on: `MemoryEvolution`, `MemoryHFEvaluator`, `TaskSampler`, memory policy
- Used by: `main.py` via Hydra instantiation

**Memory Policy Layer (Parameterizable):**
- Purpose: Define how to select, weight, and manage cached tokens
- Location: `memory_policy/` directory
- Contains: Base classes (`MemoryPolicy`, `ParamMemoryPolicy`), deep implementations, scoring/selection networks
- Depends on: `DeepMemoryPolicyComponent`, `stateless_parallel_modules`
- Used by: `MemoryModelWrapper` to control token retention during inference

**Model Wrapper Layer (Integration):**
- Purpose: Integrate memory policy into LLM forward passes
- Location: `memory_llms/` directory
- Contains: `MemoryModelWrapper` base, `LlamaNTKScalingRotaryEmbedding`, custom attention/MLP layers
- Depends on: HuggingFace Transformers, memory policy
- Used by: `MemoryHFEvaluator` for inference, trainer for parameter extraction

**Evaluation Layer (Task Execution):**
- Purpose: Execute LLM inference on benchmark tasks and compute scores
- Location: `memory_evaluator.py` (800+ lines)
- Contains: `MemoryHFEvaluator` class with batch handling, memory management, task evaluation
- Depends on: Model wrapper, task sampler, LM-Eval, Accelerate library
- Used by: Trainer to compute fitness scores for evolution

**Evolution Algorithm:**
- Purpose: Generate, rank, and update memory policy parameters
- Location: `memory_evolution/` directory
- Contains: `MemoryEvolution` base class, `CMAES` implementation with covariance matrix management
- Depends on: NumPy for matrix operations, PyTorch for tensor storage
- Used by: Trainer's optimization loop via `ask()` and `tell()` interface

**Task Sampling (Data Access):**
- Purpose: Load benchmarks, sample prompts, manage dataset access
- Location: `task_sampler.py` (450+ lines)
- Contains: `TaskSampler` for LongBench/ChouBun dataset loading and sampling
- Depends on: HuggingFace Datasets
- Used by: Trainer and evaluator for task iteration

**Utility Modules:**
- Purpose: Cross-cutting concerns (GPU memory, logging, tensor operations)
- Location: `utils.py`, `utils_longbench.py`, `utils_log.py`, metric files
- Contains: Memory monitoring, collation, metric computation
- Used by: Evaluator and trainer for inference optimization

## Data Flow

**Training Loop Iteration:**

1. **Ask** (Evolution): `evolution_algorithm.ask()` → generates population of parameter vectors
2. **Set Params** (Model): `model.set_memory_params(params)` → activates each parameter set
3. **Evaluate** (Inference): For each population member:
   - Sample tasks from `TaskSampler`
   - Run `MemoryHFEvaluator.evaluate_lb()` on LLM with memory policy
   - Collect logits, compute scores via task metrics
4. **Aggregation** (Sync): Gather scores across DDP processes, compute population fitness
5. **Tell** (Evolution): `evolution_algorithm.tell(fitness)` → update CMA-ES distribution
6. **State Management**: Store buffers via `model.merge_buffers_list()` (synchronized params across generations)

**Inference Path (per sample):**

1. **Prompt Encoding** → Input tokens through LLM layers
2. **Memory Policy Decision** (at each layer):
   - Score tokens: `scoring_network(token_reps)` → importance scores
   - Select tokens: `selection_network(scores, cache_state)` → binary or topk mask
   - Update cache: Apply mask, maintain `cache_size` limit
3. **Attention Compute**: Attend only to selected cached tokens
4. **Generation**: Beam search or sampling with memory constraints

**State Management:**

- **Model Parameters**: Memory policy weights evolve via CMA-ES
- **Buffers**: Synchronized statistics (e.g., EMA momentum) aggregated across population
- **Cache**: Token KV states managed during generation, cleared per-sequence
- **Checkpoints**: Saved to `{out_dir}/latest.pt` with evolution state and model params

## Key Abstractions

**MemoryPolicy:**
- Purpose: Defines how to select tokens for caching
- Examples: `ParamMemoryPolicy`, `DeepMP`, `DynamicMemoryPolicy`, `RecencyParams`
- Files: `memory_policy/base.py`, `memory_policy/deep.py`
- Pattern: ABC with registration callbacks for model-specific initialization

**StatelessGeneralizedOperation:**
- Purpose: Parallel linear/MLP transforms over population dimension
- Examples: `GeneralizedLinear`, `GeneralizedGELU` in `stateless_parallel_modules/`
- Pattern: Takes `(pop_size × batch × features)` tensors, applies independent transforms per population index

**MemoryModelWrapper:**
- Purpose: Bridge between policy logic and LLM inference
- Examples: LlamaNTKScaling wrapper for Llama-3
- Files: `memory_llms/base.py`, `memory_llms/llama.py`
- Pattern: Wraps pretrained model, swaps attention/MLP with memory-aware variants

**MemoryEvolution:**
- Purpose: ES strategy for parameter optimization
- Examples: `CMAES` with rank-one and rank-μ updates
- Files: `memory_evolution/base.py`, `memory_evolution/cma_es.py`
- Pattern: ABC with `ask()`/`tell()` for parameter generation and fitness reporting

## Entry Points

**Command Line:**
- Location: `main.py:main()`
- Triggers: `python -m torch.distributed.launch main.py` (DDP) or `python main.py` (single GPU)
- Responsibilities:
  1. Parse Hydra config from `cfgs/config.yaml`
  2. Set up DDP process groups and random seeds
  3. Instantiate components: LLM, memory policy, evolution algorithm, trainer
  4. Call `trainer.train()` to start optimization loop

**Training Loop:**
- Location: `memory_trainer.py:MemoryTrainer.train()`
- Triggers: Called from `main.py` after setup
- Responsibilities:
  1. Iterate from `start_iter` to `max_iters`
  2. On `eval_interval`: Call `_evaluate()` to compute population fitness
  3. Otherwise: Call `_train_step()` to evolve parameters
  4. Manage checkpoints, early stopping, logging

**Evaluation:**
- Location: `memory_evaluator.py:MemoryHFEvaluator.evaluate_lb()`
- Triggers: Called from trainer's `_evaluate()` method
- Responsibilities:
  1. Batch encode/decode LLM outputs
  2. Compute task-specific metrics (F1, EM, ROUGE, etc.)
  3. Return population-level scores

## Error Handling

**Strategy:** Gradual degradation with retry logic and fallbacks

**Patterns:**

- **OOM Handling** (`utils.is_oom_exception()`): Catches CUDA OOM errors, triggers cache clearing and batch size reduction via `find_executable_batch_size()`
- **Distributed Sync**: Early stopping and fitness aggregation synchronized via `dist.all_reduce()`, broadcast via `dist.scatter_object_list()`
- **Checkpoint Safety**: Try load latest checkpoint; fall back to init_from path; restart from scratch if both fail
- **Task Loading**: Dataset loading retried via HuggingFace cache; local prompt files fallback to built-in config

## Cross-Cutting Concerns

**Logging:**
- Framework: `print()` for rank-specific output (e.g., `f"RANK {global_rank}:"`), optionally `wandb.log()` on master process
- Pattern: Trainer logs population statistics, evolution stats, model params every `log_interval` steps
- Location: `memory_trainer.py:train()` and `_train_step()`

**Validation:**
- Population-level: Fitness range checks (negative values indicate OOM)
- Model-level: Memory policy parameter bounds enforced via `param_clip` in evolution
- Task-level: Prompt length vs. context window validated in evaluator

**Authentication:**
- Framework: HuggingFace Hub token via `huggingface-cli login`
- Pattern: Implicit via Transformers library for model downloads

**Memory Management:**
- GPU: Tracked via `get_gpu_memory_mb()`, cleared with `empty_gpu_cache()` and `torch.cuda.empty_cache()`
- Batch Size: Dynamic inference via `find_executable_batch_size()` from Accelerate
- Cache: KV cache size limited to `max_memory_length` per evaluator config

---

*Architecture analysis: 2026-02-25*
