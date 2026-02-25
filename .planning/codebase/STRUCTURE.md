# Codebase Structure

**Analysis Date:** 2026-02-25

## Directory Layout

```
NAMM_implementation/
├── main.py                           # Entry point: DDP setup, Hydra config loading
├── memory_trainer.py                 # Training loop orchestration (1400+ lines)
├── memory_evaluator.py               # LLM inference and evaluation (800+ lines)
├── task_sampler.py                   # Dataset loading and prompt sampling (450+ lines)
├── utils.py                          # Utilities: GPU memory, tensor ops, logging
├── utils_longbench.py                # Benchmark-specific scoring and processing
├── utils_hydra.py                    # Hydra helper utilities
├── utils_log.py                      # Logging utilities
├── choubun_metrics.py                # ChouBun task metrics
├── longbench_metrics.py              # LongBench task metrics
│
├── memory_llms/                      # LLM model wrappers
│   ├── __init__.py
│   ├── base.py                       # MemoryModelWrapper, MemoryAttention base classes
│   └── llama.py                      # Llama-specific implementation with NTK scaling
│
├── memory_policy/                    # Memory selection policies (parameterizable)
│   ├── __init__.py
│   ├── base.py                       # MemoryPolicy, ParamMemoryPolicy ABC
│   ├── base_dynamic.py               # DynamicMemoryPolicy, dynamic recency/attention params
│   ├── base_deep_components.py       # DeepMemoryPolicyComponent, embeddings, networks
│   ├── shared.py                     # RegistrationCompatible, SynchronizableBufferStorage
│   ├── deep.py                       # DeepMP: composed deep memory policies
│   ├── deep_embedding.py             # Recency/attention exponent embeddings
│   ├── deep_embedding_shared.py      # Positional and base embeddings
│   ├── deep_embedding_spectogram.py  # STFT-based attention spectrograms
│   ├── deep_embedding_wrappers.py    # Embedding wrapper utilities
│   ├── deep_scoring.py               # MLP, Generalized, TCN scoring networks
│   ├── deep_scoring_bam.py           # BAM (Broadcast Aggregate Module) scoring
│   ├── deep_selection.py             # TopK, Binary, Dynamic selection networks
│   └── auxiliary_losses.py           # Sparsity and L2 norm regularization losses
│
├── memory_evolution/                 # Evolution algorithms
│   ├── __init__.py
│   ├── base.py                       # MemoryEvolution ABC with ask/tell interface
│   └── cma_es.py                     # CMA-ES implementation with eigen-decomposition
│
├── stateless_parallel_modules/       # Efficient parameter transforms
│   ├── __init__.py
│   ├── base.py                       # StatelessGeneralizedOperation ABC
│   ├── attention.py                  # Parallel attention compute
│   └── mlp.py                        # Parallel MLP forward pass
│
├── cfgs/                             # Hydra configuration hierarchy
│   ├── config.yaml                   # Root config with defaults
│   ├── config_run_eval.yaml          # Evaluation-only config
│   ├── trainer/
│   │   └── default.yaml              # MemoryTrainer hyperparameters
│   ├── model/
│   │   ├── hf_evaluator.yaml         # MemoryHFEvaluator config
│   │   └── wrapped_llm/              # LLM model configs
│   │       ├── base.yaml
│   │       └── llama3-8b-rope-x4NTK.yaml
│   ├── policy/
│   │   ├── none.yaml                 # No-op memory policy
│   │   ├── deep.yaml                 # Deep memory policy config
│   │   ├── deep_embedding/           # Embedding strategy configs
│   │   ├── deep_scoring/             # Scoring network configs
│   │   └── deep_selection/           # Selection network configs
│   ├── evolution/
│   │   ├── cma_es.yaml               # CMA-ES config with pop_size, learning rates
│   │   └── dummy.yaml                # Dummy evolution for testing
│   ├── auxiliary_loss/
│   │   └── none.yaml
│   ├── task/
│   │   └── passage_retrieval_en.yaml # Task/metric config
│   ├── run/
│   │   └── base_memory_policy/       # Experiment configs
│   └── typing/
│       └── default.yaml              # Type checking config
│
├── LongBench/                        # LongBench benchmark data
│   └── config/
│       ├── dataset2prompt.json       # Prompt templates per task
│       └── dataset2maxlen.json       # Max generation lengths
│
├── ChouBun/                          # ChouBun benchmark data
│   └── config/
│       ├── dataset2prompt.json
│       └── dataset2maxlen.json
│
├── lb_reference_scores/              # Reference scores for normalization
│   └── per_request/
│
└── generated_outputs/                # (Generated during runs) Model outputs
    └── temp/
```

## Directory Purposes

**`memory_llms/`:**
- Purpose: Model wrappers that integrate memory policies into LLM forward passes
- Contains: ABC base class, Llama-specific implementations with custom attention and position embeddings
- Key files: `base.py` (interface), `llama.py` (Llama-3 with NTK scaling)

**`memory_policy/`:**
- Purpose: Parameterizable memory selection policies that evolve during training
- Contains: Base classes (stateless policies), deep neural network implementations (embeddings, scoring, selection)
- Organization:
  - `base.py`: Core ABC defining policy interface
  - `deep*.py`: Modular components (embeddings, networks) combined via composition
  - `auxiliary_losses.py`: Regularization for evolved parameters

**`memory_evolution/`:**
- Purpose: Evolutionary algorithms for optimizing memory policy parameters
- Contains: CMA-ES with full covariance matrix tracking and rank-one/rank-μ updates
- Key files: `base.py` (interface), `cma_es.py` (implementation with eigendecomposition)

**`stateless_parallel_modules/`:**
- Purpose: Efficient parallel computation over population dimension
- Contains: Generic operations (Linear, MLP, Attention) that process `(pop_size × batch × features)` tensors
- Pattern: Used by policies to apply parameter transforms without explicit loops

**`cfgs/`:**
- Purpose: Hydra configuration hierarchy for reproducible experiments
- Naming: Each subdirectory corresponds to a component (trainer, model, policy, etc.)
- Usage: Merged at runtime based on `defaults` in `config.yaml`

**`LongBench/` and `ChouBun/`:**
- Purpose: Store benchmark metadata and prompt templates
- Contents: JSON files mapping task names to prompt formats and max generation tokens
- Generated: Populated by `TaskSampler` when loading datasets from HuggingFace

## Key File Locations

**Entry Points:**
- `main.py`: Command-line entry, DDP initialization, component instantiation
- `memory_trainer.py::MemoryTrainer.train()`: Main training loop

**Configuration:**
- `cfgs/config.yaml`: Root Hydra config with all defaults
- `cfgs/trainer/default.yaml`: Trainer hyperparameters (batch sizes, eval intervals, checkpointing)
- `cfgs/policy/deep.yaml`: Memory policy architecture selection

**Core Logic:**
- `memory_trainer.py::_train_step()`: Population evaluation and evolution update
- `memory_trainer.py::_evaluate()`: Fitness computation for population
- `memory_evaluator.py::MemoryHFEvaluator.evaluate_lb()`: LLM inference on benchmark
- `task_sampler.py::TaskSampler.evaluate()`: Task execution and scoring

**Testing & Utilities:**
- `utils.py`: GPU memory, tensor operations, collation
- `utils_longbench.py`: Metric computation, dataset processing
- `longbench_metrics.py`, `choubun_metrics.py`: Task-specific scoring

## Naming Conventions

**Files:**
- `memory_*.py`: Core module files (trainer, evaluator, policy, LLMs, evolution)
- `utils*.py`: Utility helpers grouped by domain (utils.py for general, utils_longbench.py for benchmarks)
- `*_metrics.py`: Task-specific metric implementations
- `.yaml`: Hydra configuration files

**Directories:**
- `memory_*`: Core modules handling primary logic (policy, LLMs, evolution)
- `stateless_*`: Parallel computation helpers
- `cfgs`: Hydra configuration root
- `LongBench`, `ChouBun`: Benchmark-specific data directories

**Classes:**
- `Memory*`: Core components (MemoryTrainer, MemoryPolicy, MemoryEvolution, MemoryHFEvaluator)
- `*Wrapper`: Model integration classes (MemoryModelWrapper, MemoryAttention)
- `*Network`: Neural network components (ScoringNetwork, SelectionNetwork, TokenEmbedding)

**Configuration Keys:**
- `max_iters`: Total evolution iterations
- `pop_size`: Population size for CMA-ES
- `cache_size`: Max tokens retained per layer
- `task_batch_size`: Number of tasks per iteration
- `eval_interval`: Steps between fitness evaluations

## Where to Add New Code

**New Memory Policy Component:**
- Implementation: `memory_policy/deep_*.py` (follow pattern of `deep_embedding.py`, `deep_scoring.py`)
- Config: `cfgs/policy/deep_*/new_component.yaml`
- Integration: Add to `__init__.py` exports, reference in `deep.py` composition
- Example: Custom embedding strategy → `memory_policy/deep_embedding_custom.py` + `cfgs/policy/deep_embedding/custom.yaml`

**New Evaluation Task:**
- Implementation: `task_sampler.py` → add `add_*_task()` method
- Metrics: New file `*_metrics.py` following structure of `longbench_metrics.py`
- Config: `cfgs/task/new_task.yaml`
- Data: Create `NewBench/config/dataset2prompt.json` with prompt templates

**New Evolution Algorithm:**
- Implementation: `memory_evolution/new_algo.py` inheriting `MemoryEvolution`
- Config: `cfgs/evolution/new_algo.yaml`
- Integration: Add to `__init__.py` exports
- Example: Genetic Algorithm → `memory_evolution/genetic_algorithm.py`

**New LLM Model:**
- Implementation: `memory_llms/new_model.py` inheriting `MemoryModelWrapper`
- Config: `cfgs/model/wrapped_llm/new_model.yaml`
- Pattern: Wrap HuggingFace model, define custom attention layer with memory integration

**Utilities:**
- Reusable helpers: `utils.py` (tensor ops, GPU monitoring)
- Benchmark-specific: `utils_longbench.py` (scoring, collation)
- Metrics: `*_metrics.py` for task-specific logic

## Special Directories

**`cfgs/`:**
- Purpose: Hydra configuration hierarchy
- Generated: No (hand-authored)
- Committed: Yes
- Update pattern: Add new `.yaml` files for new experiment variants or components

**`generated_outputs/`:**
- Purpose: Temporary storage for LLM generation outputs during evaluation
- Generated: Yes (created by `task_sampler.py` if `store_gen_outputs=True`)
- Committed: No (in `.gitignore`)

**`lb_reference_scores/`:**
- Purpose: Pre-computed baseline scores for normalization
- Generated: Pre-populated (static reference data)
- Committed: Yes
- Usage: Loaded by trainer to normalize population fitness scores

**Output Directory (`out_dir`):**
- Purpose: Checkpoints, logs, evaluation results
- Generated: Yes (created at runtime per Hydra config)
- Committed: No (configured via `out_folder` in config.yaml)
- Structure:
  ```
  {out_folder}/{project}/{group}/{run_name}/{seed}/
  ├── latest.pt          # Latest checkpoint
  ├── ckpt.pt            # Best checkpoint
  ├── iter_*.pt          # Numbered checkpoints (if enabled)
  ├── eval_*.json        # Evaluation results per iteration
  ├── .hydra/            # Hydra metadata
  └── outputs.log        # Merged logs
  ```

---

*Structure analysis: 2026-02-25*
