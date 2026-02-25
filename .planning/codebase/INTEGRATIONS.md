# External Integrations

**Analysis Date:** 2026-02-25

## APIs & External Services

**Language Models (HuggingFace Hub):**
- Service: Hugging Face Model Hub
  - SDK/Client: `transformers` library (AutoModelForCausalLM, AutoTokenizer)
  - Usage: `utils.py`, `utils_hydra.py`, `memory_llms/llama.py`
  - Default model: `meta-llama/Meta-Llama-3-8B`
  - Supported: Any HF-compatible model (configurable via `pretrained_llm_name`)
  - Location: `cfgs/model/wrapped_llm/base.yaml` for model loading config

**Model Inference Optimization:**
- Service: vLLM
  - SDK/Client: `vllm` package (0.5.0.post1)
  - Usage: `env_minimal.yaml`, `env.yaml`
  - Purpose: Optimized inference for large language models
  - Related: `vllm-flash-attn` for Flash Attention optimization

## Data Storage

**Datasets:**
- HuggingFace Datasets
  - Client: `datasets` library (2.20.0)
  - Usage: `task_sampler.py` - loads evaluation benchmarks
  - Tasks supported:
    - LongBench tasks: `load_dataset()` for lb/* tasks
    - Choubun tasks: `load_dataset()` for choubun/* tasks
  - Config location: `cfgs/task/passage_retrieval_en.yaml`, `cfgs/task/lb_full.yaml`, etc.

**Evaluation Benchmarks:**
- LongBench dataset with configurations
  - Config files: `LongBench/config/dataset2prompt.json`, `LongBench/config/dataset2maxlen.json`
  - Loaded in: `task_sampler.py`
  - Location path: Via `longbench_path` configuration

- Choubun dataset
  - Config files: `Choubun/config/dataset2prompt.json`, `Choubun/config/dataset2maxlen.json`
  - Loaded in: `task_sampler.py`
  - Location path: Via `choubun_path` configuration

**Local File Storage:**
- JSON files for task configuration and results
  - Results stored: `memory_trainer.py` - saves eval results to `${out_dir}/eval_{}.json`
  - Model checkpoints: Saved locally to `${out_dir}` directory
  - Configuration: Hydra output configuration in `cfgs/config.yaml`

**Model Configuration Caching:**
- File: `LongBench/config/model2maxlen.json`
  - Loaded in: `memory_evaluator.py`
  - Purpose: Model-specific maximum token length settings

## Authentication & Identity

**HuggingFace Hub Access:**
- Auth: HuggingFace token (implicit via environment)
  - Token location: Typically `~/.huggingface/token` or `HF_TOKEN` env var
  - Not explicitly configured in codebase
  - Handled transparently by `transformers` library

**Weights and Biases (wandb):**
- Auth: wandb API key
  - Config: `main.py` - `wandb_init()` function
  - Env requirement: `WANDB_API_KEY` environment variable
  - Conditional: Controlled by `wandb_config.wandb_log` flag (default: true)
  - Project name: `cfg.wandb_config.wandb_project` (default: "memory_evolution_hf")

## Monitoring & Observability

**Experiment Tracking:**
- Weights and Biases (wandb)
  - Purpose: Experiment logging and visualization
  - Usage in: `main.py`, `memory_trainer.py`
  - Initialization: `wandb_init()` in `main.py` (line 43-56)
  - Logging: `wandb.log()` calls in `memory_trainer.py` for training metrics
  - Config structure:
    - `wandb_config.wandb_project` - Project name
    - `wandb_config.wandb_group_name` - Experiment group (128 char limit)
    - `wandb_config.wandb_run_name` - Run name (128 char limit)
    - `wandb_config.wandb_log` - Enable/disable flag
  - Location: `cfgs/config.yaml` lines 12-17

**Error Tracking:**
- Sentry SDK
  - Package: `sentry-sdk==2.5.1`
  - Purpose: Error and exception monitoring
  - Not actively integrated in visible code but available in environment

**Logging:**
- Standard Python logging with console output
- Utility functions: `utils_log.py` for structured logging
- Training statistics logged to JSON files and wandb

**Metrics:**
- Prometheus client (0.20.0) and FastAPI instrumentator (7.0.0)
  - Available in environment for potential metrics exposition
  - Not actively used in current training code

## CI/CD & Deployment

**Hosting:**
- Local/On-premises: Training runs on NVIDIA GPU clusters with NCCL backend
- Distributed training: DDP (Distributed Data Parallel) support via PyTorch
- Multi-GPU support configured in `main.py`: `ddp_setup()` function

**CI Pipeline:**
- Not detected - No GitHub Actions, GitLab CI, or other CI service configuration found

**Version Control:**
- Git integration via `gitpython` library (3.1.43)
- Not actively used in training code

## Environment Configuration

**Required Environment Variables:**

Critical for HuggingFace:
- `HF_TOKEN` - Hugging Face API token (for model downloads)

Critical for wandb:
- `WANDB_API_KEY` - Weights and Biases API key (if `wandb_log=true`)

CUDA/Distributed Training:
- `LOCAL_RANK` - Local rank in distributed training
- `RANK` - Global rank in distributed training (from OMPI or PyTorch)
- `WORLD_SIZE` - Total number of training processes
- `OMPI_COMM_WORLD_LOCAL_RANK` - OpenMPI local rank (fallback)
- `OMPI_COMM_WORLD_RANK` - OpenMPI global rank (fallback)
- `OMPI_COMM_WORLD_SIZE` - OpenMPI world size (fallback)

Optional CUDA:
- `CUBLAS_WORKSPACE_CONFIG` - Set to `:4096:8` for deterministic behavior (see `main.py` line 114)

**Secrets Location:**
- HuggingFace token: Implicit via `transformers` library (standard HF auth)
- wandb token: Environment variable or wandb CLI login
- No `.env` file detected in codebase
- Credentials managed outside codebase (expected for ML platforms)

## Webhooks & Callbacks

**Incoming:**
- Not detected - No webhook endpoints configured

**Outgoing:**
- wandb notifications (if configured in wandb dashboard)
- No explicit webhook calls in codebase

## HuggingFace Models

**Model Loading:**
- Location: `cfgs/model/wrapped_llm/base.yaml`
- Method: `AutoModelForCausalLM.from_pretrained()`
- Tokenizer: `AutoTokenizer.from_pretrained()`
- Default: `meta-llama/Meta-Llama-3-8B` (configurable)

**Supported Model Variants:**
- LLaMA variants with RoPE scaling (see `cfgs/model/wrapped_llm/llama3-8b-rope-x4NTK.yaml`)
- Any Hugging Face compatible causal language model

**Model Configuration:**
- Position embeddings detection from model config
- RoPE scaling detection and support for dynamic/NTK scaling
- Maximum position embeddings auto-detection
- Long context support via attention mask handling

## Evaluation & Benchmarks

**Benchmark Integration:**
- LongBench:
  - Full dataset access via HuggingFace Datasets
  - Task configs: `cfgs/task/lb_full.yaml`, `cfgs/task/lb_2subset.yaml`, `cfgs/task/lb_3subset_incr.yaml`
  - Metric: Performance-based (perf)

- Choubun:
  - Full dataset support: `cfgs/task/choubun_full.yaml`
  - Integrated via `load_dataset()`

**Evaluation Framework:**
- LM-Eval: Core evaluation framework (0.4.2)
- CRFM HELM: Benchmark evaluation framework (0.5.2)
- Custom evaluation metrics in `longbench_metrics.py`, `choubun_metrics.py`

---

*Integration audit: 2026-02-25*
