# Technology Stack

**Analysis Date:** 2026-02-25

## Languages

**Primary:**
- Python 3.10 - Core implementation language for all modules

**Secondary:**
- YAML - Configuration files (Hydra configs)

## Runtime

**Environment:**
- PyTorch with CUDA 12.1 - Deep learning framework and GPU acceleration

**Package Manager:**
- Conda - Primary package management via `env.yaml` and `env_minimal.yaml`
- pip - Secondary package installation within conda environment
- Lockfile: `env.yaml` (full dependencies snapshot with exact versions)

## Frameworks

**Core ML:**
- PyTorch 2.3.0 - Neural network implementation
- Transformers 4.41.2 - Hugging Face library for language models
- Accelerate 0.31.0 - Distributed training utilities
- vLLM 0.5.0.post1 - Large language model inference optimization
- vLLM-flash-attn 2.5.9 - Flash attention for vLLM

**Configuration:**
- Hydra 1.3.2 - Configuration framework with composition support
- OmegaConf 2.3.0 - Configuration object management

**Evaluation:**
- LM-Eval 0.4.2 - Language model evaluation framework
- Lm-format-enforcer 0.10.1 - Output format constraints for LM generation
- Outlines 0.0.45 - Structured generation for language models

**NLP/Text Processing:**
- spaCy 3.7.5 - NLP pipeline (with legacy and loggers)
- NLTK 3.8.1 - Natural language toolkit
- Tokenizers 0.19.1 - Fast tokenizer implementation
- tiktoken 0.7.0 - OpenAI tokenizer
- SentencePiece 0.2.0 - Language model preprocessing
- Fugashi 1.3.2 - Japanese morphological analyzer
- ftfy 6.2.0 - Text encoding fixer

**Testing/Evaluation:**
- crfm-helm 0.5.2 - Benchmark evaluation framework
- Rouge-score 0.1.2 - ROUGE metric for summarization
- Evaluate 0.4.2 - Hugging Face evaluation metrics
- Datasets 2.20.0 - Hugging Face dataset loading

**Build/Development:**
- CMake 3.29.5.1 - Build system
- Ninja 1.11.1.1 - Build runner

## Key Dependencies

**Critical:**
- torch 2.3.0 - Deep learning core
- transformers 4.41.2 - Language model implementations
- vllm 0.5.0.post1 - Inference optimization (required for efficient model evaluation)
- numpy 1.26.4 - Numerical computing
- accelerate 0.31.0 - Distributed training (DDP support)

**Infrastructure:**
- bitsandbytes 0.43.1 - CUDA operations for quantization and optimization
- einops 0.8.0 - Tensor operations simplification
- peft 0.11.1 - Parameter-efficient fine-tuning adapters
- safetensors 0.4.3 - Safe model serialization
- xformers 0.0.26.post1 - Optimized transformer operations

**Data Processing:**
- pandas 2.2.2 - Data manipulation
- scipy 1.13.1 - Scientific computing
- scikit-learn 1.5.0 - Machine learning utilities
- tqdm 4.66.4 - Progress bars

**Monitoring/Logging:**
- wandb 0.17.2 - Weights and Biases experiment tracking
- sentry-sdk 2.5.1 - Error tracking and monitoring
- prometheus-client 0.20.0 - Prometheus metrics
- prometheus-fastapi-instrumentator 7.0.0 - FastAPI instrumentation

**Serialization:**
- PyYAML 6.0.1 - YAML parsing
- PyArrow 16.1.0 - Apache Arrow data format
- msgpack 1.0.8 - Binary serialization

**Testing/Validation:**
- pydantic 2.7.4 - Data validation via Python types
- jsonschema 4.22.0 - JSON schema validation

**Utilities:**
- click 8.1.7 - CLI framework
- typer 0.12.3 - CLI with type hints
- FastAPI 0.111.0 - API framework (for potential serving)
- Starlette 0.37.2 - ASGI framework
- uvicorn 0.30.1 - ASGI server
- python-dotenv 1.0.1 - Environment variable loading
- requests 2.32.3 - HTTP client
- cloudpathlib 0.18.1 - Cloud storage path handling
- Ray 2.24.0 - Distributed computing framework

**Additional Tools:**
- gitpython 3.1.43 - Git repository interaction
- matplotlib 3.9.1 - Visualization
- seaborn 0.13.2 - Statistical visualization
- PIL/Pillow 10.3.0 - Image processing
- FFmpeg 7.0.1 - Video/audio processing

## Configuration

**Environment:**
- Conda environment specification: `env.yaml` (full) and `env_minimal.yaml` (minimal)
- Configuration via `cfgs/config.yaml` - Main Hydra configuration entry point
- Task configs: `cfgs/task/*.yaml` - Task definitions (passage retrieval, LongBench, Choubun)
- Model configs: `cfgs/model/*.yaml` - Model loading specifications
- Policy configs: `cfgs/policy/*.yaml` - Memory policy configurations
- Evolution configs: `cfgs/evolution/*.yaml` - Evolution algorithm settings
- Trainer configs: `cfgs/trainer/*.yaml` - Training parameters

**Key config structure:**
- Defaults composition with Hydra
- Model: `pretrained_llm_name` (defaults to meta-llama/Meta-Llama-3-8B)
- Device: CUDA (with distributed training support)
- Seed-based reproducibility: `seed`, `deterministic_behavior` flags

## Platform Requirements

**Development:**
- NVIDIA GPU with CUDA 12.1 support (GPU acceleration required)
- Linux environment (based on codebase design)
- 12GB+ VRAM for base model inference
- Python 3.10 environment

**Production:**
- NVIDIA CUDA 12.1 compatible hardware
- Distributed training support via NCCL backend
- DDP (Distributed Data Parallel) for multi-GPU training
- bfloat16 precision support (configurable dtype in training)

**Network:**
- Hugging Face Hub access (for model downloads)
- Weights and Biases (wandb) API endpoint access (if logging enabled)
- Sentry API endpoint (if error tracking enabled)

---

*Stack analysis: 2026-02-25*
