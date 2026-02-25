# Coding Conventions

**Analysis Date:** 2026-02-25

## Naming Patterns

**Files:**
- Lowercase with underscores: `utils.py`, `memory_trainer.py`, `utils_longbench.py`
- Utility modules prefixed with `utils_`: `utils_hydra.py`, `utils_log.py`, `utils_longbench.py`
- Category-specific files with descriptive names: `memory_trainer.py`, `task_sampler.py`, `memory_evaluator.py`
- Metric files named after benchmark: `choubun_metrics.py`, `longbench_metrics.py`
- Submodules in directories: `memory_policy/base.py`, `memory_llms/base.py`, `memory_evolution/cma_es.py`

**Functions:**
- Snake case throughout: `get_gpu_memory_mb()`, `reset_peak_gpu_memory_stats()`, `get_nonlinearity()`
- Private functions use leading underscore: `_modules`, `_record_eval_stats`
- Descriptive names indicating action/result: `compute_masked_statistics()`, `merge_statistics()`, `unpack_kv_cache()`
- Getter/setter patterns: `get_first_subseq_split()`, `set_params_batch_idxs()`, `get_layer_params()`
- Predicate functions use `is_` or `has_` prefix: `is_oom_exception()`, `are_sync_buffers_frozen()`, `memory_policy_has_buffers_to_merge()`

**Variables:**
- Snake case for local and instance variables: `batch_size`, `max_len`, `padding_side`, `mask_i`
- Descriptive names for tensors: `padded_attn_mx`, `unpacked_cache`, `masked_values`
- Constants in UPPERCASE: `BYTES_TO_MB` (line 35 in `utils.py`)
- Loop variables use single letter only when appropriate (rare): typically full names preferred
- ANSI color codes defined as class attributes: `COLOR.BLACK`, `COLOR.RED`, etc.

**Types:**
- Use descriptive names: `MemoryTrainer`, `MemoryPolicy`, `MemoryEvolution`, `MemoryHFEvaluator`
- Base classes with Base prefix: `MemoryPolicy`, `ParamMemoryPolicy`, `BaseEmbeddingWrapper`
- Configuration dataclasses use suffix `Config`: `TrainerConfig`, `WandbConfig`
- Abstract base classes inherit from `abc.ABC`

## Code Style

**Formatting:**
- No explicit linting/formatting config found (.eslintrc, .prettierrc, pyproject.toml not present)
- Follows Python standard conventions observed in code
- Import statements organized with stdlib first, then third-party
- Indentation: 4 spaces (standard Python)
- Line length: appears to be ~100-120 characters based on observed code

**Linting:**
- No explicit linting configuration detected
- Code does not appear to enforce strict type checking
- Error handling uses explicit exception types and strings

## Import Organization

**Order:**
1. Standard library imports: `os`, `json`, `gc`, `inspect`, `collections`, `copy`, `time`, etc.
2. Third-party imports: `torch`, `numpy`, `hydra`, `transformers`, `omegaconf`, `datasets`, etc.
3. Project-level imports: `from utils import ...`, `from main import ...`, `from memory_policy import ...`

**Path Aliases:**
- No path aliases detected (no `PYTHONPATH` configurations)
- Imports use relative module structure: `from memory_policy import base`, `from memory_evolution import cma_es`
- Hydra imports for configuration management: `from hydra import compose, initialize`, `import hydra.utils`

**Examples from codebase:**
```python
# Standard library first
import os
import json
import gc
import copy
import time
from collections import OrderedDict
from dataclasses import dataclass

# Third-party
import torch
import numpy as np
from torch.nn import functional as F
import hydra
from omegaconf import DictConfig, OmegaConf

# Project imports
from utils import aggregate_score_dict
from memory_trainer import MemoryTrainer
```

## Error Handling

**Patterns:**
- Explicit exception types used: `RuntimeError`, `ValueError`, `NotImplementedError`
- Out-of-memory (OOM) detection via string matching: `is_oom_exception()` checks for CUDA/CUDNN/CPU memory messages (line 125-134 in `utils.py`)
- NotImplementedError raised for unsupported configurations or incomplete implementations (lines 64, 352, etc. in `utils.py`)
- Assertions for validation: `assert self.pop_size % self.world_size == 0` (line 155 in `memory_trainer.py`)
- Try-except used in metric calculations: `try: scores = rouge.get_scores() except: return 0.0` (lines 27-30 in `choubun_metrics.py`)

**Error message style:**
- Descriptive messages with context: `'ERROR: Using auxiliary loss with memory policy with no parameters'` (line 57 in `memory_policy/base.py`)
- Multi-line string concatenation for complex messages: `'ERROR: Repeated prompt indexes found when ' + ...` (line 908 in `memory_trainer.py`)
- ValueError for argument validation: `raise ValueError('Ensure eval_candidate_samples + 1 ...')` (line 228 in `memory_trainer.py`)

## Logging

**Framework:** `print()` for main output, no structured logging library detected

**Patterns:**
- Direct print statements for progress: `print('IN PROMPT')`, `print('SWAPPING')`, `print('Loading the following configurations:')`
- Color output available via `COLOR` class (ANSI codes) but used sparingly
- Rank-aware logging in distributed setting: `print(f'RANK {self.global_rank}: Group processes: {self.all_group_processes}')`
- Warning messages: `print('Warning - decreasing eval candidate samples to' + ...)` (line 223 in `memory_trainer.py`)

**Logging locations in key files:**
- `main.py`: Configuration and instantiation logging (lines 60-78)
- `memory_trainer.py`: Distributed training and checkpoint events
- `utils_hydra.py`: Model loading progress (line 48)
- `utils_longbench.py`: Debug prompt output (lines 57-58)

## Comments

**When to Comment:**
- Multi-line docstrings (triple quotes) for complex operations: function purposes and algorithmic references
- Inline comments for non-obvious tensor shapes and dimensions: `# bs x 1 x n_k`, `# num_samples x num_layers`
- Algorithm references and sources: `# based on accelerate library`, `# http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf`
- Commented-out debug code kept in place: `# raise NotImplementedError`, `# model_config = sub_module.config`
- TODO/NOTE comments: `# NOTE: likely can remove offset`, `# NOTE: add unsqueeze + expand for kv cache`

**JSDoc/TSDoc:**
- Not used; Python project uses docstrings with triple quotes
- Example from `utils.py` (lines 83-84):
```python
def num_attending_queries(n_q, n_k, attn_mask):
    '''determines number of attending queries for each key, applying
       causal ordering'''
```

## Function Design

**Size:** Functions tend to be medium-length (20-100 lines), with utility functions being shorter (5-20 lines)

**Parameters:**
- Explicit parameter names over positional args
- Optional parameters use `Optional[Type]` hint with default `None`
- Batch processing functions use `batch_size` parameter
- Device/dtype parameters explicit: `device=None`, `move_to_gpu=True`, `move_to_cpu=True`

**Return Values:**
- Single returns for utility functions: `return tuple` or `return tensor`
- Multiple return tuples common: `return mean, variance, total_num` (line 231 in `utils.py`)
- Optional returns: `return None` for skip conditions or `-> Optional[Type]` annotation
- Empty returns used for side-effect-only functions: `def set_params(self, params) -> None: pass` (line 63-64 in `memory_policy/base.py`)

**Example patterns:**
```python
# Utility function with clear parameters
def compute_masked_statistics(values, mask, reduce_dims,):
    '''Computing sample mean, summed variances, and number of elements...'''
    mask = mask.expand_as(values)
    masked_values = torch.where(mask, values, torch.zeros_like(values))
    total_num = mask.to(dtype=torch.long).sum(dim=reduce_dims, keepdim=True)
    mean = masked_values.sum(dim=reduce_dims, keepdim=True)/total_num
    variance_sum = (masked_values - mean).square().sum(dim=reduce_dims, keepdim=True)
    return mean, variance_sum, total_num

# Method with optional parameters
def __init__(self, cache_size, init_module: bool = True):
    if init_module:
        nn.Module.__init__(self=self)
    self.cache_size = cache_size
```

## Module Design

**Exports:**
- Implicit - Python files export all top-level definitions by default
- No `__all__` declarations observed
- Utility files export functions and constants: `utils.py` exports 30+ utility functions
- Classes exported from module packages: `memory_policy/base.py` exports `MemoryPolicy` class

**Barrel Files:**
- `__init__.py` files minimal or empty in observed modules
- Example: `memory_evolution/__init__.py` likely imports from `base.py`
- Imports done at submodule level in dependent code

## Type Hints

**Usage Level:** Partial type hints applied
- Function signatures have return type annotations: `-> Callable`, `-> torch.Tensor`, `-> List[torch.Tensor]`, `-> dict`
- Parameter types annotated with Optional/Union: `nonlinearity: Optional[Union[str, Callable]]`
- Dataclass fields fully typed: `model_state: OrderedDict[str, torch.Tensor]`
- Not all internal variables type-hinted
- PyTorch tensor operations rely on runtime type information

**Common patterns:**
```python
from typing import Optional, Union, List, Dict, Tuple, Callable, Any

def get_nonlinearity(
        nonlinearity: Optional[Union[str, Callable]],
) -> Callable:
    pass

def unpack_attn_mxs_from_attn_mask(
        mxs: List[torch.Tensor], attn_mask: torch.Tensor,
        move_to_cpu: bool = True, unpack_dim: int = 0
) -> List[torch.Tensor]:
    pass
```

---

*Convention analysis: 2026-02-25*
