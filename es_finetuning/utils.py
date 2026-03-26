"""Utilities for ES fine-tuning."""

import gc

import torch


def force_memory_cleanup():
    """Force aggressive memory cleanup (works on GPU, TPU, and CPU)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
