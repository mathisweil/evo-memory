"""TPU-specific guardrails that are testable without TPU hardware."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional in minimal dev envs
    np = None


def is_tpu_device(device: Any) -> bool:
    """Return True if *device* is an XLA/TPU device handle."""
    return str(device).startswith("xla")


def validate_tpu_batch_settings(
    batch_size: Any,
    mini_batch_size: Optional[int] = None,
    *,
    context: str = "training",
) -> int:
    """Validate fixed TPU batch settings and return normalized int batch size."""
    if batch_size == "auto":
        raise ValueError(
            f"TPU {context} requires a fixed integer batch size. "
            "Set --batch_size explicitly (do not use 'auto')."
        )
    if not isinstance(batch_size, Integral):
        raise ValueError(
            f"TPU {context} requires an integer batch size. "
            f"Received: {batch_size!r}"
        )

    fixed_batch_size = int(batch_size)
    if fixed_batch_size <= 0:
        raise ValueError(
            f"TPU {context} requires batch_size > 0. "
            f"Received: {fixed_batch_size}"
        )

    if mini_batch_size is not None and mini_batch_size != fixed_batch_size:
        raise ValueError(
            "TPU requires mini_batch_size == batch_size for stable shapes. "
            f"Received mini_batch_size={mini_batch_size}, "
            f"batch_size={fixed_batch_size}."
        )
    return fixed_batch_size


def pad_partial_tpu_batch(
    contexts: list[str],
    chunk_pop_idxs: Optional[np.ndarray],
    chunk_precached_tensors: Optional[list[Any]],
    batch_size: int,
    *,
    device: Any,
) -> tuple[list[str], Optional[np.ndarray], Optional[list[Any]], int]:
    """Pad a partial batch on TPU to keep batch dimension fixed for XLA."""
    original_context_count = len(contexts)
    if (
        is_tpu_device(device)
        and isinstance(batch_size, int)
        and 0 < original_context_count < batch_size
    ):
        pad_count = batch_size - original_context_count
        contexts = contexts + [contexts[-1]] * pad_count

        if chunk_pop_idxs is not None:
            if np is not None and isinstance(chunk_pop_idxs, np.ndarray):
                pad_pop_idxs = np.repeat(chunk_pop_idxs[-1], pad_count)
                chunk_pop_idxs = np.concatenate([chunk_pop_idxs, pad_pop_idxs])
            else:
                chunk_pop_idxs = list(chunk_pop_idxs) + [chunk_pop_idxs[-1]] * pad_count

        if chunk_precached_tensors is not None:
            chunk_precached_tensors = (
                chunk_precached_tensors
                + [chunk_precached_tensors[-1]] * pad_count
            )

    return contexts, chunk_pop_idxs, chunk_precached_tensors, original_context_count
