"""Helpers for TPU-friendly prompt-length bucketing."""

from __future__ import annotations

import bisect
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

try:
    from namm.evaluator import SEQ_LEN_BUCKETS
except ImportError:  # pragma: no cover - tests can pass explicit buckets
    SEQ_LEN_BUCKETS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]


def bucket_sequence_length(
    seq_len: int,
    *,
    bucket_boundaries: Sequence[int] = SEQ_LEN_BUCKETS,
) -> int:
    """Round *seq_len* up to the nearest configured bucket length."""
    idx = bisect.bisect_left(bucket_boundaries, seq_len)
    if idx < len(bucket_boundaries):
        return int(bucket_boundaries[idx])
    return int(((seq_len + 1023) // 1024) * 1024)


def build_bucketed_request_pools(
    *,
    task_prompts: Mapping[str, Sequence[str]],
    tokenizer,
    sampled_requests_per_task: int,
    max_prompt_conditioning: int | None,
    add_special_tokens: bool,
    bucket_boundaries: Sequence[int] = SEQ_LEN_BUCKETS,
) -> Dict[int, Dict[str, List[int]]]:
    """Build request-index pools keyed by TPU sequence-length bucket."""
    if sampled_requests_per_task <= 0:
        raise ValueError(
            "sampled_requests_per_task must be > 0, received "
            f"{sampled_requests_per_task}"
        )

    bucket_pools: Dict[int, Dict[str, List[int]]] = {}

    for task_name, prompts in task_prompts.items():
        per_task_buckets: Dict[int, List[int]] = {}
        for idx, prompt in enumerate(prompts):
            token_count = len(
                tokenizer(
                    prompt,
                    add_special_tokens=add_special_tokens,
                ).input_ids
            )
            if max_prompt_conditioning is not None:
                token_count = min(token_count, max_prompt_conditioning)
            bucket_len = bucket_sequence_length(
                token_count,
                bucket_boundaries=bucket_boundaries,
            )
            per_task_buckets.setdefault(bucket_len, []).append(idx)

        for bucket_len, indices in per_task_buckets.items():
            bucket_pools.setdefault(bucket_len, {})[task_name] = indices

    eligible_pools = {}
    task_names = sorted(task_prompts)
    for bucket_len, per_task_indices in sorted(bucket_pools.items()):
        if all(
            len(per_task_indices.get(task_name, ())) >= sampled_requests_per_task
            for task_name in task_names
        ):
            eligible_pools[bucket_len] = {
                task_name: list(per_task_indices[task_name])
                for task_name in task_names
            }

    return eligible_pools


def sample_bucketed_requests(
    bucket_pools: Mapping[int, Mapping[str, Sequence[int]]],
    *,
    sampled_requests_per_task: int,
) -> tuple[int, Dict[str, np.ndarray]]:
    """Sample one training minibatch from a single eligible bucket."""
    if not bucket_pools:
        raise ValueError("bucket_pools is empty.")

    bucket_choices = np.array(sorted(bucket_pools), dtype=np.int64)
    bucket_len = int(np.random.choice(bucket_choices))

    requests_dict = {}
    for task_name, indices in bucket_pools[bucket_len].items():
        chosen = np.random.choice(
            np.array(indices, dtype=np.int64),
            size=sampled_requests_per_task,
            replace=False,
        )
        requests_dict[task_name] = chosen

    return bucket_len, requests_dict


def make_bucketed_resample_fn(
    task_sampler,
    bucket_pools: Mapping[int, Mapping[str, Sequence[int]]],
    *,
    sampled_requests_per_task: int,
    log_fn=None,
):
    """Build a resample function that keeps each training step in one bucket."""

    def resample_fn():
        bucket_len, requests_dict = sample_bucketed_requests(
            bucket_pools,
            sampled_requests_per_task=sampled_requests_per_task,
        )
        task_sampler.set_requests_per_task(requests_dict)
        if log_fn is not None:
            log_fn(bucket_len, requests_dict)
        return requests_dict

    return resample_fn


def iter_compile_warmup_requests(
    bucket_pools: Mapping[int, Mapping[str, Sequence[int]]],
    *,
    sampled_requests_per_task: int,
) -> Iterable[tuple[int, Dict[str, np.ndarray]]]:
    """Yield deterministic warmup request selections for each active bucket."""
    for bucket_len in sorted(bucket_pools):
        requests_dict = {}
        for task_name, indices in bucket_pools[bucket_len].items():
            requests_dict[task_name] = np.array(
                list(indices[:sampled_requests_per_task]),
                dtype=np.int64,
            )
        yield bucket_len, requests_dict
