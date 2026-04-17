"""NAMM Efficiency Audit — Profiling & Benchmark Script

Instruments the NAMM forward path to measure wall-clock time per stage,
GPU memory usage, and optionally runs torch.profiler for top-N operator
breakdown.

Usage:
    python scripts/profile_namm.py [hydra overrides]

Example:
    python scripts/profile_namm.py run=namm_bam_i1_llama32_1b

The script reuses the existing hydra config infrastructure to load a model
and memory policy, then runs instrumented NAMM calls at different cache
sizes and reports the results.
"""

import os
import sys
import time
import copy
import contextlib
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig

# Ensure project root is on path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from namm.run_utils import make_eval_model, stochasticity_setup


# ── Timing helpers ───────────────────────────────────────────────────────────

class CudaTimer:
    """Context manager for GPU-synchronised wall-clock timing."""

    def __init__(self):
        self.elapsed_ms = 0.0

    def __enter__(self):
        torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize()
        self._t1 = time.perf_counter()
        self.elapsed_ms = (self._t1 - self._t0) * 1000.0


class MemoryTracker:
    """Track GPU memory delta across a code block."""

    def __init__(self):
        self.delta_bytes = 0

    def __enter__(self):
        torch.cuda.synchronize()
        self._before = torch.cuda.memory_allocated()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize()
        self._after = torch.cuda.memory_allocated()
        self.delta_bytes = self._after - self._before


# ── Instrumented NAMM call ───────────────────────────────────────────────────

def profile_single_namm_call(
    memory_policy,
    past_key_values,
    attn_weights_list,
    attention_mask,
    num_new_tokens,
    position_ids=None,
):
    """Run one update_cache call with per-stage timing.

    Returns a dict of stage_name -> elapsed_ms.
    """
    from namm.policy.deep import DeepMP

    if not isinstance(memory_policy, DeepMP):
        # Fallback: just time the entire call
        with CudaTimer() as t:
            out = memory_policy.update_cache(
                past_key_values=past_key_values,
                num_new_tokens=num_new_tokens,
                attn_weights_list=attn_weights_list,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return {"TOTAL": t.elapsed_ms}, out

    # ── Monkey-patch update_layer_cache_impl_ for per-stage timing ──
    timings = defaultdict(float)
    original_impl = memory_policy.update_layer_cache_impl_

    def timed_impl(
        token_embedding_params,
        scoring_network_params,
        seletion_criteria_params,
        layer_id,
        key_cache,
        value_cache,
        num_new_tokens,
        attn_weights=None,
        attn_mask=None,
        position_ids=None,
        **kwargs,
    ):
        bs, n_heads, num_all_tokens, n_embd = key_cache.shape
        device = key_cache.device
        new_sequences = num_all_tokens == num_new_tokens

        if attn_mask is not None:
            attn_mask_proc = attn_mask.unsqueeze(-2)[..., -num_all_tokens:]
        else:
            attn_mask_proc = torch.ones(
                bs, 1, num_all_tokens, dtype=torch.long, device=device)

        if memory_policy.requires_position_ids:
            pos_ids = memory_policy.process_position_ids(
                position_ids=position_ids,
                num_all_tokens=num_all_tokens,
                num_new_tokens=num_new_tokens,
                attention_mask=attn_mask_proc)
            if not new_sequences and memory_policy.cache_position_ids[layer_id] is not None:
                curr_pos_ids = memory_policy.cache_position_ids[layer_id]
                pos_ids = torch.concat([curr_pos_ids, pos_ids], dim=-1)
        else:
            pos_ids = position_ids

        if memory_policy.requires_attn_scores:
            attn_weights = memory_policy.process_attn_weights(
                attn_weights=attn_weights)

        # Stage 1: Feature extraction (STFT + EMA)
        with CudaTimer() as t1:
            token_embedding = memory_policy.token_embedding.get_tokens_embedding(
                layer_id=layer_id,
                parameters=token_embedding_params,
                key_cache=key_cache,
                value_cache=value_cache,
                new_sequences=new_sequences,
                num_new_tokens=num_new_tokens,
                attn_weights=attn_weights,
                attn_mask=attn_mask_proc,
                position_ids=pos_ids,
                analyze=False,
            )
        timings["1_feature_extraction"] += t1.elapsed_ms

        # Stage 2: Scoring network (MLP/BAM)
        with CudaTimer() as t2:
            token_scores = memory_policy.scoring_network.get_tokens_score(
                layer_id=layer_id,
                parameters=scoring_network_params,
                token_embeddings=token_embedding,
                new_sequences=new_sequences,
                num_new_tokens=num_new_tokens,
                attn_weights=attn_weights,
                attn_mask=attn_mask_proc,
                position_ids=pos_ids,
                analyze=False,
            )
        timings["2_scoring_network"] += t2.elapsed_ms

        # Stage 3: Selection (threshold/topk)
        from namm.policy.base_dynamic import is_tpu
        _tpu_kwargs = dict(kwargs)
        if is_tpu() and hasattr(memory_policy, 'cache_validity_mask'):
            cvm = memory_policy.cache_validity_mask[layer_id]
            if cvm is not None:
                _tpu_kwargs['cache_validity_mask'] = cvm

        with CudaTimer() as t3:
            retained_idxs, new_mask = memory_policy.selection_criteria.select_new_tokens(
                layer_id=layer_id,
                parameters=seletion_criteria_params,
                token_scores=token_scores,
                new_sequences=new_sequences,
                num_new_tokens=num_new_tokens,
                attn_weights=attn_weights,
                attn_mask=attn_mask_proc,
                position_ids=pos_ids,
                threshold_shift=0.0,
                analyze=False,
                **_tpu_kwargs,
            )
        timings["3_selection"] += t3.elapsed_ms

        # Stage 4: Filter buffers
        with CudaTimer() as t4:
            memory_policy.selection_criteria.filter_buffer_values(
                layer_id=layer_id, retained_idxs=retained_idxs)
            memory_policy.scoring_network.filter_buffer_values(
                layer_id=layer_id, retained_idxs=retained_idxs)
            memory_policy.token_embedding.filter_buffer_values(
                layer_id=layer_id, retained_idxs=retained_idxs)
        timings["4_filter_buffers"] += t4.elapsed_ms

        # Stage 5: KV cache reindexing
        with CudaTimer() as t5:
            exp_retained_idxs = retained_idxs.unsqueeze(-1).expand(
                -1, -1, -1, n_embd)
            key_cache = torch.gather(key_cache, dim=-2, index=exp_retained_idxs)
            value_cache = torch.gather(value_cache, dim=-2, index=exp_retained_idxs)
        timings["5_kv_reindex"] += t5.elapsed_ms

        # Position ID update
        if memory_policy.requires_position_ids:
            memory_policy.cache_position_ids[layer_id] = torch.gather(
                pos_ids, dim=-1, index=retained_idxs)

        return key_cache, value_cache

    # Patch and run
    memory_policy.update_layer_cache_impl_ = timed_impl

    with CudaTimer() as t_total:
        out = memory_policy.update_cache(
            past_key_values=past_key_values,
            num_new_tokens=num_new_tokens,
            attn_weights_list=attn_weights_list,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    timings["TOTAL"] = t_total.elapsed_ms

    # Restore original
    memory_policy.update_layer_cache_impl_ = original_impl

    return dict(timings), out


# ── Main profiling routine ───────────────────────────────────────────────────

def run_profiling(model, memory_policy, device, dtype, num_warmup=3,
                  num_trials=10, cache_sizes=None):
    """Profile NAMM across different cache sizes."""

    if cache_sizes is None:
        cache_sizes = [256, 512, 1024]

    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    num_attn_heads = config.num_attention_heads

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
               'float16': torch.float16}.get(dtype, torch.bfloat16)

    results = {}

    for cache_size in cache_sizes:
        print(f"\n{'='*60}")
        print(f"Cache size: {cache_size} tokens")
        print(f"{'='*60}")

        # Determine a reasonable num_new_tokens
        # (the NAMM is called with nup new tokens at a time)
        num_new_tokens = min(256, cache_size)
        num_all_tokens = cache_size

        # Create synthetic inputs
        bs = 1
        past_key_values = tuple(
            (torch.randn(bs, num_kv_heads, num_all_tokens, head_dim,
                         device=device, dtype=ptdtype),
             torch.randn(bs, num_kv_heads, num_all_tokens, head_dim,
                         device=device, dtype=ptdtype))
            for _ in range(num_layers)
        )

        # Attention weights: (bs, num_attn_heads, num_new_tokens, num_all_tokens)
        attn_weights_list = [
            torch.randn(bs, num_attn_heads, num_new_tokens, num_all_tokens,
                        device=device, dtype=ptdtype).softmax(dim=-1)
            for _ in range(num_layers)
        ]

        attention_mask = torch.ones(bs, num_all_tokens, device=device,
                                    dtype=torch.long)

        position_ids = torch.arange(
            num_new_tokens, device=device).unsqueeze(0)

        # Initialize memory policy state
        memory_policy.initialize_buffers()
        for lid in range(memory_policy.num_memory_layers):
            memory_policy.update_layer_rotary_offset(
                layer_id=lid,
                num_new_tokens=num_all_tokens,
                num_all_tokens=num_all_tokens,
            )

        # Set pop params
        pop_params = memory_policy.pop_params
        memory_policy.set_params_batch_idxs(
            param_idxs=torch.zeros(bs, dtype=torch.long, device=device))

        # Warmup
        for _ in range(num_warmup):
            # Reset buffers between runs
            memory_policy.initialize_buffers()
            for lid in range(memory_policy.num_memory_layers):
                memory_policy.update_layer_rotary_offset(
                    layer_id=lid,
                    num_new_tokens=num_all_tokens,
                    num_all_tokens=num_all_tokens,
                )
            profile_single_namm_call(
                memory_policy=memory_policy,
                past_key_values=past_key_values,
                attn_weights_list=attn_weights_list,
                attention_mask=attention_mask,
                num_new_tokens=num_new_tokens,
                position_ids=position_ids,
            )

        # Measured runs
        all_timings = defaultdict(list)
        for trial in range(num_trials):
            memory_policy.initialize_buffers()
            for lid in range(memory_policy.num_memory_layers):
                memory_policy.update_layer_rotary_offset(
                    layer_id=lid,
                    num_new_tokens=num_all_tokens,
                    num_all_tokens=num_all_tokens,
                )
            timings, _ = profile_single_namm_call(
                memory_policy=memory_policy,
                past_key_values=past_key_values,
                attn_weights_list=attn_weights_list,
                attention_mask=attention_mask,
                num_new_tokens=num_new_tokens,
                position_ids=position_ids,
            )
            for stage, ms in timings.items():
                all_timings[stage].append(ms)

        # Print results
        print(f"\n{'Stage':<30} | {'Mean (ms)':>10} | {'Std (ms)':>10}")
        print("-" * 56)
        for stage in sorted(all_timings.keys()):
            vals = all_timings[stage]
            mean_ms = np.mean(vals)
            std_ms = np.std(vals)
            print(f"{stage:<30} | {mean_ms:>10.2f} | {std_ms:>10.2f}")

        results[cache_size] = {
            stage: (np.mean(vals), np.std(vals))
            for stage, vals in all_timings.items()
        }

    return results


def run_torch_profiler(model, memory_policy, device, dtype):
    """Run torch.profiler and report top-10 ops by self CUDA time."""
    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    num_attn_heads = config.num_attention_heads

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
               'float16': torch.float16}.get(dtype, torch.bfloat16)

    cache_size = 512
    num_new_tokens = 256
    bs = 1

    past_key_values = tuple(
        (torch.randn(bs, num_kv_heads, cache_size, head_dim,
                     device=device, dtype=ptdtype),
         torch.randn(bs, num_kv_heads, cache_size, head_dim,
                     device=device, dtype=ptdtype))
        for _ in range(num_layers)
    )
    attn_weights_list = [
        torch.randn(bs, num_attn_heads, num_new_tokens, cache_size,
                    device=device, dtype=ptdtype).softmax(dim=-1)
        for _ in range(num_layers)
    ]
    attention_mask = torch.ones(bs, cache_size, device=device, dtype=torch.long)
    position_ids = torch.arange(num_new_tokens, device=device).unsqueeze(0)

    memory_policy.initialize_buffers()
    for lid in range(memory_policy.num_memory_layers):
        memory_policy.update_layer_rotary_offset(
            layer_id=lid, num_new_tokens=cache_size, num_all_tokens=cache_size)

    memory_policy.set_params_batch_idxs(
        param_idxs=torch.zeros(bs, dtype=torch.long, device=device))

    # Warmup
    for _ in range(3):
        memory_policy.initialize_buffers()
        for lid in range(memory_policy.num_memory_layers):
            memory_policy.update_layer_rotary_offset(
                layer_id=lid, num_new_tokens=cache_size,
                num_all_tokens=cache_size)
        memory_policy.update_cache(
            past_key_values=past_key_values,
            num_new_tokens=num_new_tokens,
            attn_weights_list=attn_weights_list,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    # Profile
    memory_policy.initialize_buffers()
    for lid in range(memory_policy.num_memory_layers):
        memory_policy.update_layer_rotary_offset(
            layer_id=lid, num_new_tokens=cache_size,
            num_all_tokens=cache_size)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        memory_policy.update_cache(
            past_key_values=past_key_values,
            num_new_tokens=num_new_tokens,
            attn_weights_list=attn_weights_list,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    print("\n" + "=" * 60)
    print("Top-10 ops by self CUDA time")
    print("=" * 60)
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=10))

    return prof


# ── Entry point ──────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../config",
            config_name="config")
def main(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stochasticity_setup(cfg, seed_offset=0)

    print("Loading model and memory policy...")
    (memory_policy, memory_model, memory_evaluator,
     evolution_algorithm, _) = make_eval_model(cfg)

    dtype = getattr(cfg, 'dtype', 'bfloat16')
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
               'float16': torch.float16}.get(dtype, torch.bfloat16)

    memory_model = memory_model.to(device=device, dtype=ptdtype)
    memory_policy = memory_model.memory_policy

    # Set initial params from CMA-ES
    init_params = evolution_algorithm.best_params.unsqueeze(0).expand(
        memory_policy.pop_size, -1).to(device)
    memory_model.set_memory_params(init_params)

    print(f"\nModel: {memory_model.config.name_or_path}")
    print(f"Layers: {memory_model.config.num_hidden_layers}")
    print(f"KV heads: {memory_model.config.num_key_value_heads}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"NAMM param count: {memory_policy.param_size}")

    # Run profiling
    print("\n" + "=" * 60)
    print("NAMM Stage-Level Profiling")
    print("=" * 60)
    results = run_profiling(
        model=memory_model,
        memory_policy=memory_policy,
        device=device,
        dtype=dtype,
        num_warmup=3,
        num_trials=10,
        cache_sizes=[256, 512, 1024],
    )

    # Run torch.profiler
    if device == 'cuda':
        run_torch_profiler(
            model=memory_model,
            memory_policy=memory_policy,
            device=device,
            dtype=dtype,
        )

    # Print summary
    print("\n" + "=" * 60)
    print("NAMM Efficiency Audit - Profiling Summary")
    print("=" * 60)
    print(f"Hardware: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    print(f"Model: {memory_model.config.name_or_path}")
    print(f"NAMM params: {memory_policy.param_size}")

    for cache_size, stages in results.items():
        print(f"\n--- Cache size: {cache_size} ---")
        print(f"{'Stage':<30} | {'Mean (ms)':>10} | {'Std (ms)':>10}")
        print("-" * 56)
        for stage in sorted(stages.keys()):
            mean_ms, std_ms = stages[stage]
            print(f"{stage:<30} | {mean_ms:>10.2f} | {std_ms:>10.2f}")


if __name__ == "__main__":
    main()
