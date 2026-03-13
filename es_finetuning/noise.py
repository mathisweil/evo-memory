"""Deterministic noise perturbation and ES weight update utilities.

Ported from:
- utils/worker_extn.py:23-47 (correlated mode)
- es_fine-tuning_conciseness_iid.py:110-115 (iid mode)
- es_fine-tuning_conciseness.py:291-310 (ES update)
"""

import torch

from .device import sync_device, empty_cache


def _get_param_entries(model, param_names):
    """Cache stable parameter references for the ES hot path."""
    cache_key = tuple(param_names)
    cache = getattr(model, "_es_param_entries_cache", None)
    if cache is None or cache["key"] != cache_key:
        all_params = dict(model.named_parameters())
        entries = [(name, all_params[name]) for name in param_names]
        cache = {"key": cache_key, "entries": entries}
        setattr(model, "_es_param_entries_cache", cache)
    return cache["entries"]


def _get_update_buffers(model, param_names, param_entries):
    """Reuse per-parameter update buffers across ES iterations."""
    cache_key = tuple(param_names)
    cache = getattr(model, "_es_update_buffers_cache", None)
    needs_rebuild = cache is None or cache["key"] != cache_key
    if not needs_rebuild:
        for buffer, (_, param) in zip(cache["buffers"], param_entries):
            if (buffer.shape != param.shape
                    or buffer.dtype != param.dtype
                    or buffer.device != param.device):
                needs_rebuild = True
                break
    if needs_rebuild:
        buffers = [torch.zeros_like(param) for _, param in param_entries]
        cache = {"key": cache_key, "buffers": buffers}
        setattr(model, "_es_update_buffers_cache", cache)
    return cache["buffers"]


def _make_noise(shape, dtype, device, seed):
    """Generate deterministic noise on CPU, then move to device.

    torch.Generator does not support XLA devices, so we always generate
    on CPU and transfer.  The overhead is negligible vs. inference.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    noise = torch.randn(shape, dtype=dtype, device="cpu", generator=gen)
    return noise.to(device)


def perturb_weights(model, seed, sigma, param_names, mode="correlated"):
    """Add deterministic noise to model parameters (in-place).

    Args:
        model: The model whose weights to perturb.
        seed: Random seed for reproducible noise generation.
        sigma: Noise scale.
        param_names: List of parameter names to perturb.
        mode: "correlated" (same seed for all params) or
              "iid" (seed + param_index for independent noise per param).
    """
    param_entries = _get_param_entries(model, param_names)
    for i, (name, p) in enumerate(param_entries):
        effective_seed = int(seed) if mode == "correlated" else int(seed + i)
        noise = _make_noise(p.shape, p.dtype, p.device, effective_seed)
        p.data.add_(sigma * noise)
        del noise

    device = param_entries[0][1].device
    sync_device(device)
    empty_cache(device)


def restore_weights(model, seed, sigma, param_names, mode="correlated"):
    """Remove previously added noise from model parameters (in-place).

    Uses the same seed to regenerate the exact noise that was added,
    then subtracts it.

    Args:
        model: The model whose weights to restore.
        seed: Same seed used in the corresponding perturb_weights call.
        sigma: Same sigma used in the corresponding perturb_weights call.
        param_names: List of parameter names that were perturbed.
        mode: Must match the mode used in perturb_weights.
    """
    param_entries = _get_param_entries(model, param_names)
    for i, (name, p) in enumerate(param_entries):
        effective_seed = int(seed) if mode == "correlated" else int(seed + i)
        noise = _make_noise(p.shape, p.dtype, p.device, effective_seed)
        p.data.add_(-sigma * noise)
        del noise

    device = param_entries[0][1].device
    sync_device(device)
    empty_cache(device)


def apply_es_update(model, seeds, normalized_rewards, sigma, alpha,
                    param_names, population_size, mode="correlated"):
    """Apply the ES weight update: weighted sum of perturbation directions.

    For each parameter, reconstructs each population member's noise using
    its seed, weights by normalized reward, and applies the aggregated
    gradient estimate.

    Args:
        model: The model to update.
        seeds: List of seeds (one per population member).
        normalized_rewards: Array/list of normalized rewards, same length as seeds.
        sigma: Noise scale (same as used in perturbation).
        alpha: Learning rate.
        param_names: List of parameter names to update.
        population_size: Number of population members.
        mode: "correlated" or "iid".
    """
    param_entries = _get_param_entries(model, param_names)
    update_buffers = _get_update_buffers(model, param_names, param_entries)
    for i, ((name, p), update) in enumerate(zip(param_entries, update_buffers)):
        update.zero_()

        for seed_idx in range(population_size):
            r_norm = normalized_rewards[seed_idx]
            seed = seeds[seed_idx]
            effective_seed = int(seed) if mode == "correlated" else int(seed + i)

            noise = _make_noise(p.shape, p.dtype, p.device, effective_seed)
            noise.mul_(float(r_norm))
            update.add_(noise)
            del noise

        update.div_(population_size)
        p.data.add_(alpha * update)

    device = param_entries[0][1].device
    sync_device(device)
    empty_cache(device)
