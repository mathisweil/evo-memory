"""Deterministic noise perturbation and ES weight update utilities.

Ported from:
- utils/worker_extn.py:23-47 (correlated mode)
- es_fine-tuning_conciseness_iid.py:110-115 (iid mode)
- es_fine-tuning_conciseness.py:291-310 (ES update)
"""

import torch

from .device import sync_device, empty_cache


def _get_param_dict(model, param_names):
    """Build a dict of {name: param} for the specified param names."""
    all_params = dict(model.named_parameters())
    return {name: all_params[name] for name in param_names}


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
    params = _get_param_dict(model, param_names)
    for i, (name, p) in enumerate(params.items()):
        effective_seed = int(seed) if mode == "correlated" else int(seed + i)
        noise = _make_noise(p.shape, p.dtype, p.device, effective_seed)
        p.data.add_(sigma * noise)
        del noise

    device = next(iter(params.values())).device
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
    params = _get_param_dict(model, param_names)
    for i, (name, p) in enumerate(params.items()):
        effective_seed = int(seed) if mode == "correlated" else int(seed + i)
        noise = _make_noise(p.shape, p.dtype, p.device, effective_seed)
        p.data.add_(-sigma * noise)
        del noise

    device = next(iter(params.values())).device
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
    params = _get_param_dict(model, param_names)
    for i, (name, p) in enumerate(params.items()):
        update = torch.zeros_like(p)

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
        del update

    device = next(iter(params.values())).device
    sync_device(device)
    empty_cache(device)
