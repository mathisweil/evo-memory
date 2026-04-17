"""Quick diagnostic: check per-layer token retention for a NAMM eval run.

Loads the model with record_eval_stats enabled and runs a small batch to
see how many tokens are actually retained per layer vs the cache_size budget.

Threshold-only mode (--cache_size 0): sets selection_criteria.cache_size=None
and uses the score threshold as the sole eviction criterion, equivalent to
training with `threshold_only=true` in run_namm.py. Diagnostic results from
--cache_size 0 are therefore directly comparable to threshold-mode training runs.
"""

import argparse
import os
import sys
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hydra import compose, initialize
from namm.run_utils import make_eval_model, make_task_sampler
from es_finetuning.device import get_device


def main():
    device = get_device()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_size", type=int, default=3072)
    ap.add_argument("--es_checkpoint", type=str, default="")
    ap.add_argument("--namm_checkpoint", type=str,
                    default="exp_local/pretrained/namm_pretrained_romain_v2.pt")
    ap.add_argument("--num_samples", type=int, default=10)
    cli = ap.parse_args()

    cache_size = cli.cache_size
    run_config = "namm_bam_i1_llama32_1b"
    namm_ckpt = cli.namm_checkpoint
    es_ckpt = cli.es_checkpoint

    # Build model
    overrides = [
        f"run@_global_={run_config}",
        "wandb_log=false",
        "wandb_project=Experiments",
    ]
    if cache_size > 0:
        overrides.append(f"cache_size={cache_size}")
        overrides.append(f"max_memory_length={cache_size}")
    else:
        # cache_size=0 means no cap on selection criteria's topk
        # Keep policy-level cache_size at 5120 for evaluator memory management
        overrides.append("cache_size=5120")
        overrides.append("max_memory_length=5120")
    with initialize(version_base=None, config_path="../config", job_name="eviction_check"):
        cfg = compose(config_name="config", overrides=overrides)

    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator,
         evolution_algorithm, auxiliary_loss) = make_eval_model(cfg=cfg)
    memory_model.to(device)
    memory_evaluator.device = device

    # Load NAMM weights
    print(f"Loading NAMM checkpoint: {namm_ckpt}")
    ckpt = torch.load(namm_ckpt, map_location="cpu", weights_only=False)
    evo_state = ckpt['evolution_state']
    best_member = evo_state['best_member']
    params = best_member.unsqueeze(0).to(device)
    memory_model.set_memory_params(params)

    buffers_prefix = 'stored_buffers_to_save.'
    buffers_dict = {
        k[len(buffers_prefix):]: v.to(device)
        for k, v in evo_state.items()
        if k.startswith(buffers_prefix)
    }
    if buffers_dict:
        memory_model.load_buffers_dict(buffers_dict=buffers_dict)
    print(f"  NAMM loaded ({best_member.shape[0]} params)")

    # Load ES checkpoint (optional)
    if es_ckpt:
        print(f"Loading ES checkpoint: {es_ckpt}")
        es_state = torch.load(es_ckpt, map_location="cpu", weights_only=False)
        is_delta = es_state.pop("__format__", None) == "delta"
        model_params = dict(memory_model.named_parameters())
        loaded = 0
        for name, val in es_state.items():
            if name in model_params:
                param_dev = model_params[name].device
                if is_delta:
                    model_params[name].data.add_(val.to(param_dev))
                else:
                    model_params[name].data.copy_(val.to(param_dev))
                loaded += 1
        print(f"  Loaded {loaded}/{len(es_state)} ES-tuned parameters ({('delta' if is_delta else 'absolute')} format)")
    else:
        print("No ES checkpoint — using base LLM weights")

    batch_idxs = np.zeros([1])
    memory_policy.set_params_batch_idxs(batch_idxs)

    # If cache_size=0, null out the selection criteria's cache_size so threshold is the only filter
    if cache_size == 0:
        if hasattr(memory_policy, 'selection_criteria'):
            memory_policy.selection_criteria.cache_size = None
            print("Selection criteria cache_size set to None (threshold-only mode)")

    # Enable per-layer stats recording
    memory_policy.record_eval_stats = True
    memory_policy.record_mask_based_sparsity = True
    memory_policy.initialize_stat_objects()
    effective = cache_size if cache_size > 0 else "None (threshold only)"
    print(f"\nStats recording enabled. cache_size budget = {effective}")

    # Create task sampler with a few samples
    task_sampler = make_task_sampler(cfg=cfg, train_split=0.9, split_seed=42)
    task_sampler.filter_answers_by_token_count(memory_evaluator.tokenizer, 64)

    print(f"\nRunning eval on a few samples to collect eviction stats...\n")

    with torch.no_grad():
        score_dicts = task_sampler.evaluate(
            lm=memory_evaluator,
            train=False,
            evolved_model=False,
            pop_reps=1,
            resample_requests=True,
            sampled_requests_per_task=cli.num_samples,
            model_kwargs={},
        )

    # Retrieve and print per-layer stats
    stats = memory_policy.get_param_stats(reset=False)

    print("=" * 70)
    print(f"PER-LAYER TOKEN RETENTION STATS (cache_size budget = {cache_size})")
    print("=" * 70)

    num_layers = memory_policy.num_memory_layers
    print(f"\n{'Layer':<8} {'Avg Cache Size':<16} {'Final Cache Size':<18} {'Unmasked Samples':<18} {'Unmasked/Head':<15}")
    print("-" * 75)

    for i in range(num_layers):
        prefix = f'mem_stats/layer_id_{i}/'
        avg_cs = stats.get(prefix + 'dynamic_cache_sizes', 'N/A')
        final_cs = stats.get(prefix + 'final_dynamic_cache_sizes', 'N/A')
        unmasked = stats.get(prefix + 'unmasked_samples', 'N/A')
        unmasked_head = stats.get(prefix + 'unmasked_samples_per_head', 'N/A')

        def fmt(v):
            return f"{v:.1f}" if isinstance(v, (int, float, np.floating)) else str(v)

        print(f"  {i:<6} {fmt(avg_cs):<16} {fmt(final_cs):<18} {fmt(unmasked):<18} {fmt(unmasked_head):<15}")

    print("-" * 75)
    overall = 'mem_stats/overall/'
    print(f"  {'ALL':<6} "
          f"{stats.get(overall + 'dynamic_cache_sizes', 'N/A'):<16.1f} "
          f"{stats.get(overall + 'final_dynamic_cache_sizes', 'N/A'):<18.1f} "
          f"{stats.get(overall + 'unmasked_sample_final', 'N/A'):<18.1f} "
          f"{stats.get(overall + 'unmasked_sample_per_head_final', 'N/A'):<15.1f}")

    # Also dump raw per-layer dynamic_cache_sizes lists for detail
    print(f"\n\nRAW per-layer final cache sizes (last generation step per sequence):")
    for i in range(num_layers):
        sizes = memory_policy.dynamic_cache_sizes[i]
        finals = memory_policy.final_dynamic_cache_sizes[i]
        if finals:
            print(f"  Layer {i}: finals={finals[-5:]}")  # last 5
        elif sizes:
            print(f"  Layer {i}: last_size={sizes[-1]}")

    print(f"\nScore: {score_dicts}")


if __name__ == "__main__":
    main()
