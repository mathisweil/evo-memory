#!/usr/bin/env python3
"""
Analysis 5 -- Attention Entropy Shift Under Eviction
=====================================================
Compares attention distributions between M1 (full-context LoRA) and
M3 cs1024 (LoRA + frozen NAMM eviction) to study how eviction changes
the model's attention patterns.

Produces three plots in analysis/report_5/:
  attention_entropy.png   -- per-layer mean attention entropy (M1 vs M3)
  attention_sinks.png     -- attention mass on first-5 tokens per layer
  entropy_by_layer.png    -- per-head entropy heatmap for each condition

Requires GPU inference to extract attention weights.  Run with
``--plot-only`` to skip inference and regenerate plots from saved data.

Usage:
    # Full run (GPU required):
    python analysis/report_5/generate_plots.py \
        --namm_checkpoint exp_local/pretrained/namm_pretrained_romain.pt

    # Plot-only (CPU, after data has been extracted):
    python analysis/report_5/generate_plots.py --plot-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# ── Repo setup ──────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = SCRIPT_DIR
DATA_FILE = OUT_DIR / "attention_data.json"

# ── Checkpoint paths (relative to REPO_ROOT) ───────────────────────────────

M1_LORA_CKPT = REPO_ROOT / "experiment_artifacts" / "gcs" / "M1" / "best_ckpt.pt"
M3_LORA_CKPT = REPO_ROOT / "experiment_artifacts" / "gcs" / "M3_cs1024" / "best_ckpt.pt"
M2_NAMM_CKPT = REPO_ROOT / "experiment_artifacts" / "gcs" / "M2_cs1024" / "ckpt.pt"

# ── Model / data config ────────────────────────────────────────────────────

RUN_CONFIG = "namm_bam_i1_llama32_1b_5t"
CACHE_SIZE = 1024
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
SPLIT_SEED = 42
FILTER_BY_TOKENS = 6500
FILTER_ANSWERS_BY_TOKENS = 64
NUM_LAYERS = 16   # LLaMA 3.2-1B has 16 layers
NUM_HEADS = 32    # 32 attention heads (8 KV heads with GQA, expanded to 32)
SINK_TOKENS = 5   # number of initial tokens for sink analysis
MAX_SAMPLES = 20  # limit test samples for tractable attention extraction


# ── Data extraction (GPU) ──────────────────────────────────────────────────

def load_model_and_data(args):
    """Load the base model, task sampler, and evaluator via Hydra config.

    Follows the same pattern as scripts/run_eval.py and scripts/run_lora.py.
    """
    import torch
    from hydra import compose, initialize
    from namm.run_utils import make_eval_model, make_task_sampler
    from es_finetuning.device import get_device
    from experiment_utils import load_hydra_config

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = get_device()
    print(f"Device: {device}")

    # Load Hydra config
    cfg = load_hydra_config(
        RUN_CONFIG,
        extra_overrides=[
            f"cache_size={CACHE_SIZE}",
            f"max_memory_length={CACHE_SIZE}",
        ],
    )

    # Build model
    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator,
         evolution_algorithm, auxiliary_loss) = make_eval_model(cfg=cfg)

    # Create task sampler with the same split used during training
    task_sampler = make_task_sampler(
        cfg=cfg, train_split=TRAIN_SPLIT, split_seed=SPLIT_SEED)

    # Token-based filtering to match training setup
    task_sampler.filter_by_token_count(
        memory_evaluator.tokenizer, FILTER_BY_TOKENS)
    task_sampler.filter_answers_by_token_count(
        memory_evaluator.tokenizer, FILTER_ANSWERS_BY_TOKENS)

    # Apply the 3-way train/val/test split
    task_sampler.apply_train_val_test_split(
        train_frac=TRAIN_SPLIT,
        val_frac=VAL_SPLIT,
        max_conditioning_length=cfg.get('max_conditioning_length', 6500),
        min_conditioning_length=cfg.get('min_conditioning_length', None),
        tokenizer=memory_evaluator.tokenizer,
    )

    return (cfg, device, memory_policy, memory_model, memory_evaluator,
            task_sampler)


def load_namm_weights(memory_model, memory_policy, device, namm_ckpt_path,
                      prefer_mean=True):
    """Load NAMM scoring network weights from a checkpoint.

    Follows the same pattern as scripts/run_lora.py lines 219-249.
    """
    import torch

    print(f"Loading NAMM checkpoint: {namm_ckpt_path}")
    ckpt = torch.load(namm_ckpt_path, map_location="cpu", weights_only=False)
    evo_state = ckpt['evolution_state']

    if prefer_mean and 'mean' in evo_state:
        params_vec = evo_state['mean']
        param_source = "mean"
    else:
        params_vec = evo_state['best_member']
        param_source = "best_member"

    params = params_vec.unsqueeze(0).to(device)
    memory_model.set_memory_params(params)

    buffers_prefix = 'stored_buffers_to_save.'
    buffers_dict = {
        k[len(buffers_prefix):]: v.to(device)
        for k, v in evo_state.items()
        if k.startswith(buffers_prefix)
    }
    if buffers_dict:
        memory_model.load_buffers_dict(buffers_dict=buffers_dict)

    batch_idxs = np.zeros([1])
    memory_policy.set_params_batch_idxs(batch_idxs)
    print(f"  Loaded NAMM {param_source} ({params_vec.shape[0]} params)")


def load_lora_weights(memory_model, lora_ckpt_path, device):
    """Load LoRA adapter weights from a checkpoint.

    Follows the pattern from grad_lora_finetuning/trainer.py.
    The checkpoint contains lora_state_dict and lora_config.
    """
    import torch

    print(f"Loading LoRA checkpoint: {lora_ckpt_path}")
    ckpt = torch.load(lora_ckpt_path, map_location="cpu", weights_only=False)

    # Apply LoRA adapters if not already present
    if not memory_model.has_lora_adapters():
        lora_cfg = ckpt.get('lora_config', {})
        rank = lora_cfg.get('rank', 4)
        target_modules = lora_cfg.get('target_modules', ['q_proj', 'v_proj'])
        memory_model.apply_lora_adapters(
            rank=rank, target_modules=target_modules)

    # Load LoRA weights
    lora_state = ckpt['lora_state_dict']
    loaded = 0
    for n, p in memory_model.model.named_parameters():
        if p.requires_grad and n in lora_state:
            p.data.copy_(lora_state[n].to(p.device))
            loaded += 1
    print(f"  Loaded {loaded} LoRA tensors (best_step={ckpt.get('best_step', '?')}, "
          f"best_val={ckpt.get('best_val_score', '?')})")


def reset_lora_weights(memory_model):
    """Zero out LoRA adapter weights to reset to base model behaviour.

    After calling this, the LoRA adapters produce zero output (identity),
    so the model behaves as if no LoRA were applied.
    """
    import torch

    for n, p in memory_model.model.named_parameters():
        if p.requires_grad:
            p.data.zero_()
    print("  Reset LoRA weights to zero (identity)")


def get_test_inputs(task_sampler, tokenizer, max_samples=MAX_SAMPLES):
    """Get tokenised test-set inputs for attention extraction.

    Returns a list of dicts, each with 'input_ids' and 'task' keys.
    """
    import torch

    test_inputs = []
    test_idxs = task_sampler._test_idxs_per_task
    if test_idxs is None:
        print("WARNING: No test split available, using first samples")
        return []

    for task_name in sorted(test_idxs.keys()):
        idxs = test_idxs[task_name]
        prompts = task_sampler.lb_prompts_per_task[task_name]
        for idx in idxs[:max_samples]:
            prompt = prompts[idx]
            tokens = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=FILTER_BY_TOKENS)
            test_inputs.append({
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens.get("attention_mask"),
                "task": task_name,
                "idx": int(idx),
                "seq_len": tokens["input_ids"].shape[1],
            })

    # Sort by sequence length (shorter first) for efficient batching
    test_inputs.sort(key=lambda x: x["seq_len"])

    # Limit total samples
    if len(test_inputs) > max_samples:
        # Sample uniformly from the sorted list
        step = len(test_inputs) / max_samples
        selected = [test_inputs[int(i * step)]
                    for i in range(max_samples)]
        test_inputs = selected

    print(f"Prepared {len(test_inputs)} test inputs "
          f"(seq_len range: {test_inputs[0]['seq_len']}-{test_inputs[-1]['seq_len']})")
    return test_inputs


def extract_attention_weights(memory_model, test_inputs, device,
                              apply_memory_policy=True):
    """Run forward passes and extract attention weights.

    Args:
        memory_model: The WrappedLlamaForCausalLM model.
        test_inputs: List of dicts with 'input_ids' and 'attention_mask'.
        device: torch device.
        apply_memory_policy: If True, run with NAMM eviction active.

    Returns:
        List of dicts, one per sample, containing:
          - entropies: [num_layers, num_heads] mean entropy per head
          - sink_fractions: [num_layers, num_heads] attention mass on first-k tokens
          - task: task name
          - seq_len: sequence length
    """
    import torch

    memory_model.eval()
    results = []

    for i, inp in enumerate(test_inputs):
        input_ids = inp["input_ids"].to(device)
        attention_mask = inp.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        try:
            # Reset memory policy state for each sample
            memory_model.memory_policy.reset()

            with torch.no_grad():
                outputs = memory_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    apply_memory_policy=apply_memory_policy,
                    use_cache=True,
                )

            # outputs.attentions is a tuple of (num_layers,) tensors
            # Each tensor shape: (batch, num_heads, seq_len, seq_len) for
            # the final chunk, or (batch, num_heads, chunk_len, cache_len)
            # when split processing is used.
            attentions = outputs.attentions
            if attentions is None:
                print(f"  Sample {i}: no attentions returned, skipping")
                continue

            sample_entropies = []
            sample_sink_fracs = []

            for layer_idx, attn in enumerate(attentions):
                # attn shape: (batch, num_heads, q_len, kv_len)
                attn = attn.float()  # ensure float32 for log
                # Use the last query position's attention as representative
                # (this is the position that will generate the answer)
                attn_last = attn[0, :, -1, :]  # (num_heads, kv_len)

                # Clamp to avoid log(0)
                attn_last = attn_last.clamp(min=1e-12)

                # Shannon entropy: H = -sum(a * log(a))
                entropy = -(attn_last * torch.log(attn_last)).sum(dim=-1)
                # entropy shape: (num_heads,)

                # Attention sink: fraction of mass on first SINK_TOKENS tokens
                kv_len = attn_last.shape[-1]
                sink_end = min(SINK_TOKENS, kv_len)
                sink_frac = attn_last[:, :sink_end].sum(dim=-1)
                # sink_frac shape: (num_heads,)

                sample_entropies.append(entropy.cpu().numpy().tolist())
                sample_sink_fracs.append(sink_frac.cpu().numpy().tolist())

            results.append({
                "entropies": sample_entropies,
                "sink_fractions": sample_sink_fracs,
                "task": inp["task"],
                "seq_len": inp["seq_len"],
                "idx": inp["idx"],
            })

            if (i + 1) % 5 == 0 or i == 0:
                print(f"  Processed {i+1}/{len(test_inputs)} samples")

        except Exception as e:
            print(f"  Sample {i} failed: {e}")
            continue

        finally:
            # Free GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def run_extraction(args):
    """Run the full extraction pipeline: load models, extract attention data."""
    import torch

    # Validate checkpoint paths
    if not M1_LORA_CKPT.exists():
        print(f"ERROR: M1 checkpoint not found: {M1_LORA_CKPT}")
        print("Run scripts/download_artifacts.py to download checkpoints.")
        sys.exit(1)
    if not M3_LORA_CKPT.exists():
        print(f"ERROR: M3 checkpoint not found: {M3_LORA_CKPT}")
        sys.exit(1)

    namm_ckpt = args.namm_checkpoint
    if namm_ckpt is None:
        # Fall back to M2 cs1024 checkpoint which contains NAMM weights
        namm_ckpt = str(M2_NAMM_CKPT)
    if not Path(namm_ckpt).exists():
        print(f"ERROR: NAMM checkpoint not found: {namm_ckpt}")
        sys.exit(1)

    # Load model infrastructure
    print("=" * 60)
    print("Analysis 5: Attention Entropy Shift Under Eviction")
    print("=" * 60)
    print()

    (cfg, device, memory_policy, memory_model, memory_evaluator,
     task_sampler) = load_model_and_data(args)

    tokenizer = memory_evaluator.tokenizer

    # Get test inputs
    print("\nPreparing test inputs...")
    test_inputs = get_test_inputs(task_sampler, tokenizer,
                                  max_samples=args.max_samples)
    if not test_inputs:
        print("ERROR: No test inputs available")
        sys.exit(1)

    all_data = {}

    # ── M1: Full-context LoRA (no NAMM eviction) ───────────────────────
    print("\n" + "=" * 60)
    print("Extracting attention weights: M1 (full-context LoRA)")
    print("=" * 60)

    # For M1, swap to Recency (passthrough) policy -- no eviction
    from namm.policy.base import Recency
    recency_policy = Recency(cache_size=None)  # no eviction limit
    memory_evaluator.swap_memory_policy(recency_policy)
    memory_evaluator.max_memory_length = memory_evaluator.max_conditioning_length

    # Apply LoRA adapters and load M1 weights
    memory_model.apply_lora_adapters(rank=4, target_modules=['q_proj', 'v_proj'])
    load_lora_weights(memory_model, str(M1_LORA_CKPT), device)
    memory_model.to(dtype=torch.bfloat16, device=device)

    m1_results = extract_attention_weights(
        memory_model, test_inputs, device, apply_memory_policy=False)
    all_data["M1"] = m1_results
    print(f"M1: extracted {len(m1_results)} samples")

    # ── M3 cs1024: LoRA + frozen NAMM eviction ─────────────────────────
    print("\n" + "=" * 60)
    print("Extracting attention weights: M3 cs1024 (LoRA + NAMM)")
    print("=" * 60)

    # Rebuild the model from scratch for M3 so we get a fresh NAMM policy.
    # Reuse the same Hydra config (cfg) to avoid calling initialize() twice.
    del memory_model
    del memory_evaluator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nReloading model for M3...")
    from namm.run_utils import make_eval_model

    with torch.no_grad():
        (memory_policy_m3, memory_model_m3, memory_evaluator_m3,
         _, _) = make_eval_model(cfg=cfg)

    # Load NAMM weights
    load_namm_weights(memory_model_m3, memory_policy_m3, device,
                      namm_ckpt)

    # Apply LoRA and load M3 weights
    memory_model_m3.apply_lora_adapters(
        rank=4, target_modules=['q_proj', 'v_proj'])
    load_lora_weights(memory_model_m3, str(M3_LORA_CKPT), device)
    memory_model_m3.to(dtype=torch.bfloat16, device=device)

    m3_results = extract_attention_weights(
        memory_model_m3, test_inputs, device, apply_memory_policy=True)
    all_data["M3_cs1024"] = m3_results
    print(f"M3 cs1024: extracted {len(m3_results)} samples")

    # ── Save intermediate data ──────────────────────────────────────────
    print(f"\nSaving attention data to {DATA_FILE}...")
    with open(DATA_FILE, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"  Saved ({DATA_FILE.stat().st_size / 1024:.1f} KB)")

    return all_data


# ── Plotting (CPU) ──────────────────────────────────────────────────────────

def load_saved_data():
    """Load previously extracted attention data."""
    if not DATA_FILE.exists():
        print(f"ERROR: Data file not found: {DATA_FILE}")
        print("Run without --plot-only first to extract attention data.")
        sys.exit(1)
    with open(DATA_FILE) as f:
        data = json.load(f)
    print(f"Loaded attention data from {DATA_FILE}")
    for cond, samples in data.items():
        print(f"  {cond}: {len(samples)} samples")
    return data


def compute_stats(data):
    """Compute per-layer, per-head statistics from raw attention data.

    Returns dicts mapping condition -> arrays of shape (num_layers, num_heads).
    """
    stats = {}
    for cond, samples in data.items():
        if not samples:
            continue

        all_entropies = []
        all_sinks = []
        for s in samples:
            ent = np.array(s["entropies"])   # (num_layers, num_heads)
            snk = np.array(s["sink_fractions"])  # (num_layers, num_heads)
            all_entropies.append(ent)
            all_sinks.append(snk)

        # Stack and compute mean/std across samples
        ent_stack = np.stack(all_entropies)  # (n_samples, n_layers, n_heads)
        snk_stack = np.stack(all_sinks)

        stats[cond] = {
            "entropy_mean": ent_stack.mean(axis=0),  # (n_layers, n_heads)
            "entropy_std": ent_stack.std(axis=0),
            "entropy_layer_mean": ent_stack.mean(axis=(0, 2)),  # (n_layers,)
            "entropy_layer_std": ent_stack.std(axis=(0, 2)),
            "sink_mean": snk_stack.mean(axis=0),
            "sink_std": snk_stack.std(axis=0),
            "sink_layer_mean": snk_stack.mean(axis=(0, 2)),
            "sink_layer_std": snk_stack.std(axis=(0, 2)),
            "n_samples": len(samples),
        }

    return stats


def plot_attention_entropy(stats, out_dir):
    """Plot 1: Per-layer mean attention entropy for M1 vs M3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"M1": "#d62728", "M3_cs1024": "#1f77b4"}
    labels = {"M1": "M1 (full context)", "M3_cs1024": "M3 cs1024 (evicted)"}
    linestyles = {"M1": "-", "M3_cs1024": "--"}

    layers = np.arange(NUM_LAYERS)

    for cond in ["M1", "M3_cs1024"]:
        if cond not in stats:
            continue
        s = stats[cond]
        mean = s["entropy_layer_mean"]
        std = s["entropy_layer_std"]

        ax.plot(layers, mean,
                label=labels.get(cond, cond),
                color=colors.get(cond, "black"),
                linestyle=linestyles.get(cond, "-"),
                linewidth=2, marker="o", markersize=4)
        ax.fill_between(layers, mean - std, mean + std,
                        alpha=0.15, color=colors.get(cond, "black"))

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Attention Entropy (nats)", fontsize=12)
    ax.set_title("Per-Layer Mean Attention Entropy\n"
                 "H = -sum(a_i * log(a_i)) at last query position",
                 fontsize=13)
    ax.set_xticks(layers)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = out_dir / "attention_entropy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_attention_sinks(stats, out_dir):
    """Plot 2: Attention sink fraction (mass on first-5 tokens) per layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"M1": "#d62728", "M3_cs1024": "#1f77b4"}
    labels = {"M1": "M1 (full context)", "M3_cs1024": "M3 cs1024 (evicted)"}
    linestyles = {"M1": "-", "M3_cs1024": "--"}

    layers = np.arange(NUM_LAYERS)

    for cond in ["M1", "M3_cs1024"]:
        if cond not in stats:
            continue
        s = stats[cond]
        mean = s["sink_layer_mean"]
        std = s["sink_layer_std"]

        ax.plot(layers, mean,
                label=labels.get(cond, cond),
                color=colors.get(cond, "black"),
                linestyle=linestyles.get(cond, "-"),
                linewidth=2, marker="s", markersize=4)
        ax.fill_between(layers, mean - std, mean + std,
                        alpha=0.15, color=colors.get(cond, "black"))

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel(f"Attention Mass on First {SINK_TOKENS} Tokens", fontsize=12)
    ax.set_title(f"Attention Sink Analysis\n"
                 f"Fraction of attention on first {SINK_TOKENS} tokens "
                 f"(BOS / system prompt)", fontsize=13)
    ax.set_xticks(layers)
    ax.set_ylim(0, None)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = out_dir / "attention_sinks.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_entropy_by_layer(stats, out_dir):
    """Plot 3: Per-head entropy heatmap for each condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = [c for c in ["M1", "M3_cs1024"] if c in stats]
    if not conditions:
        print("  No data for entropy heatmap, skipping")
        return

    n_conds = len(conditions)
    fig, axes = plt.subplots(1, n_conds, figsize=(7 * n_conds, 6),
                              squeeze=False)
    cond_labels = {"M1": "M1 (full context)",
                   "M3_cs1024": "M3 cs1024 (evicted)"}

    # Find global min/max for consistent colorbar
    all_means = [stats[c]["entropy_mean"] for c in conditions]
    vmin = min(m.min() for m in all_means)
    vmax = max(m.max() for m in all_means)

    for i, cond in enumerate(conditions):
        ax = axes[0, i]
        ent_mean = stats[cond]["entropy_mean"]  # (n_layers, n_heads)

        im = ax.imshow(ent_mean, aspect="auto", cmap="viridis",
                       vmin=vmin, vmax=vmax, origin="lower")
        ax.set_xlabel("Head", fontsize=11)
        ax.set_ylabel("Layer", fontsize=11)
        ax.set_title(cond_labels.get(cond, cond), fontsize=12)
        ax.set_xticks(range(0, ent_mean.shape[1], 4))
        ax.set_yticks(range(ent_mean.shape[0]))

    fig.suptitle("Per-Head Attention Entropy by Layer\n"
                 "(mean across test samples)", fontsize=13, y=1.02)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Entropy (nats)",
                 shrink=0.8)
    fig.tight_layout()

    path = out_dir / "entropy_by_layer.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def generate_plots(data):
    """Generate all plots from extracted attention data."""
    import matplotlib
    matplotlib.use("Agg")

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\nComputing statistics...")
    stats = compute_stats(data)

    for cond, s in stats.items():
        print(f"\n  {cond} ({s['n_samples']} samples):")
        print(f"    Mean entropy across layers: "
              f"{s['entropy_layer_mean'].mean():.3f} +/- "
              f"{s['entropy_layer_std'].mean():.3f}")
        print(f"    Mean sink fraction across layers: "
              f"{s['sink_layer_mean'].mean():.4f} +/- "
              f"{s['sink_layer_std'].mean():.4f}")

    print("\nGenerating plots...")
    plot_attention_entropy(stats, OUT_DIR)
    plot_attention_sinks(stats, OUT_DIR)
    plot_entropy_by_layer(stats, OUT_DIR)

    # Save summary stats for report
    summary = {}
    for cond, s in stats.items():
        summary[cond] = {
            "n_samples": s["n_samples"],
            "mean_entropy": float(s["entropy_layer_mean"].mean()),
            "mean_entropy_std": float(s["entropy_layer_std"].mean()),
            "mean_sink_fraction": float(s["sink_layer_mean"].mean()),
            "mean_sink_std": float(s["sink_layer_std"].mean()),
            "per_layer_entropy": s["entropy_layer_mean"].tolist(),
            "per_layer_sink": s["sink_layer_mean"].tolist(),
        }
    summary_path = OUT_DIR / "summary_stats.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary stats saved to {summary_path}")

    print("\nDone. All plots saved to:", OUT_DIR)


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analysis 5: Attention Entropy Shift Under Eviction")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip inference; generate plots from saved data")
    parser.add_argument("--namm_checkpoint", type=str, default=None,
                        help="Path to NAMM checkpoint. Defaults to "
                             "experiment_artifacts/gcs/M2_cs1024/ckpt.pt")
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES,
                        help=f"Max test samples for attention extraction "
                             f"(default: {MAX_SAMPLES})")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.plot_only:
        data = load_saved_data()
    else:
        data = run_extraction(args)

    generate_plots(data)


if __name__ == "__main__":
    main()
