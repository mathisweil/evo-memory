#!/usr/bin/env python3
"""Analysis 6 -- Token Importance Alignment (NAMM Scores vs Attention).

Measures alignment between NAMM's learned token importance scores and the
LLM's attention weights. For each model condition (M1, M3), we:
  1. Run a forward pass WITHOUT eviction to get full-context attention weights
     and KV cache.
  2. Call memory_policy.update_cache(analyze=True) on the full KV cache to
     extract NAMM token importance scores and eviction decisions.
  3. Compute Spearman rank correlation between NAMM scores and attention-based
     importance, plus eviction regret (attention mass on evicted tokens).

Usage:
    # Full run (GPU required):
    source activate.sh
    PYTHONPATH=. python analysis/report_6/generate_plots.py

    # Plot only from saved data (CPU ok):
    python analysis/report_6/generate_plots.py --plot-only
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
OUT_DIR = SCRIPT_DIR
DATA_FILE = OUT_DIR / "alignment_data.json"

# Checkpoint paths
ARTIFACTS = REPO_ROOT / "experiment_artifacts" / "gcs"
M1_LORA_CKPT = ARTIFACTS / "M1" / "best_ckpt.pt"
M3_LORA_CKPT = ARTIFACTS / "M3_cs1024" / "best_ckpt.pt"
M2_NAMM_CKPT = ARTIFACTS / "M2_cs1024" / "ckpt.pt"

RUN_CONFIG = "namm_bam_i1_llama32_1b_5t"
CACHE_SIZE = 1024
MAX_SEQ_LEN = 1024  # truncate inputs to avoid OOM with output_attentions
NUM_SAMPLES_PER_TASK = 3
TASKS = ["qasper", "2wikimqa", "qasper_e", "hotpotqa_e", "2wikimqa_e"]


def parse_args():
    p = argparse.ArgumentParser(description="Analysis 6: Token Importance Alignment")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--num-samples", type=int, default=NUM_SAMPLES_PER_TASK)
    return p.parse_args()


# ── Model setup ──────────────────────────────────────────────────────────────

def build_model_and_data(device="cuda"):
    """Build model infrastructure via Hydra config. Returns cfg, model objects."""
    import torch
    from scripts.experiment_utils import load_hydra_config
    from namm.run_utils import make_eval_model, make_task_sampler

    cfg = load_hydra_config(
        RUN_CONFIG,
        extra_overrides=[
            f"cache_size={CACHE_SIZE}",
            f"max_memory_length={CACHE_SIZE}",
        ],
    )

    with torch.no_grad():
        memory_policy, memory_model, memory_evaluator, _, _ = make_eval_model(cfg=cfg)

    memory_model.to(device)
    memory_model.eval()

    tokenizer = memory_evaluator.tokenizer

    # Build task sampler and apply splits
    task_sampler = make_task_sampler(cfg=cfg, train_split=0.7, split_seed=42)
    task_sampler.filter_by_token_count(tokenizer, 6500)
    task_sampler.filter_answers_by_token_count(tokenizer, 64)
    task_sampler.apply_train_val_test_split(
        train_frac=0.7, val_frac=0.15,
        max_conditioning_length=6500,
        min_conditioning_length=4096,
        tokenizer=tokenizer,
    )

    return cfg, memory_policy, memory_model, memory_evaluator, task_sampler, tokenizer


def load_namm_weights(memory_model, memory_policy, namm_ckpt_path, device="cuda"):
    """Load NAMM scoring network weights from checkpoint."""
    import torch

    ckpt = torch.load(namm_ckpt_path, map_location="cpu", weights_only=False)
    evo_state = ckpt["evolution_state"]

    params_vec = evo_state.get("mean", evo_state["best_member"])
    params = params_vec.unsqueeze(0).to(device)
    memory_model.set_memory_params(params)

    buffers_prefix = "stored_buffers_to_save."
    buffers_dict = {
        k[len(buffers_prefix):]: v.to(device)
        for k, v in evo_state.items()
        if k.startswith(buffers_prefix)
    }
    if buffers_dict:
        memory_model.load_buffers_dict(buffers_dict=buffers_dict)

    batch_idxs = np.zeros([1])
    memory_policy.set_params_batch_idxs(batch_idxs)
    print(f"  NAMM loaded ({params_vec.shape[0]} params)")


def load_lora_weights(memory_model, lora_ckpt_path, device="cuda"):
    """Load LoRA adapter weights into the wrapped memory model."""
    import torch

    ckpt = torch.load(lora_ckpt_path, map_location="cpu", weights_only=False)
    lora_cfg = ckpt.get("lora_config", {})
    lora_sd = ckpt["lora_state_dict"]

    # Apply LoRA adapters if not already present
    if not memory_model.has_lora_adapters():
        rank = lora_cfg.get("rank", 8)
        target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
        memory_model.apply_lora_adapters(rank=rank, target_modules=target_modules)

    # Load weights — checkpoint keys come from model.model.named_parameters()
    # which is the PeftModel wrapping LlamaMemoryModel
    loaded = 0
    for n, p in memory_model.model.named_parameters():
        if p.requires_grad and n in lora_sd:
            p.data.copy_(lora_sd[n].to(p.device))
            loaded += 1

    if loaded == 0:
        raise RuntimeError(
            f"No LoRA weights loaded from {lora_ckpt_path}! "
            "Key format mismatch between checkpoint and model."
        )
    print(f"  LoRA loaded ({loaded} tensors, "
          f"best_val={ckpt.get('best_val_score', '?')})")


def reset_lora_weights(memory_model):
    """Zero out LoRA weights to get base model behavior."""
    for n, p in memory_model.model.named_parameters():
        if p.requires_grad:
            p.data.zero_()


def get_test_prompts(task_sampler, tokenizer, num_per_task=NUM_SAMPLES_PER_TASK):
    """Get test prompts truncated to MAX_SEQ_LEN."""
    import torch

    test_idxs = task_sampler._test_idxs_per_task
    if test_idxs is None:
        raise RuntimeError("No test split available")

    prompts = []
    for task_name in sorted(test_idxs.keys()):
        idxs = test_idxs[task_name]
        task_prompts = task_sampler.lb_prompts_per_task[task_name]
        for idx in idxs[:num_per_task]:
            text = task_prompts[idx]
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=MAX_SEQ_LEN)
            prompts.append({
                "input_ids": ids["input_ids"],
                "task": task_name,
                "idx": int(idx),
                "seq_len": ids["input_ids"].shape[1],
            })
    print(f"  {len(prompts)} test prompts "
          f"(seq_len range: {min(p['seq_len'] for p in prompts)}-"
          f"{max(p['seq_len'] for p in prompts)})")
    return prompts


# ── Score extraction ─────────────────────────────────────────────────────────

import torch


def compute_spearman(x, y):
    """Spearman rank correlation."""
    from scipy.stats import spearmanr
    if len(x) < 3 or len(y) < 3:
        return float("nan")
    rho, _ = spearmanr(x, y)
    return float(rho)


def extract_alignment_data(memory_model, memory_policy, prompts, device="cuda"):
    """Extract NAMM scores and attention weights for each prompt.

    Strategy:
      1. Forward pass with apply_memory_policy=False, output_attentions=True
         → get full-context attention weights and KV cache
      2. Call memory_policy.update_cache(past_kv, analyze=True)
         → get NAMM token scores and eviction decisions (requires grad enabled
           because the analyze path computes Jacobians via autograd)
      3. Compute per-layer Spearman correlation and eviction regret
    """

    results = []

    for i, p in enumerate(prompts):
        input_ids = p["input_ids"].to(device)
        seq_len = input_ids.shape[1]

        # Reset memory policy state
        memory_policy.initialize_buffers()
        memory_policy.record_eval_stats = False

        # Pass 1: full-context forward (no eviction), no grad needed
        with torch.no_grad():
            outputs = memory_model(
                input_ids=input_ids,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
                apply_memory_policy=False,
            )

        # Extract per-KV-position mean attention received
        # attention shape per layer: (batch, n_heads, query_len, kv_len)
        attention_per_kv = []
        if outputs.attentions is not None:
            for layer_attn in outputs.attentions:
                # Average over heads and query positions → per-KV importance
                per_kv = layer_attn[0].float().mean(dim=0).mean(dim=0)  # (kv_len,)
                attention_per_kv.append(per_kv.cpu().numpy())

        # Pass 2: NAMM analyze on the full KV cache
        # The analyze path uses torch.autograd.grad internally, so we need
        # grad enabled. Detach KV cache tensors and re-enable requires_grad.
        past_kv = outputs.past_key_values
        # Convert DynamicCache to legacy tuple format if needed
        if hasattr(past_kv, 'to_legacy_cache'):
            past_kv = past_kv.to_legacy_cache()
        elif not isinstance(past_kv, tuple):
            past_kv = tuple(
                (past_kv.key_cache[j], past_kv.value_cache[j])
                for j in range(len(past_kv.key_cache))
            )

        attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # Reset policy for clean analyze pass
        memory_policy.initialize_buffers()

        # Analyze needs grad enabled for Jacobian computation
        try:
            evicted_kv, analysis_dicts = memory_policy.update_cache(
                past_key_values=past_kv,
                num_new_tokens=seq_len,
                attn_weights_list=outputs.attentions if memory_policy.requires_attn_scores else [],
                attention_mask=attention_mask,
                position_ids=position_ids,
                analyze=True,
            )
        except Exception as e:
            print(f"  Sample {i} analyze failed: {e}")
            del outputs
            torch.cuda.empty_cache()
            continue

        # Extract per-layer scores and eviction decisions
        sample_layers = []
        for layer_id, ad in enumerate(analysis_dicts):
            token_scores = ad.get("token_scores", None)
            retained_idxs = ad.get("retained_idxs", None)

            if token_scores is not None:
                # token_scores shape: (batch, n_heads, num_tokens)
                # Average across heads for per-token importance
                scores_np = token_scores[0].float().detach().mean(dim=0).cpu().numpy()
            else:
                scores_np = np.array([])

            if retained_idxs is not None:
                retained = retained_idxs[0, 0].detach().cpu().numpy().tolist()
            else:
                retained = list(range(seq_len))

            all_idxs = set(range(len(scores_np)))
            evicted = sorted(all_idxs - set(retained))

            # Spearman correlation with attention
            attn = attention_per_kv[layer_id] if layer_id < len(attention_per_kv) else np.array([])
            min_len = min(len(scores_np), len(attn))
            if min_len >= 3:
                rho = compute_spearman(scores_np[:min_len], attn[:min_len])
            else:
                rho = float("nan")

            # Eviction regret: attention mass on tokens NAMM would evict
            if len(evicted) > 0 and len(attn) > 0:
                evicted_valid = [e for e in evicted if e < len(attn)]
                total_regret = float(np.sum(attn[evicted_valid])) if evicted_valid else 0.0
                mean_regret = float(np.mean(attn[evicted_valid])) if evicted_valid else 0.0
            else:
                total_regret = 0.0
                mean_regret = 0.0

            sample_layers.append({
                "layer_id": layer_id,
                "spearman_rho": rho,
                "total_regret": total_regret,
                "mean_regret": mean_regret,
                "num_tokens": len(scores_np),
                "num_retained": len(retained),
                "num_evicted": len(evicted),
            })

        results.append({
            "task": p["task"],
            "idx": p["idx"],
            "seq_len": seq_len,
            "layers": sample_layers,
        })

        print(f"  Sample {i+1}/{len(prompts)} ({p['task']}) done")

        del outputs, past_kv
        torch.cuda.empty_cache()

    return results


# ── Main inference pipeline ──────────────────────────────────────────────────

def run_inference(args):
    """Run full pipeline on M1 and M3."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("GPU required. Use --plot-only on CPU.")

    # Validate checkpoints
    for path, name in [(M1_LORA_CKPT, "M1 LoRA"), (M3_LORA_CKPT, "M3 LoRA"),
                       (M2_NAMM_CKPT, "M2 NAMM")]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} checkpoint not found: {path}\n"
                "Run scripts/download_artifacts.py first.")

    print("=" * 60)
    print("Analysis 6: Token Importance Alignment")
    print("=" * 60)

    # Build model and data
    print("\nBuilding model infrastructure...")
    cfg, memory_policy, memory_model, evaluator, task_sampler, tokenizer = \
        build_model_and_data(device)

    print("\nPreparing test prompts...")
    prompts = get_test_prompts(task_sampler, tokenizer, args.num_samples)

    # Load NAMM weights (shared between M1 and M3 — same frozen NAMM)
    print("\nLoading NAMM weights...")
    load_namm_weights(memory_model, memory_policy, str(M2_NAMM_CKPT), device)

    results = {"cache_size": CACHE_SIZE, "conditions": {}}

    # ── M1: LoRA full-context weights + NAMM scoring ──
    print("\n" + "=" * 60)
    print("Condition: M1 (LoRA full-context) + NAMM scoring")
    print("=" * 60)
    load_lora_weights(memory_model, str(M1_LORA_CKPT), device)

    m1_results = extract_alignment_data(memory_model, memory_policy, prompts, device)
    results["conditions"]["M1"] = {"samples": m1_results}
    print(f"  M1: {len(m1_results)} samples extracted")

    # ── M3: LoRA eviction-aware weights + NAMM scoring ──
    print("\n" + "=" * 60)
    print("Condition: M3 cs1024 (LoRA + frozen NAMM)")
    print("=" * 60)
    # Reset LoRA, then load M3
    reset_lora_weights(memory_model)
    load_lora_weights(memory_model, str(M3_LORA_CKPT), device)

    m3_results = extract_alignment_data(memory_model, memory_policy, prompts, device)
    results["conditions"]["M3_cs1024"] = {"samples": m3_results}
    print(f"  M3: {len(m3_results)} samples extracted")

    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

def generate_plots(data):
    """Generate all plots from alignment data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    })

    conditions = data.get("conditions", {})
    if not conditions:
        print("WARNING: No data, skipping plots")
        return

    colors = {"M1": "#d62728", "M3_cs1024": "#1f77b4"}
    labels = {"M1": "M1 (LoRA full-context)", "M3_cs1024": "M3 (LoRA + frozen NAMM)"}

    # ── Plot 1: Spearman correlation by layer ────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    for cond_name, cond_data in conditions.items():
        layer_rhos = {}
        for sample in cond_data["samples"]:
            for ld in sample["layers"]:
                rho = ld["spearman_rho"]
                if not np.isnan(rho):
                    layer_rhos.setdefault(ld["layer_id"], []).append(rho)

        if not layer_rhos:
            continue

        layers = sorted(layer_rhos.keys())
        means = [np.mean(layer_rhos[l]) for l in layers]
        stds = [np.std(layer_rhos[l]) for l in layers]

        color = colors.get(cond_name, "#333")
        label = labels.get(cond_name, cond_name)
        ax.plot(layers, means, "-o", color=color, label=label,
                linewidth=2, markersize=5)
        ax.fill_between(layers,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=color)

    ax.axhline(0, color="grey", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman Rank Correlation")
    ax.set_title("NAMM Score vs Attention Importance Alignment\n"
                 "(Spearman rho per layer, averaged across test samples)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "score_attention_correlation.png")
    plt.close(fig)
    print("  Saved score_attention_correlation.png")

    # ── Plot 2: Eviction regret by layer ─────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for cond_name, cond_data in conditions.items():
        layer_total = {}
        layer_mean = {}
        for sample in cond_data["samples"]:
            for ld in sample["layers"]:
                layer_total.setdefault(ld["layer_id"], []).append(ld["total_regret"])
                layer_mean.setdefault(ld["layer_id"], []).append(ld["mean_regret"])

        layers = sorted(layer_total.keys())
        color = colors.get(cond_name, "#333")
        label = labels.get(cond_name, cond_name)

        ax1.plot(layers, [np.mean(layer_total[l]) for l in layers],
                 "-o", color=color, label=label, linewidth=2, markersize=5)
        ax2.plot(layers, [np.mean(layer_mean[l]) for l in layers],
                 "-o", color=color, label=label, linewidth=2, markersize=5)

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Total Attention Mass on Evicted Tokens")
    ax1.set_title("Eviction Regret (Total)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Mean Attention per Evicted Token")
    ax2.set_title("Eviction Regret (Per Token)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Eviction Regret: Attention Mass Lost to Evicted Tokens",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eviction_regret.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved eviction_regret.png")

    # ── Plot 3: Per-task alignment shift (M1 vs M3) ──────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    task_cond_rhos = {}
    for cond_name, cond_data in conditions.items():
        for sample in cond_data["samples"]:
            rhos = [ld["spearman_rho"] for ld in sample["layers"]
                    if not np.isnan(ld["spearman_rho"])]
            if rhos:
                task_cond_rhos.setdefault(sample["task"], {}).setdefault(
                    cond_name, []).append(np.mean(rhos))

    if task_cond_rhos:
        task_names = sorted(task_cond_rhos.keys())
        x = np.arange(len(task_names))
        width = 0.35
        cond_list = [c for c in ["M1", "M3_cs1024"] if c in conditions]

        for i, cond_name in enumerate(cond_list):
            vals = []
            errs = []
            for t in task_names:
                rho_list = task_cond_rhos.get(t, {}).get(cond_name, [])
                vals.append(np.mean(rho_list) if rho_list else 0.0)
                errs.append(np.std(rho_list) if rho_list else 0.0)

            color = colors.get(cond_name, "#333")
            label = labels.get(cond_name, cond_name)
            ax.bar(x + i * width, vals, width, yerr=errs,
                   label=label, color=color, edgecolor="white", capsize=3)

        task_display = {
            "qasper": "Qasper", "2wikimqa": "2WikiMQA",
            "qasper_e": "Qasper-E", "hotpotqa_e": "HotpotQA-E",
            "2wikimqa_e": "2WikiMQA-E",
        }
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([task_display.get(t, t) for t in task_names])

    ax.axhline(0, color="grey", linestyle=":", alpha=0.5)
    ax.set_ylabel("Mean Spearman rho (NAMM score vs attention)")
    ax.set_title("Alignment Shift: M1 vs M3\n"
                 "(Does M3 fine-tuning improve NAMM-attention alignment?)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alignment_shift.png")
    plt.close(fig)
    print("  Saved alignment_shift.png")

    # Print summary
    print("\n--- Summary ---")
    for cond_name, cond_data in conditions.items():
        all_rhos = []
        all_regrets = []
        for sample in cond_data["samples"]:
            for ld in sample["layers"]:
                if not np.isnan(ld["spearman_rho"]):
                    all_rhos.append(ld["spearman_rho"])
                all_regrets.append(ld["total_regret"])
        print(f"  {cond_name}:")
        print(f"    Mean Spearman rho: {np.mean(all_rhos):.4f} +/- {np.std(all_rhos):.4f}")
        print(f"    Mean total regret: {np.mean(all_regrets):.4f}")


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.plot_only:
        if not DATA_FILE.exists():
            print(f"ERROR: {DATA_FILE} not found. Run on GPU first.")
            return
        print(f"Loading data from {DATA_FILE}")
        with open(DATA_FILE) as f:
            data = json.load(f)
    else:
        data = run_inference(args)
        print(f"\nSaving data to {DATA_FILE}")
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)

    print("\nGenerating plots...")
    generate_plots(data)
    print("\nDone.")


if __name__ == "__main__":
    main()
