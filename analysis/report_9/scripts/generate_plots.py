#!/usr/bin/env python3
"""Analysis 9 -- Gradient Flow and Loss Attribution Under Eviction (plots).

Loads pre-computed gradient data from data/maskfix_gradient_data.json and generates:
  plots/loss_stratified.png             -- Box plot of per-sample loss by retention stratum
  plots/grad_norms.png                  -- Per-layer LoRA gradient L2 norms (evicted vs full)

  plots/grad_direction_consistency.png  -- Per-layer cosine similarity of gradient directions

Only M1 vs M3 data is plotted (mask-fixed run). No buggy data is used.

Runnable without a GPU:
    PYTHONPATH=. .venv/bin/python analysis/report_9/scripts/generate_plots.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG -- how to regenerate the data
# ---------------------------------------------------------------------------
# Data file produced by analysis/report_9/scripts/generate_data.py on a GPU node.
# That script loads the following checkpoints:
#   M3 LoRA   : experiment_artifacts/gcs/M3_cs1024/best_ckpt.pt
#   M2 NAMM   : experiment_artifacts/gcs/M2_cs1024/ckpt.pt
#                (frozen NAMM scoring network, shared between M2 and M3)
# and performs instrumented forward+backward passes over training data under
# two conditions: (1) with NAMM eviction at cache_size=1024, and (2) without
# eviction (full context). Per-sample CE loss on answer tokens and per-layer
# LoRA gradient norms are recorded.
#
# To regenerate the data:
#   PYTHONPATH=. .venv/bin/python analysis/report_9/scripts/generate_data.py
#
# JSON structure:
#   evicted:               list[dict] -- per-sample results under eviction
#     each dict has: loss, per_layer_grad_norms (dict[str,float]),
#                    retention_ratio, n_answer_tokens, task, idx, seq_len
#   full_context:          list[dict] -- per-sample results without eviction
#     (same keys as evicted, retention_ratio is always 1.0)
#   cosine_sims_per_layer: dict[str, list[float]] -- gradient cosine similarities
#     keyed by layer index (as string), each value is a list of per-param
#     cosine similarities between evicted and full-context gradients
#   cache_size:            int
#   max_samples:           int
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent
DATA_FILE = REPORT_DIR / "data" / "maskfix_gradient_data.json"
OUT_DIR = REPORT_DIR / "plots"

NUM_LAYERS = 16  # LLaMA 3.2-1B decoder layers


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_loss_stratified(data: dict[str, Any], out_dir: Path) -> None:
    """Box plot of per-sample loss, stratified by retention ratio."""
    evicted = data["evicted"]
    full = data["full_context"]

    if not evicted or not full:
        print("  Skipping loss_stratified (no data)")
        return

    retention_ratios = [s["retention_ratio"] for s in evicted]
    median_retention = float(np.median(retention_ratios))

    high_ret_losses = [
        s["loss"] for s in evicted if s["retention_ratio"] >= median_retention
    ]
    low_ret_losses = [
        s["loss"] for s in evicted if s["retention_ratio"] < median_retention
    ]
    full_losses = [s["loss"] for s in full]

    fig, ax = plt.subplots(figsize=(10, 6))

    box_data = [full_losses, high_ret_losses, low_ret_losses]
    box_labels = [
        f"Full context\n(n={len(full_losses)})",
        f"Evicted, high retention\n(>= {median_retention:.3f}, n={len(high_ret_losses)})",
        f"Evicted, low retention\n(< {median_retention:.3f}, n={len(low_ret_losses)})",
    ]
    colors = ["#2ca02c", "#1f77b4", "#d62728"]

    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points
    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(box_data, colors)):
        jitter = rng.normal(0, 0.04, size=len(vals))
        ax.scatter(
            np.ones(len(vals)) * (i + 1) + jitter,
            vals, alpha=0.4, s=15, color=color, zorder=3,
        )

    ax.set_ylabel("Cross-Entropy Loss (answer tokens)", fontsize=12)
    ax.set_title(
        "Per-Sample Loss Stratified by Retention Ratio\n"
        f"(median retention = {median_retention:.3f}, "
        f"cache_size={data.get('cache_size', '?')})",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    path = out_dir / "loss_stratified.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_grad_norms(data: dict[str, Any], out_dir: Path) -> None:
    """Per-layer LoRA gradient L2 norms, evicted vs full context."""
    evicted = data["evicted"]
    full = data["full_context"]

    if not evicted or not full:
        print("  Skipping grad_norms (no data)")
        return

    evicted_layer_norms: dict[int, list[float]] = {}
    full_layer_norms: dict[int, list[float]] = {}

    for s in evicted:
        for layer_str, norm_val in s["per_layer_grad_norms"].items():
            evicted_layer_norms.setdefault(int(layer_str), []).append(norm_val)

    for s in full:
        for layer_str, norm_val in s["per_layer_grad_norms"].items():
            full_layer_norms.setdefault(int(layer_str), []).append(norm_val)

    layers = sorted(set(evicted_layer_norms.keys()) | set(full_layer_norms.keys()))
    if not layers:
        print("  Skipping grad_norms (no layer data)")
        return

    ev_means = [np.mean(evicted_layer_norms.get(l, [0])) for l in layers]
    ev_stds = [np.std(evicted_layer_norms.get(l, [0])) for l in layers]
    fu_means = [np.mean(full_layer_norms.get(l, [0])) for l in layers]
    fu_stds = [np.std(full_layer_norms.get(l, [0])) for l in layers]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(layers))
    width = 0.35

    ax.bar(x - width / 2, ev_means, width, yerr=ev_stds,
           label="Evicted (NAMM cs1024)", color="#d62728",
           edgecolor="white", capsize=3, alpha=0.8)
    ax.bar(x + width / 2, fu_means, width, yerr=fu_stds,
           label="Full context", color="#1f77b4",
           edgecolor="white", capsize=3, alpha=0.8)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("LoRA Gradient L2 Norm", fontsize=12)
    ax.set_title(
        "Per-Layer LoRA Gradient Norms\n"
        "(evicted vs full context, M3 checkpoint)",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    path = out_dir / "grad_norms.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")



def plot_grad_direction_consistency(data: dict[str, Any], out_dir: Path) -> None:
    """Per-layer cosine similarity between evicted and full-context gradients."""
    cosine_sims = data.get("cosine_sims_per_layer", {})
    if not cosine_sims:
        print("  Skipping grad_direction_consistency (no data)")
        return

    layer_data: list[tuple[int, list[float]]] = []
    for layer_str, sims in cosine_sims.items():
        try:
            layer_idx = int(layer_str)
        except ValueError:
            continue
        layer_data.append((layer_idx, sims))
    layer_data.sort(key=lambda x: x[0])

    if not layer_data:
        print("  Skipping grad_direction_consistency (no layer data)")
        return

    layers = [ld[0] for ld in layer_data]
    means = [np.mean(ld[1]) for ld in layer_data]
    stds = [np.std(ld[1]) for ld in layer_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: bar chart of mean cosine similarity per layer
    x = np.arange(len(layers))
    colors = [
        "#2ca02c" if m > 0.9 else "#ff7f0e" if m > 0.5 else "#d62728"
        for m in means
    ]
    ax1.bar(x, means, yerr=stds, color=colors, edgecolor="white",
            capsize=3, alpha=0.8)
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5,
                label="Perfect alignment")
    ax1.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Cosine Similarity", fontsize=12)
    ax1.set_title("Gradient Direction Consistency\n(evicted vs full context)",
                  fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(l) for l in layers])
    ax1.set_ylim(-0.2, 1.15)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: box plot per layer
    box_data = [ld[1] for ld in layer_data]
    bp = ax2.boxplot(box_data, tick_labels=[str(l) for l in layers],
                     patch_artist=True, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.6)
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Cosine Similarity", fontsize=12)
    ax2.set_title("Per-Layer Distribution of\nGradient Cosine Similarity",
                  fontsize=13)
    ax2.set_ylim(-0.2, 1.15)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Gradient Direction Alignment: Evicted vs Full Context\n"
        "(cos=1.0 means eviction does not change gradient direction)",
        fontsize=14, y=1.04,
    )
    fig.tight_layout()

    path = out_dir / "grad_direction_consistency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_FILE}\n"
            "Generate it on a GPU node with:\n"
            "  PYTHONPATH=. .venv/bin/python analysis/report_9/scripts/generate_data.py"
        )

    with open(DATA_FILE) as f:
        data = json.load(f)

    print(f"Loaded {DATA_FILE}")
    print(f"  Top-level keys: {list(data.keys())}")
    print(f"  evicted samples:      {len(data.get('evicted', []))}")
    print(f"  full_context samples: {len(data.get('full_context', []))}")
    print(f"  cosine_sims layers:   {len(data.get('cosine_sims_per_layer', {}))}")
    print(f"  cache_size:           {data.get('cache_size', '?')}")

    os.makedirs(OUT_DIR, exist_ok=True)

    plot_loss_stratified(data, OUT_DIR)
    plot_grad_norms(data, OUT_DIR)

    plot_grad_direction_consistency(data, OUT_DIR)

    # Print summary
    evicted = data.get("evicted", [])
    full = data.get("full_context", [])
    cosine_sims = data.get("cosine_sims_per_layer", {})

    print("\nSummary:")
    if evicted:
        ev_losses = [s["loss"] for s in evicted]
        retentions = [s["retention_ratio"] for s in evicted]
        print(f"  Evicted loss:    {np.mean(ev_losses):.4f} +/- {np.std(ev_losses):.4f}")
        print(f"  Retention ratio: {np.mean(retentions):.4f} +/- {np.std(retentions):.4f} "
              f"(median: {np.median(retentions):.4f})")
    if full:
        fu_losses = [s["loss"] for s in full]
        print(f"  Full loss:       {np.mean(fu_losses):.4f} +/- {np.std(fu_losses):.4f}")
    if evicted and full:
        delta = np.mean(ev_losses) - np.mean(fu_losses)
        pct = delta / max(np.mean(fu_losses), 1e-8) * 100
        print(f"  Loss increase:   {delta:.4f} ({pct:.1f}%)")
    if cosine_sims:
        all_sims = []
        for sims in cosine_sims.values():
            all_sims.extend(sims)
        print(f"  Cosine sim:      {np.mean(all_sims):.4f} +/- {np.std(all_sims):.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
