"""Compare LoRA weight matrices between two checkpoints (e.g. M1 vs M4).

No GPU needed — just loads .pt files and computes per-layer:
  - Frobenius distance: how different are the learned adaptations?
  - Cosine similarity: are they in the same subspace?
  - Norm ratio: did one model learn larger adaptations?

Outputs plots + a JSON summary.

Usage:
    python scripts/analyze_lora_divergence.py \
        --ckpt_a checkpoints_backup/lora_m1_original/best_ckpt.pt \
        --ckpt_b checkpoints_backup/lora_m4_cs1024_maskfix/best_ckpt_step260_val52.06.pt \
        --label_a "M1 (full cache)" \
        --label_b "M4 (NAMM in-loop)" \
        --output_dir analysis_out/lora_divergence
"""

import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare LoRA weights between two checkpoints.")
    p.add_argument("--ckpt_a", type=str, required=True)
    p.add_argument("--ckpt_b", type=str, required=True)
    p.add_argument("--label_a", type=str, default="Model A")
    p.add_argument("--label_b", type=str, default="Model B")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def extract_layer_module(key: str):
    """Parse 'base_model.model.layers.5.self_attn.q_proj.lora_A.default.weight'
    into (layer=5, module='q_proj', matrix='A')."""
    m = re.search(r"layers\.(\d+)\.self_attn\.(\w+)\.lora_([AB])\.", key)
    if m:
        return int(m.group(1)), m.group(2), m.group(3)
    return None, None, None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_a = torch.load(args.ckpt_a, map_location="cpu", weights_only=False)
    ckpt_b = torch.load(args.ckpt_b, map_location="cpu", weights_only=False)
    sd_a = ckpt_a["lora_state_dict"]
    sd_b = ckpt_b["lora_state_dict"]

    print(f"A: {args.ckpt_a} (step {ckpt_a.get('best_step','?')}, "
          f"val {ckpt_a.get('best_val_score','?')})")
    print(f"B: {args.ckpt_b} (step {ckpt_b.get('best_step','?')}, "
          f"val {ckpt_b.get('best_val_score','?')})")
    print(f"  {len(sd_a)} tensors in A, {len(sd_b)} tensors in B")

    # ── Per-layer, per-module, per-matrix metrics ─────────────────────────
    shared_keys = sorted(set(sd_a.keys()) & set(sd_b.keys()))
    print(f"  {len(shared_keys)} shared keys")

    records = []
    for key in shared_keys:
        layer, module, matrix = extract_layer_module(key)
        if layer is None:
            continue
        wa = sd_a[key].float()
        wb = sd_b[key].float()

        norm_a = wa.norm().item()
        norm_b = wb.norm().item()
        diff = (wa - wb)
        frob_dist = diff.norm().item()
        cos_sim = (torch.nn.functional.cosine_similarity(
            wa.flatten().unsqueeze(0),
            wb.flatten().unsqueeze(0)).item())

        # Effective LoRA update: for a given layer+module, the update is
        # delta_W = B @ A. Compute this for both and compare.
        records.append({
            "layer": layer,
            "module": module,
            "matrix": matrix,
            "key": key,
            "norm_a": norm_a,
            "norm_b": norm_b,
            "norm_ratio": norm_b / max(norm_a, 1e-10),
            "frob_distance": frob_dist,
            "cosine_similarity": cos_sim,
        })

    # ── Compute effective delta_W = B @ A per layer/module ────────────────
    # Group by (layer, module)
    grouped = defaultdict(dict)
    for key in shared_keys:
        layer, module, matrix = extract_layer_module(key)
        if layer is None:
            continue
        grouped[(layer, module)][matrix] = key

    effective_records = []
    for (layer, module), matrices in sorted(grouped.items()):
        if "A" in matrices and "B" in matrices:
            Aa = sd_a[matrices["A"]].float()
            Ba = sd_a[matrices["B"]].float()
            Ab = sd_b[matrices["A"]].float()
            Bb = sd_b[matrices["B"]].float()
            dW_a = Ba @ Aa  # effective update for model A
            dW_b = Bb @ Ab  # effective update for model B

            diff = dW_a - dW_b
            frob = diff.norm().item()
            cos = torch.nn.functional.cosine_similarity(
                dW_a.flatten().unsqueeze(0),
                dW_b.flatten().unsqueeze(0)).item()
            norm_a = dW_a.norm().item()
            norm_b = dW_b.norm().item()

            effective_records.append({
                "layer": layer,
                "module": module,
                "dW_norm_a": norm_a,
                "dW_norm_b": norm_b,
                "dW_norm_ratio": norm_b / max(norm_a, 1e-10),
                "dW_frob_distance": frob,
                "dW_cosine_similarity": cos,
            })

    # ── Aggregate per layer ───────────────────────────────────────────────
    layers = sorted(set(r["layer"] for r in effective_records))
    modules = sorted(set(r["module"] for r in effective_records))

    per_layer = defaultdict(lambda: defaultdict(dict))
    for r in effective_records:
        per_layer[r["layer"]][r["module"]] = r

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Effective LoRA update (delta_W = B @ A) divergence per layer")
    print(f"{'='*70}")
    print(f"{'Layer':>6} | {'Module':>8} | {'‖dW_A‖':>8} | {'‖dW_B‖':>8} | "
          f"{'Frob dist':>9} | {'Cosine':>7} | {'Norm ratio':>10}")
    print("-" * 70)
    for l in layers:
        for m in modules:
            if m in per_layer[l]:
                r = per_layer[l][m]
                print(f"  L{l:2d}  | {m:>8} | {r['dW_norm_a']:8.4f} | "
                      f"{r['dW_norm_b']:8.4f} | {r['dW_frob_distance']:9.4f} | "
                      f"{r['dW_cosine_similarity']:7.4f} | "
                      f"{r['dW_norm_ratio']:10.4f}")

    # ── Save JSON ─────────────────────────────────────────────────────────
    output = {
        "ckpt_a": args.ckpt_a,
        "ckpt_b": args.ckpt_b,
        "label_a": args.label_a,
        "label_b": args.label_b,
        "per_matrix": records,
        "effective_dW": effective_records,
    }
    json_path = os.path.join(args.output_dir, "lora_divergence.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for m_idx, module in enumerate(modules):
        mod_recs = [r for r in effective_records if r["module"] == module]
        mod_layers = [r["layer"] for r in mod_recs]
        frobs = [r["dW_frob_distance"] for r in mod_recs]
        cosines = [r["dW_cosine_similarity"] for r in mod_recs]
        ratios = [r["dW_norm_ratio"] for r in mod_recs]

        marker = "o" if m_idx == 0 else "s"
        color = "#1f77b4" if module == "q_proj" else "#e377c2"

        axes[0].plot(mod_layers, frobs, f"{marker}-", color=color,
                     label=module, linewidth=2, markersize=5)
        axes[1].plot(mod_layers, cosines, f"{marker}-", color=color,
                     label=module, linewidth=2, markersize=5)
        axes[2].plot(mod_layers, ratios, f"{marker}-", color=color,
                     label=module, linewidth=2, markersize=5)

    axes[0].set_ylabel("Frobenius distance")
    axes[0].set_title(f"‖dW_A − dW_B‖\n({args.label_a} vs {args.label_b})")
    axes[1].set_ylabel("Cosine similarity")
    axes[1].set_title("cos(dW_A, dW_B)\n(1=same direction, 0=orthogonal)")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].axhline(0, color="gray", linestyle=":", linewidth=0.5)
    axes[2].set_ylabel(f"‖dW_B‖ / ‖dW_A‖")
    axes[2].set_title(f"Norm ratio\n(>1 means {args.label_b} has larger updates)")
    axes[2].axhline(1, color="gray", linestyle=":", linewidth=0.5)

    for ax in axes:
        ax.set_xlabel("Layer")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(f"LoRA Weight Divergence: {args.label_a} vs {args.label_b}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "lora_divergence.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Per-layer norm comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for m_idx, module in enumerate(modules):
        mod_recs = [r for r in effective_records if r["module"] == module]
        mod_layers = [r["layer"] for r in mod_recs]
        norms_a = [r["dW_norm_a"] for r in mod_recs]
        norms_b = [r["dW_norm_b"] for r in mod_recs]

        color_a = "#1f77b4" if module == "q_proj" else "#aec7e8"
        color_b = "#e377c2" if module == "q_proj" else "#f7b6d2"

        axes[m_idx].bar(np.array(mod_layers) - 0.2, norms_a, 0.35,
                        label=args.label_a, color=color_a, edgecolor="black",
                        linewidth=0.3)
        axes[m_idx].bar(np.array(mod_layers) + 0.2, norms_b, 0.35,
                        label=args.label_b, color=color_b, edgecolor="black",
                        linewidth=0.3)
        axes[m_idx].set_xlabel("Layer")
        axes[m_idx].set_ylabel("‖B @ A‖ (effective update norm)")
        axes[m_idx].set_title(f"{module}")
        axes[m_idx].legend()
        axes[m_idx].grid(axis="y", alpha=0.3)

    plt.suptitle("Per-Layer Effective LoRA Update Magnitude",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "lora_norms.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
