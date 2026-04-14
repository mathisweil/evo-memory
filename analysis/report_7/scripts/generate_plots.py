#!/usr/bin/env python3
"""Analysis 7 -- Representation Similarity (CKA) plots.

Loads pre-computed CKA data from data/maskfix_data.npz and generates:
  plots/cka_by_layer.png  -- Bar chart of linear CKA per layer (M1 vs M3)
  plots/cka_heatmap.png   -- Cross-layer CKA heatmap (M1 layer i vs M3 layer j)

Only the mask-fixed ("maskfix") data is plotted.  The buggy-run keys
(layer_cka_buggy, cross_cka_buggy) in the npz are ignored.

Runnable without a GPU:
    PYTHONPATH=. .venv/bin/python analysis/report_7/scripts/generate_plots.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG -- how to regenerate the data
# ---------------------------------------------------------------------------
# Data file produced by analysis/generate_data_4_5_7.py on a GPU node.
# That script loads the following checkpoints:
#   M1 LoRA   : experiment_artifacts/gcs/M1/best_ckpt.pt
#   M3 LoRA   : experiment_artifacts/gcs/M3_cs1024/best_ckpt.pt
#   M2 NAMM   : experiment_artifacts/gcs/M2_cs1024/ckpt.pt
# and computes linear CKA between M1 and M3 hidden states on the test split.
#
# To regenerate the data:
#   PYTHONPATH=. .venv/bin/python analysis/generate_data_4_5_7.py
#
# NPZ keys used here:
#   layer_cka_maskfix   : (17,) float64  -- per-layer CKA (emb + 16 layers)
#   cross_cka_maskfix   : (17, 17) float64 -- cross-layer CKA matrix
# Ignored keys (buggy run):
#   layer_cka_buggy, cross_cka_buggy
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent
DATA_FILE = REPORT_DIR / "data" / "maskfix_data.npz"
OUT_DIR = REPORT_DIR / "plots"

NUM_LAYERS = 16  # LLaMA 3.2-1B decoder layers
LAYER_LABELS = ["emb"] + [f"L{i}" for i in range(NUM_LAYERS)]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_cka_by_layer(layer_cka: np.ndarray, out_dir: Path) -> None:
    """Bar chart of linear CKA at each layer (y-axis 0.95 -- 1.005)."""
    n = len(layer_cka)
    labels = LAYER_LABELS[:n]

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = plt.cm.RdYlGn(np.interp(layer_cka, [0.95, 1.0], [0.0, 1.0]))
    bars = ax.bar(range(n), layer_cka, color=colors, edgecolor="black",
                  linewidth=0.5)
    ax.bar_label(bars, fmt="%.4f", fontsize=7, padding=2, rotation=45)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Linear CKA", fontsize=12)
    ax.set_title(
        "Representation Similarity: M1 vs M3 (cs1024)\n"
        "Linear CKA per Layer (mask-fixed run)",
        fontsize=13,
    )
    ax.set_ylim(0.95, 1.005)
    ax.axhline(y=1.0, color="grey", linestyle=":", alpha=0.4)
    ax.grid(True, alpha=0.3, axis="y")

    ax.annotate(
        "M1 = LoRA full-context | M3 = LoRA + frozen NAMM cs1024",
        xy=(0.98, 0.02), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    fig.tight_layout()
    path = out_dir / "cka_by_layer.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_cka_heatmap(cross_cka: np.ndarray, out_dir: Path) -> None:
    """Cross-layer CKA heatmap (M1 layer i vs M3 layer j)."""
    n = cross_cka.shape[0]
    labels = LAYER_LABELS[:n]

    fig, ax = plt.subplots(figsize=(10, 9))

    im = ax.imshow(cross_cka, cmap="viridis", vmin=0, vmax=1,
                   aspect="equal", origin="lower")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_xlabel("M3 (LoRA + NAMM cs1024) Layer", fontsize=12)
    ax.set_ylabel("M1 (LoRA-only, full context) Layer", fontsize=12)
    ax.set_title(
        "Cross-Layer CKA Heatmap: M1 vs M3\n"
        "Off-diagonal peaks suggest computational shifts",
        fontsize=13,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Linear CKA", fontsize=11)

    # Annotate cell values
    if n <= 17:
        for i in range(n):
            for j in range(n):
                val = cross_cka[i, j]
                text_color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=text_color)

    # Diagonal reference
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5], "r--", linewidth=1, alpha=0.5)

    fig.tight_layout()
    path = out_dir / "cka_heatmap.png"
    fig.savefig(path, dpi=150)
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
            "  PYTHONPATH=. .venv/bin/python analysis/generate_data_4_5_7.py"
        )

    data = np.load(DATA_FILE)
    print(f"Loaded {DATA_FILE}")
    print(f"  Keys: {sorted(data.keys())}")

    layer_cka = data["layer_cka_maskfix"]       # (17,)
    cross_cka = data["cross_cka_maskfix"]        # (17, 17)

    print(f"  layer_cka_maskfix: shape={layer_cka.shape}, "
          f"range=[{layer_cka.min():.4f}, {layer_cka.max():.4f}]")
    print(f"  cross_cka_maskfix: shape={cross_cka.shape}")

    os.makedirs(OUT_DIR, exist_ok=True)

    plot_cka_by_layer(layer_cka, OUT_DIR)
    plot_cka_heatmap(cross_cka, OUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
