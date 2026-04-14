"""Report 3 -- Per-Layer Retention Pattern Analysis (Mask-Fix).

Pulls retention data from WandB for M3 maskfix (LoRA + frozen NAMM with
corrected attention mask) and generates:
  1. layer_retention_profile.png  -- mean retention ratio per layer
  2. retention_heatmap.png        -- heatmap of retention over training
  3. retention_over_training.png  -- mean retention + val F1 dual-axis
  4. retention_vs_f1.png          -- scatter: mean retention vs val F1

Run:
    PYTHONPATH=. .venv/bin/python analysis/report_3/generate_plots.py
"""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as spstats
import wandb

# ── Config ──────────────────────────────────────────────────────────────────

ENTITY = "SNLP_NAMM"
PROJECT = "memory_evolution_hf"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

NUM_LAYERS = 16
LAYER_COLS = [f"retention/layer_{i}" for i in range(NUM_LAYERS)]

# M3 maskfix run (LoRA + frozen NAMM, cs1024, corrected attention mask)
M3_MASKFIX_RUN_ID = "h0bzg6on"
M3_LABEL = "M3 maskfix cs1024"
M3_COLOR = "#1f77b4"

SMOOTH_WINDOW = 15


# ── Data fetching ───────────────────────────────────────────────────────────


def get_api() -> wandb.Api:
    return wandb.Api()


def run_path(run_id: str) -> str:
    return f"{ENTITY}/{PROJECT}/{run_id}"


def fetch_run_data(api: wandb.Api, run_id: str) -> pd.DataFrame:
    """Fetch retention and val F1 data for a single run."""
    run = api.run(run_path(run_id))
    h = run.history(pandas=True, samples=10000)

    keep_cols = ["lora/global_step"] + LAYER_COLS
    if "lora/val_lb_avg_f1" in h.columns:
        keep_cols.append("lora/val_lb_avg_f1")

    existing = [c for c in keep_cols if c in h.columns]
    df = h[existing].copy()
    df = df.dropna(subset=["lora/global_step"])
    df = df.sort_values("lora/global_step").reset_index(drop=True)
    return df


# ── Plot 1: Layer retention profile ────────────────────────────────────────


def plot_layer_retention_profile(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    means = []
    stds = []
    for col in LAYER_COLS:
        vals = df[col].dropna()
        means.append(vals.mean())
        stds.append(vals.std())

    layers = list(range(NUM_LAYERS))
    ax.bar(
        layers, means, yerr=stds, capsize=2,
        color=M3_COLOR, alpha=0.8, edgecolor="white", linewidth=0.5,
        error_kw={"linewidth": 0.8, "alpha": 0.5},
    )

    overall_mean = float(np.mean(means))
    ax.axhline(y=overall_mean, color="red", linestyle="--", linewidth=1.2,
               alpha=0.7)
    ax.text(NUM_LAYERS - 0.5, overall_mean + 0.01,
            f"mean={overall_mean:.3f}", fontsize=9, color="red",
            ha="right", va="bottom")

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean Retention Ratio", fontsize=11)
    ax.set_title(
        f"Per-Layer Retention Profile -- {M3_LABEL}\n"
        "(mean over all training steps, error bars = 1 std)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(layers)
    ax.set_xticklabels(layers, fontsize=9)
    ax.set_ylim(0, max(means) * 1.3)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "layer_retention_profile.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 2: Retention heatmap ──────────────────────────────────────────────


def plot_retention_heatmap(df: pd.DataFrame) -> None:
    sub = df.dropna(subset=LAYER_COLS[:1]).copy()
    steps = sub["lora/global_step"].values
    retention_matrix = sub[LAYER_COLS].values.T  # (16, n_steps)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(
        retention_matrix, aspect="auto", cmap="viridis",
        interpolation="nearest",
        extent=[steps[0], steps[-1], NUM_LAYERS - 0.5, -0.5],
        vmin=0, vmax=1,
    )
    fig.colorbar(im, ax=ax, label="Retention Ratio", pad=0.02)
    ax.set_xlabel("Global Step", fontsize=11)
    ax.set_ylabel("Layer ID", fontsize=11)
    ax.set_yticks(range(NUM_LAYERS))
    ax.set_title(
        f"Retention Ratio Heatmap -- {M3_LABEL}\n(raw, unsmoothed)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "retention_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 3: Retention over training (dual axis) ───────────────────────────


def plot_retention_over_training(df: pd.DataFrame) -> None:
    sub_ret = df.dropna(subset=LAYER_COLS[:1]).copy()
    sub_ret["mean_retention"] = sub_ret[LAYER_COLS].mean(axis=1)
    sub_f1 = df.dropna(subset=["lora/val_lb_avg_f1"]).copy()

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(sub_ret["lora/global_step"], sub_ret["mean_retention"],
             color=M3_COLOR, alpha=0.3, linewidth=0.8)
    smoothed_ret = sub_ret["mean_retention"].rolling(
        window=SMOOTH_WINDOW, min_periods=1, center=True
    ).mean()
    ax1.plot(sub_ret["lora/global_step"], smoothed_ret,
             color=M3_COLOR, linewidth=2.0, label="Mean Retention (smoothed)")
    ax1.set_xlabel("Global Step", fontsize=11)
    ax1.set_ylabel("Mean Retention Ratio", fontsize=11, color=M3_COLOR)
    ax1.tick_params(axis="y", labelcolor=M3_COLOR)
    ax1.set_ylim(0, 0.6)

    ax2 = ax1.twinx()
    ax2.plot(sub_f1["lora/global_step"], sub_f1["lora/val_lb_avg_f1"],
             color="#d62728", alpha=0.4, linewidth=0.8)
    win = min(SMOOTH_WINDOW, len(sub_f1))
    smoothed_f1 = sub_f1["lora/val_lb_avg_f1"].rolling(
        window=win, min_periods=1, center=True
    ).mean()
    ax2.plot(sub_f1["lora/global_step"], smoothed_f1,
             color="#d62728", linewidth=2.0, label="Val F1 (smoothed)")
    ax2.set_ylabel("Validation F1", fontsize=11, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

    ax1.set_title(
        f"Mean Retention and Val F1 Over Training -- {M3_LABEL}",
        fontsize=13, fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "retention_over_training.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 4: Retention vs F1 scatter ────────────────────────────────────────


def plot_retention_vs_f1(df: pd.DataFrame) -> None:
    sub = df.dropna(
        subset=LAYER_COLS[:1] + ["lora/val_lb_avg_f1"]
    ).copy()
    sub["mean_retention"] = sub[LAYER_COLS].mean(axis=1)

    if len(sub) < 3:
        print("  WARNING: Not enough overlapping retention+F1 data for scatter")
        return

    x = sub["mean_retention"].values
    y = sub["lora/val_lb_avg_f1"].values
    steps = sub["lora/global_step"].values

    rho, pval = spstats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(8, 6))
    norm_steps = (steps - steps.min()) / (steps.max() - steps.min() + 1e-9)
    scatter = ax.scatter(
        x, y, c=norm_steps, cmap="Blues", s=30, alpha=0.7,
        edgecolor="grey", linewidth=0.3,
    )
    fig.colorbar(scatter, ax=ax, label="Training Progress (0=early, 1=late)")

    ax.set_xlabel("Mean Retention Ratio (across all layers)", fontsize=11)
    ax.set_ylabel("Validation F1", fontsize=11)
    ax.set_title(
        f"Retention vs Val F1 -- {M3_LABEL}\n"
        f"Spearman r={rho:.3f}, p={pval:.3e} (n={len(sub)})",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "retention_vs_f1.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    api = get_api()
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Fetching {M3_LABEL} ({M3_MASKFIX_RUN_ID})...")
    df = fetch_run_data(api, M3_MASKFIX_RUN_ID)
    ret_rows = df.dropna(subset=LAYER_COLS[:1])
    f1_rows = (
        df.dropna(subset=["lora/val_lb_avg_f1"])
        if "lora/val_lb_avg_f1" in df.columns
        else pd.DataFrame()
    )
    print(
        f"  {len(df)} total rows, {len(ret_rows)} with retention, "
        f"{len(f1_rows)} with val F1"
    )
    print(
        f"  Step range: {df['lora/global_step'].min():.0f} "
        f"to {df['lora/global_step'].max():.0f}"
    )

    print("\nPer-layer mean retention:")
    for i in range(NUM_LAYERS):
        col = f"retention/layer_{i}"
        mean = df[col].dropna().mean()
        print(f"  layer {i:2d}: {mean:.4f}")

    print("\nGenerating plots...")
    plot_layer_retention_profile(df)
    plot_retention_heatmap(df)
    plot_retention_over_training(df)
    plot_retention_vs_f1(df)

    print(f"\nDone. All plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
