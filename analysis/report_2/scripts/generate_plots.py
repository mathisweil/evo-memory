"""Report 2 -- Adaptation Rate and Learning Efficiency (Mask-Fix).

Pulls training curves from WandB for M1 (full-context LoRA) and M3 maskfix
(LoRA + frozen NAMM with corrected attention mask) and generates:
  1. learning_curves_overlay.png -- raw val F1 with light smoothing
  2. normalised_improvement.png  -- normalised improvement curves [0, 1]
  3. overfitting_gap.png         -- train F1 - val F1 over training

Run:
    PYTHONPATH=. .venv/bin/python analysis/report_2/scripts/generate_plots.py
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
import wandb

# ── Config ──────────────────────────────────────────────────────────────────

ENTITY = "SNLP_NAMM"
PROJECT = "memory_evolution_hf"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "plots")

# M1: LoRA full-context (3 segments)
M1_RUN_IDS = ["kz6vqo2o", "x9a4smmf", "qfoxxi2m"]
M1_BASELINE_RUN = "kz6vqo2o"

# M3 maskfix: LoRA + frozen NAMM, cs1024
M3_MASKFIX_RUN_ID = "h0bzg6on"

CONDITIONS = ["M1", "M3 maskfix"]
CONDITION_COLORS = {
    "M1": "#d62728",
    "M3 maskfix": "#1f77b4",
}
CONDITION_LINESTYLES = {
    "M1": "-",
    "M3 maskfix": "--",
}

SMOOTH_WINDOW = 5


# ── Data fetching ───────────────────────────────────────────────────────────


def get_api() -> wandb.Api:
    return wandb.Api()


def run_path(run_id: str) -> str:
    return f"{ENTITY}/{PROJECT}/{run_id}"


def fetch_lora_history(api: wandb.Api, run_ids: list[str]) -> pd.DataFrame:
    """Fetch and concatenate LoRA history from multiple run segments."""
    cols_of_interest = [
        "lora/global_step",
        "lora/loss",
        "lora/val_lb_avg_f1",
        "lora/train_lb_avg_f1",
        "lora/baseline_lb_avg_f1",
    ]
    frames = []
    for rid in run_ids:
        r = api.run(run_path(rid))
        h = r.history(pandas=True, samples=10000)
        keep = [c for c in cols_of_interest if c in h.columns]
        frames.append(h[keep])
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["lora/global_step"])
    df = df.drop_duplicates(subset=["lora/global_step"], keep="first")
    df = df.sort_values("lora/global_step").reset_index(drop=True)
    return df


def get_baseline(api: wandb.Api, run_id: str) -> float | None:
    r = api.run(run_path(run_id))
    h = r.history(pandas=True, samples=10000)
    if "lora/baseline_lb_avg_f1" not in h.columns:
        return None
    rows = h.dropna(subset=["lora/baseline_lb_avg_f1"])
    if len(rows) > 0:
        return float(rows["lora/baseline_lb_avg_f1"].iloc[0])
    return None


def smooth(series: pd.Series, window: int = SMOOTH_WINDOW) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


# ── Plot 1: Learning curves overlay ────────────────────────────────────────


def plot_learning_curves_overlay(
    data: dict[str, pd.DataFrame],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in CONDITIONS:
        df = data[cond]
        sub = df.dropna(subset=["lora/val_lb_avg_f1"]).copy()
        if sub.empty:
            continue

        ax.plot(
            sub["lora/global_step"],
            sub["lora/val_lb_avg_f1"],
            color=CONDITION_COLORS[cond],
            alpha=0.15,
            linewidth=1,
        )
        ax.plot(
            sub["lora/global_step"],
            smooth(sub["lora/val_lb_avg_f1"]),
            label=cond,
            color=CONDITION_COLORS[cond],
            linestyle=CONDITION_LINESTYLES[cond],
            linewidth=2,
        )

    ax.set_xlabel("Global Step", fontsize=12)
    ax.set_ylabel("Val Mean F1", fontsize=12)
    ax.set_title(
        f"Learning Curves -- Val Mean F1 (smoothed, window={SMOOTH_WINDOW})",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "learning_curves_overlay.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 2: Normalised improvement ─────────────────────────────────────────


def plot_normalised_improvement(
    data: dict[str, pd.DataFrame],
    baselines: dict[str, float],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in CONDITIONS:
        df = data[cond]
        sub = df.dropna(subset=["lora/val_lb_avg_f1"]).copy()
        if sub.empty:
            continue

        baseline = baselines[cond]
        best_val = sub["lora/val_lb_avg_f1"].max()
        denom = best_val - baseline
        if denom <= 0:
            continue

        norm_imp = (sub["lora/val_lb_avg_f1"] - baseline) / denom
        ax.plot(
            sub["lora/global_step"],
            smooth(norm_imp),
            label=f"{cond} (baseline={baseline:.1f}, best={best_val:.1f})",
            color=CONDITION_COLORS[cond],
            linestyle=CONDITION_LINESTYLES[cond],
            linewidth=2,
        )

    ax.axhline(y=0, color="grey", linestyle=":", alpha=0.5)
    ax.axhline(y=1, color="grey", linestyle=":", alpha=0.5)
    ax.set_xlabel("Global Step", fontsize=12)
    ax.set_ylabel("Normalised Improvement", fontsize=12)
    ax.set_title(
        "Normalised Improvement Curve\n"
        "(val F1 - baseline) / (best val F1 - baseline)",
        fontsize=13,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.15)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "normalised_improvement.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 3: Overfitting gap ────────────────────────────────────────────────


def plot_overfitting_gap(data: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in CONDITIONS:
        df = data[cond]
        sub = df.dropna(
            subset=["lora/train_lb_avg_f1", "lora/val_lb_avg_f1"]
        ).copy()
        if sub.empty:
            continue

        gap = sub["lora/train_lb_avg_f1"] - sub["lora/val_lb_avg_f1"]
        ax.plot(
            sub["lora/global_step"],
            smooth(gap),
            label=cond,
            color=CONDITION_COLORS[cond],
            linestyle=CONDITION_LINESTYLES[cond],
            linewidth=2,
        )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.7)
    ax.set_xlabel("Global Step", fontsize=12)
    ax.set_ylabel("Train F1 - Val F1 (Overfitting Gap)", fontsize=12)
    ax.set_title(
        "Overfitting Gap Over Training\n"
        f"(positive = overfitting, smoothed window={SMOOTH_WINDOW})",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "overfitting_gap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    api = get_api()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Fetching M1 data...")
    m1_df = fetch_lora_history(api, M1_RUN_IDS)
    print(
        f"  M1: {len(m1_df)} rows, steps "
        f"{m1_df['lora/global_step'].min():.0f}"
        f"--{m1_df['lora/global_step'].max():.0f}"
    )

    print(f"Fetching M3 maskfix data ({M3_MASKFIX_RUN_ID})...")
    m3_df = fetch_lora_history(api, [M3_MASKFIX_RUN_ID])
    print(
        f"  M3 maskfix: {len(m3_df)} rows, steps "
        f"{m3_df['lora/global_step'].min():.0f}"
        f"--{m3_df['lora/global_step'].max():.0f}"
    )

    data = {"M1": m1_df, "M3 maskfix": m3_df}

    print("\nFetching baselines...")
    m1_baseline = get_baseline(api, M1_BASELINE_RUN)
    m3_baseline = get_baseline(api, M3_MASKFIX_RUN_ID)
    if m1_baseline is None:
        raise RuntimeError("Could not fetch M1 baseline")
    if m3_baseline is None:
        m3_baseline = m1_baseline
        print(f"  M3 baseline not found, falling back to M1 baseline: {m1_baseline:.2f}")
    else:
        print(f"  M3 maskfix baseline: {m3_baseline:.2f}")
    print(f"  M1 baseline: {m1_baseline:.2f}")
    baselines = {"M1": m1_baseline, "M3 maskfix": m3_baseline}

    print("\nSummary:")
    for cond in CONDITIONS:
        df = data[cond]
        sub = df.dropna(subset=["lora/val_lb_avg_f1"])
        best_val = sub["lora/val_lb_avg_f1"].max()
        best_step = sub.loc[
            sub["lora/val_lb_avg_f1"].idxmax(), "lora/global_step"
        ]
        print(
            f"  {cond}: best val F1 = {best_val:.2f} at step {best_step:.0f}, "
            f"baseline = {baselines[cond]:.2f}"
        )

    print("\nGenerating plots...")
    plot_learning_curves_overlay(data)
    plot_normalised_improvement(data, baselines)
    plot_overfitting_gap(data)

    print(f"\nDone. All plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
