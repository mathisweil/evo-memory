#!/usr/bin/env python3
"""Regenerate plots for reports 1, 2, and 3 using maskfix data.

Report 1: Uses pre-extracted maskfix_data.json (no WandB calls).
Report 2: Fetches M1 and M3 maskfix training curves from WandB.
Report 3: Fetches per-layer retention data from M3 maskfix WandB run.

Run with:
    source activate.sh && PYTHONPATH=. HF_HOME=.hf_cache .venv/bin/python \
        analysis/regenerate_reports_1_3_plots.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_1_DIR = REPO_ROOT / "analysis" / "report_1"
REPORT_2_DIR = REPO_ROOT / "analysis" / "report_2"
REPORT_3_DIR = REPO_ROOT / "analysis" / "report_3"

TASKS = ["qasper", "2wikimqa", "qasper_e", "hotpotqa_e", "2wikimqa_e"]
TASK_LABELS = {
    "qasper":     "Qasper",
    "2wikimqa":   "2WikiMQA",
    "qasper_e":   "Qasper-E",
    "hotpotqa_e": "HotpotQA-E",
    "2wikimqa_e": "2WikiMQA-E",
}

ENTITY = "SNLP_NAMM"
PROJECT = "memory_evolution_hf"

# WandB run IDs
M1_SEGMENTS = ["kz6vqo2o", "x9a4smmf", "qfoxxi2m"]
M3_MASKFIX_RUN = "h0bzg6on"

NUM_LAYERS = 16
LAYER_COLS = [f"retention/layer_{i}" for i in range(NUM_LAYERS)]

SMOOTH_WINDOW = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def smooth(series: "pd.Series", window: int = SMOOTH_WINDOW) -> "pd.Series":
    """Rolling average with min_periods=1 to avoid NaN at edges."""
    return series.rolling(window=window, min_periods=1, center=True).mean()


# ===========================================================================
# Report 1 -- uses maskfix_data.json, no WandB needed
# ===========================================================================

def generate_report_1_plots() -> None:
    """Generate three plots for report 1 from maskfix_data.json."""
    print("=" * 60)
    print("REPORT 1: Per-Task Eviction Sensitivity (maskfix)")
    print("=" * 60)

    data_path = REPORT_1_DIR / "maskfix_data.json"
    print(f"Reading {data_path} ...")
    with open(data_path) as f:
        data = json.load(f)

    b0 = data["B0"]
    m1 = data["M1"]
    m2 = data["M2_maskfix"]
    m3 = data["M3_maskfix"]

    task_labels = [TASK_LABELS[t] for t in TASKS]

    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
    })

    # -- Plot 1: best_val_f1_comparison.png ------------------------------------
    print("  Generating best_val_f1_comparison.png ...")
    conditions = ["B0", "M1", "M2 (maskfix)", "M3 (maskfix)"]
    cond_data = [b0, m1, m2, m3]
    cond_colors = ["#95a5a6", "#f39c12", "#3498db", "#e74c3c"]

    tasks_plus_mean = TASKS + ["mean"]
    labels_plus_mean = task_labels + ["Mean"]

    fig, ax = plt.subplots(figsize=(12, 6))
    n_cond = len(conditions)
    n_groups = len(tasks_plus_mean)
    width = 0.8 / n_cond
    x = np.arange(n_groups)

    for i, (cond, cdata, color) in enumerate(
        zip(conditions, cond_data, cond_colors)
    ):
        vals = [cdata[t] for t in tasks_plus_mean]
        bars = ax.bar(
            x + i * width, vals, width,
            label=cond, color=color, edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=7, rotation=90,
            )

    ax.set_xticks(x + width * n_cond / 2 - width / 2)
    ax.set_xticklabels(labels_plus_mean)
    ax.set_ylabel("Best Validation F1")
    ax.set_title("Best Validation F1 -- B0, M1, M2 (maskfix), M3 (maskfix)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, max(max(d[t] for t in tasks_plus_mean) for d in cond_data) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = REPORT_1_DIR / "best_val_f1_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved {out_path}")

    # -- Plot 2: sensitivity_bar.png -------------------------------------------
    print("  Generating sensitivity_bar.png ...")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(TASKS))

    sens = []
    for t in TASKS:
        if abs(m1[t]) < 1e-9:
            sens.append(0.0)
        else:
            sens.append((m1[t] - m3[t]) / m1[t] * 100)

    bars = ax.bar(x, sens, 0.5, color="#e74c3c", edgecolor="white")
    for bar, val in zip(bars, sens):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2, y,
            f"{val:+.1f}%", ha="center", va=va, fontsize=9,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_ylabel("Eviction Sensitivity  (M1 - M3) / M1  [%]")
    ax.set_title(
        "Per-Task Eviction Sensitivity (maskfix M3)\n"
        "Positive = M3 worse than M1; Negative = M3 exceeds M1"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = REPORT_1_DIR / "sensitivity_bar.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved {out_path}")

    # -- Plot 3: recovery_ratio.png --------------------------------------------
    print("  Generating recovery_ratio.png ...")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(TASKS))

    rr = []
    for t in TASKS:
        if abs(m1[t]) < 1e-9:
            rr.append(float("nan"))
        else:
            rr.append(m3[t] / m1[t])
    bars = ax.bar(x, rr, 0.5, color="#2ecc71", edgecolor="white")
    for bar, val in zip(bars, rr):
        if np.isnan(val):
            continue
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2, y,
            f"{val:.2f}", ha="center", va=va, fontsize=9,
        )

    ax.axhline(1.0, color="green", linewidth=1.0, linestyle="--",
               label="Parity (M3 = M1)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_ylabel("Recovery Ratio  M3_F1 / M1_F1")
    ax.set_title("Recovery Ratio Per Task (maskfix M3 vs M1)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = REPORT_1_DIR / "recovery_ratio.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved {out_path}")

    print("Report 1 plots done.\n")


# ===========================================================================
# Report 2 -- WandB-fetched M1 and M3 maskfix training curves
# ===========================================================================

def generate_report_2_plots() -> None:
    """Generate three plots for report 2 from WandB data."""
    import pandas as pd

    print("=" * 60)
    print("REPORT 2: Adaptation Rate and Learning Efficiency (maskfix)")
    print("=" * 60)

    try:
        import wandb
    except ImportError:
        print("ERROR: wandb not installed. Skipping report 2.")
        return

    try:
        api = wandb.Api()
    except Exception as exc:
        print(f"ERROR: Could not connect to WandB API: {exc}")
        print("Skipping report 2.")
        return

    def run_path(run_id: str) -> str:
        return f"{ENTITY}/{PROJECT}/{run_id}"

    def fetch_lora_history(run_ids: list[str]) -> pd.DataFrame:
        """Fetch and concatenate LoRA history from multiple run segments."""
        cols_of_interest = [
            "lora/global_step",
            "lora/val_lb_avg_f1",
            "lora/train_loss",
            "lora/train_lb_avg_f1",
            "lora/baseline_lb_avg_f1",
        ]
        frames = []
        for rid in run_ids:
            print(f"    Fetching run {rid} ...")
            run = api.run(run_path(rid))
            h = run.history(pandas=True, samples=10000)
            keep = [c for c in cols_of_interest if c in h.columns]
            frames.append(h[keep])
        df = pd.concat(frames, ignore_index=True)
        df = df.dropna(subset=["lora/global_step"])
        df = df.drop_duplicates(subset=["lora/global_step"], keep="first")
        df = df.sort_values("lora/global_step").reset_index(drop=True)
        return df

    def get_baseline(run_id: str) -> float | None:
        """Get baseline F1 from a specific run."""
        run = api.run(run_path(run_id))
        h = run.history(pandas=True, samples=10000)
        if "lora/baseline_lb_avg_f1" not in h.columns:
            return None
        rows = h.dropna(subset=["lora/baseline_lb_avg_f1"])
        if len(rows) > 0:
            return float(rows["lora/baseline_lb_avg_f1"].iloc[0])
        return None

    # -- Fetch data ---
    print("  Fetching M1 data ...")
    try:
        m1_df = fetch_lora_history(M1_SEGMENTS)
        print(f"    M1: {len(m1_df)} rows, steps "
              f"{m1_df['lora/global_step'].min():.0f} - "
              f"{m1_df['lora/global_step'].max():.0f}")
    except Exception as exc:
        print(f"ERROR fetching M1 data: {exc}")
        print("Skipping report 2.")
        return

    print("  Fetching M3 maskfix data ...")
    try:
        m3_df = fetch_lora_history([M3_MASKFIX_RUN])
        print(f"    M3 maskfix: {len(m3_df)} rows, steps "
              f"{m3_df['lora/global_step'].min():.0f} - "
              f"{m3_df['lora/global_step'].max():.0f}")
    except Exception as exc:
        print(f"ERROR fetching M3 maskfix data: {exc}")
        print("Skipping report 2.")
        return

    print("  Fetching baselines ...")
    try:
        m1_baseline = get_baseline("kz6vqo2o")
        m3_baseline = get_baseline(M3_MASKFIX_RUN)
        if m1_baseline is None:
            m1_baseline = 22.6  # fallback from maskfix_data.json B0 mean
        if m3_baseline is None:
            m3_baseline = m1_baseline
        print(f"    M1 baseline: {m1_baseline:.2f}")
        print(f"    M3 maskfix baseline: {m3_baseline:.2f}")
    except Exception as exc:
        print(f"WARNING: Could not fetch baselines: {exc}")
        m1_baseline = 22.6
        m3_baseline = 22.6

    data = {"M1": m1_df, "M3 maskfix": m3_df}
    baselines = {"M1": m1_baseline, "M3 maskfix": m3_baseline}
    colors = {"M1": "#d62728", "M3 maskfix": "#1f77b4"}
    linestyles = {"M1": "-", "M3 maskfix": "--"}
    conditions = ["M1", "M3 maskfix"]

    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    })

    # -- Plot 1: learning_curves_overlay.png -----------------------------------
    print("  Generating learning_curves_overlay.png ...")
    fig, ax = plt.subplots(figsize=(10, 6))
    for cond in conditions:
        df = data[cond]
        sub = df.dropna(subset=["lora/val_lb_avg_f1"]).copy()
        if sub.empty:
            continue
        # Raw as faint background
        ax.plot(
            sub["lora/global_step"], sub["lora/val_lb_avg_f1"],
            color=colors[cond], alpha=0.15, linewidth=1,
        )
        # Smoothed
        ax.plot(
            sub["lora/global_step"], smooth(sub["lora/val_lb_avg_f1"]),
            label=cond, color=colors[cond],
            linestyle=linestyles[cond], linewidth=2,
        )

    ax.set_xlabel("Global Step")
    ax.set_ylabel("Val Mean F1")
    ax.set_title("Learning Curves -- Val Mean F1 (smoothed, window=5)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = REPORT_2_DIR / "learning_curves_overlay.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved {out_path}")

    # -- Plot 2: normalised_improvement.png ------------------------------------
    print("  Generating normalised_improvement.png ...")
    fig, ax = plt.subplots(figsize=(10, 6))
    for cond in conditions:
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
            sub["lora/global_step"], smooth(norm_imp),
            label=f"{cond} (baseline={baseline:.1f}, best={best_val:.1f})",
            color=colors[cond], linestyle=linestyles[cond], linewidth=2,
        )

    ax.axhline(y=0, color="grey", linestyle=":", alpha=0.5)
    ax.axhline(y=1, color="grey", linestyle=":", alpha=0.5)
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Normalised Improvement")
    ax.set_title(
        "Normalised Improvement Curve\n"
        "(val F1 - baseline) / (best val F1 - baseline)"
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.15)
    fig.tight_layout()
    out_path = REPORT_2_DIR / "normalised_improvement.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved {out_path}")

    # -- Plot 3: overfitting_gap.png -------------------------------------------
    print("  Generating overfitting_gap.png ...")
    fig, ax = plt.subplots(figsize=(10, 6))
    for cond in conditions:
        df = data[cond]
        # Use train_loss minus val_f1 since that's the requested metric
        # First try train_lb_avg_f1 - val_lb_avg_f1 (traditional gap)
        if "lora/train_lb_avg_f1" in df.columns:
            sub = df.dropna(
                subset=["lora/train_lb_avg_f1", "lora/val_lb_avg_f1"]
            ).copy()
            if not sub.empty:
                gap = sub["lora/train_lb_avg_f1"] - sub["lora/val_lb_avg_f1"]
                ax.plot(
                    sub["lora/global_step"], smooth(gap),
                    label=f"{cond} (train F1 - val F1)",
                    color=colors[cond], linestyle=linestyles[cond],
                    linewidth=2,
                )
                continue
        # Fallback: use train_loss - val_f1 (different scales but still useful)
        if "lora/train_loss" in df.columns:
            sub = df.dropna(
                subset=["lora/train_loss", "lora/val_lb_avg_f1"]
            ).copy()
            if not sub.empty:
                gap = sub["lora/train_loss"] - sub["lora/val_lb_avg_f1"]
                ax.plot(
                    sub["lora/global_step"], smooth(gap),
                    label=f"{cond} (train loss - val F1)",
                    color=colors[cond], linestyle=linestyles[cond],
                    linewidth=2,
                )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.7)
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Gap (smoothed, window=5)")
    ax.set_title("Overfitting Gap Over Training")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = REPORT_2_DIR / "overfitting_gap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved {out_path}")

    print("Report 2 plots done.\n")


# ===========================================================================
# Report 3 -- per-layer retention from M3 maskfix WandB run
# ===========================================================================

def generate_report_3_plots() -> None:
    """Generate four plots for report 3 from WandB retention data."""
    import pandas as pd

    print("=" * 60)
    print("REPORT 3: Per-Layer Retention Pattern Analysis (maskfix)")
    print("=" * 60)

    try:
        import wandb
    except ImportError:
        print("ERROR: wandb not installed. Skipping report 3.")
        return

    try:
        api = wandb.Api()
    except Exception as exc:
        print(f"ERROR: Could not connect to WandB API: {exc}")
        print("Skipping report 3.")
        return

    run_path = f"{ENTITY}/{PROJECT}/{M3_MASKFIX_RUN}"

    print(f"  Fetching retention data from {M3_MASKFIX_RUN} ...")
    try:
        run = api.run(run_path)
        h = run.history(pandas=True, samples=10000)
    except Exception as exc:
        print(f"ERROR fetching run data: {exc}")
        print("Skipping report 3.")
        return

    # Build working DataFrame
    keep_cols = ["lora/global_step"] + LAYER_COLS
    if "lora/val_lb_avg_f1" in h.columns:
        keep_cols.append("lora/val_lb_avg_f1")
    existing = [c for c in keep_cols if c in h.columns]
    df = h[existing].copy()
    df = df.dropna(subset=["lora/global_step"])
    df = df.sort_values("lora/global_step").reset_index(drop=True)

    ret_rows = df.dropna(subset=LAYER_COLS[:1])
    f1_rows = (
        df.dropna(subset=["lora/val_lb_avg_f1"])
        if "lora/val_lb_avg_f1" in df.columns
        else pd.DataFrame()
    )
    print(f"    {len(df)} total rows, {len(ret_rows)} with retention, "
          f"{len(f1_rows)} with val F1")
    print(f"    Step range: {df['lora/global_step'].min():.0f} to "
          f"{df['lora/global_step'].max():.0f}")

    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
    })

    # -- Plot 1: layer_retention_profile.png -----------------------------------
    print("  Generating layer_retention_profile.png ...")
    sub = df.dropna(subset=LAYER_COLS[:1])
    means = []
    stds = []
    for col in LAYER_COLS:
        vals = sub[col].dropna()
        means.append(vals.mean())
        stds.append(vals.std())

    fig, ax = plt.subplots(figsize=(10, 5))
    layers = list(range(NUM_LAYERS))
    bars = ax.bar(
        layers, means, yerr=stds, capsize=3,
        color="#1f77b4", alpha=0.8, edgecolor="white", linewidth=0.5,
        error_kw={"linewidth": 0.8, "alpha": 0.5},
    )
    overall_mean = np.mean(means)
    ax.axhline(
        y=overall_mean, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
    )
    ax.text(
        NUM_LAYERS - 0.5, overall_mean + 0.01,
        f"mean={overall_mean:.3f}", fontsize=9, color="red",
        ha="right", va="bottom",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Retention Ratio")
    ax.set_xticks(layers)
    ax.set_title(
        "Per-Layer Retention Profile -- M3 maskfix\n"
        "(mean over all training steps, error bars = 1 std)"
    )
    ax.set_ylim(0, max(means) * 1.3 if means else 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = REPORT_3_DIR / "layer_retention_profile.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved {out_path}")

    # -- Plot 2: retention_heatmap.png -----------------------------------------
    print("  Generating retention_heatmap.png ...")
    sub = df.dropna(subset=LAYER_COLS[:1]).copy()
    steps = sub["lora/global_step"].values
    retention_matrix = sub[LAYER_COLS].values.T  # shape: (16, n_steps)

    fig, ax = plt.subplots(figsize=(14, 6))
    if len(steps) > 0:
        im = ax.imshow(
            retention_matrix, aspect="auto", cmap="viridis",
            interpolation="nearest",
            extent=[steps[0], steps[-1], NUM_LAYERS - 0.5, -0.5],
            vmin=0, vmax=max(retention_matrix.max(), 0.5),
        )
        fig.colorbar(im, ax=ax, label="Retention Ratio", pad=0.02)
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Layer ID")
    ax.set_yticks(range(NUM_LAYERS))
    ax.set_title(
        "Retention Ratio Heatmap -- M3 maskfix\n(layer x training step)"
    )
    fig.tight_layout()
    out_path = REPORT_3_DIR / "retention_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved {out_path}")

    # -- Plot 3: retention_over_training.png -----------------------------------
    print("  Generating retention_over_training.png ...")
    sub_ret = df.dropna(subset=LAYER_COLS[:1]).copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot each layer as a separate line
    cmap = plt.cm.viridis
    for i in range(NUM_LAYERS):
        col = f"retention/layer_{i}"
        color = cmap(i / (NUM_LAYERS - 1))
        ax.plot(
            sub_ret["lora/global_step"], sub_ret[col],
            color=color, alpha=0.5, linewidth=0.8, label=f"Layer {i}",
        )

    ax.set_xlabel("Global Step")
    ax.set_ylabel("Retention Ratio")
    ax.set_title("Per-Layer Retention Over Training -- M3 maskfix")
    ax.legend(
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        fontsize=7, ncol=1, frameon=True,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = REPORT_3_DIR / "retention_over_training.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved {out_path}")

    # -- Plot 4: retention_vs_f1.png -------------------------------------------
    print("  Generating retention_vs_f1.png ...")
    if "lora/val_lb_avg_f1" in df.columns:
        sub = df.dropna(
            subset=LAYER_COLS[:1] + ["lora/val_lb_avg_f1"]
        ).copy()
        sub["mean_retention"] = sub[LAYER_COLS].mean(axis=1)

        if len(sub) >= 3:
            x_vals = sub["mean_retention"].values
            y_vals = sub["lora/val_lb_avg_f1"].values
            steps_vals = sub["lora/global_step"].values

            # Compute Spearman correlation
            try:
                from scipy import stats as spstats
                rho, pval = spstats.spearmanr(x_vals, y_vals)
                corr_str = f"Spearman r={rho:.3f}, p={pval:.3e}"
            except ImportError:
                corr_str = "scipy not available for correlation"
                rho, pval = float("nan"), float("nan")

            # Normalise steps for colour
            norm_steps = (
                (steps_vals - steps_vals.min())
                / (steps_vals.max() - steps_vals.min() + 1e-9)
            )

            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                x_vals, y_vals, c=norm_steps, cmap="Blues",
                s=30, alpha=0.7, edgecolor="grey", linewidth=0.3,
            )
            fig.colorbar(
                scatter, ax=ax,
                label="Training Progress (0=early, 1=late)",
            )
            ax.set_xlabel("Mean Retention Ratio (across all layers)")
            ax.set_ylabel("Validation F1")
            ax.set_title(
                f"Retention vs Val F1 -- M3 maskfix\n"
                f"{corr_str} (n={len(sub)})"
            )
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out_path = REPORT_3_DIR / "retention_vs_f1.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"    Saved {out_path}")
        else:
            print("    WARNING: Not enough overlapping retention+F1 data "
                  f"for scatter (got {len(sub)} rows, need >= 3)")
    else:
        print("    WARNING: No lora/val_lb_avg_f1 column in data, "
              "skipping retention_vs_f1.png")

    print("Report 3 plots done.\n")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("Regenerating plots for reports 1, 2, and 3 (maskfix data)\n")

    # Report 1: local JSON, no WandB needed
    generate_report_1_plots()

    # Reports 2 and 3: WandB needed
    generate_report_2_plots()
    generate_report_3_plots()

    print("=" * 60)
    print("All done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
