#!/usr/bin/env python3
"""
Analysis S1 -- Per-Task Eviction Sensitivity
=============================================
Pulls best-val-F1 numbers from WandB for B0, M1, M2, M3 conditions and
generates three plots:
  1. sensitivity_bar.png       -- eviction sensitivity per task x cache size
  2. best_val_f1_comparison.png -- full grouped bar comparison
  3. recovery_ratio.png        -- recovery ratio per task x cache size

Also prints a summary table to stdout (and writes numbers to results.json).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

# ── Configuration ────────────────────────────────────────────────────────────

ENTITY = "SNLP_NAMM"
PROJECT = "memory_evolution_hf"
OUT_DIR = Path(__file__).parent

TASKS = ["qasper", "2wikimqa", "qasper_e", "hotpotqa_e", "2wikimqa_e"]
TASK_LABELS = {
    "qasper":     "Qasper",
    "2wikimqa":   "2WikiMQA",
    "qasper_e":   "Qasper-E",
    "hotpotqa_e": "HotpotQA-E",
    "2wikimqa_e": "2WikiMQA-E",
}

# Run IDs ─────────────────────────────────────────────────────────────────────
# M1: LoRA only (3 segments, final is qfoxxi2m but we scan all for best val)
M1_SEGMENTS = ["kz6vqo2o", "x9a4smmf", "qfoxxi2m"]

# M2: NAMM-only, per cache size
M2_RUNS = {
    1024: ["lenhmfb1"],
    2048: ["y5fdw0f9", "ccflnsds"],
    3072: ["quc95irz"],
}

# M3: LoRA + frozen NAMM, per cache size
M3_RUNS = {
    1024: ["ovosogkj"],
    2048: ["m4knrhmr"],
    3072: ["4sgkswa6"],
}

# B0 baseline logged in qfoxxi2m at step 0
B0_RUN = "qfoxxi2m"


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_api() -> wandb.Api:
    return wandb.Api()


def run_path(run_id: str) -> str:
    return f"{ENTITY}/{PROJECT}/{run_id}"


def fetch_lora_best_val(api: wandb.Api, run_ids: list[str]) -> dict[str, float]:
    """For LoRA runs (M1, M3), find the step with max val_lb_avg_f1 across
    all segments and return per-task F1 at that step."""
    best_avg = -1.0
    best_row: dict[str, float] | None = None

    for rid in run_ids:
        run = api.run(run_path(rid))
        keys = ["lora/val_lb_avg_f1"] + [f"lora/val_lb_{t}" for t in TASKS]
        hist = run.history(keys=keys, pandas=True)
        if hist.empty:
            continue
        # Drop rows where avg is NaN (non-eval steps)
        hist = hist.dropna(subset=["lora/val_lb_avg_f1"])
        if hist.empty:
            continue
        idx = hist["lora/val_lb_avg_f1"].idxmax()
        row_avg = hist.loc[idx, "lora/val_lb_avg_f1"]
        if row_avg > best_avg:
            best_avg = row_avg
            best_row = {t: hist.loc[idx, f"lora/val_lb_{t}"] for t in TASKS}
            best_row["mean"] = row_avg

    if best_row is None:
        raise RuntimeError(f"No val data found for runs {run_ids}")
    return best_row


def fetch_namm_best_val(api: wandb.Api, run_ids: list[str]) -> dict[str, float]:
    """For NAMM runs (M2), compute mean of 5 task val metrics per iter,
    find the iter with max mean, return per-task and mean."""
    all_frames = []
    for rid in run_ids:
        run = api.run(run_path(rid))
        keys = [f"val_lb/{t}" for t in TASKS]
        hist = run.history(keys=keys, pandas=True)
        if not hist.empty:
            all_frames.append(hist)

    if not all_frames:
        raise RuntimeError(f"No val data found for NAMM runs {run_ids}")

    df = pd.concat(all_frames, ignore_index=True)
    # Drop rows with any NaN in val columns
    val_cols = [f"val_lb/{t}" for t in TASKS]
    df = df.dropna(subset=val_cols)
    if df.empty:
        raise RuntimeError(f"All rows NaN for NAMM runs {run_ids}")

    df["_mean"] = df[val_cols].mean(axis=1)
    idx = df["_mean"].idxmax()
    result = {t: df.loc[idx, f"val_lb/{t}"] for t in TASKS}
    result["mean"] = df.loc[idx, "_mean"]
    return result


def fetch_b0(api: wandb.Api) -> dict[str, float]:
    """B0 baseline values logged at start of M1 final segment."""
    run = api.run(run_path(B0_RUN))
    keys = [f"lora/baseline_lb_{t}" for t in TASKS] + ["lora/baseline_lb_avg_f1"]
    hist = run.history(keys=keys, pandas=True)
    hist = hist.dropna(subset=["lora/baseline_lb_avg_f1"])
    # Take the first logged row (baselines are logged once)
    row = hist.iloc[0]
    result = {t: row[f"lora/baseline_lb_{t}"] for t in TASKS}
    result["mean"] = row["lora/baseline_lb_avg_f1"]
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    api = get_api()

    print("Fetching B0 baseline ...")
    b0 = fetch_b0(api)
    print(f"  B0 mean = {b0['mean']:.2f}")

    print("Fetching M1 (LoRA full-context) ...")
    m1 = fetch_lora_best_val(api, M1_SEGMENTS)
    print(f"  M1 mean = {m1['mean']:.2f}")

    m2: dict[int, dict[str, float]] = {}
    for cs, rids in M2_RUNS.items():
        print(f"Fetching M2 cs{cs} ...")
        m2[cs] = fetch_namm_best_val(api, rids)
        print(f"  M2 cs{cs} mean = {m2[cs]['mean']:.2f}")

    m3: dict[int, dict[str, float]] = {}
    for cs, rids in M3_RUNS.items():
        print(f"Fetching M3 cs{cs} ...")
        m3[cs] = fetch_lora_best_val(api, rids)
        print(f"  M3 cs{cs} mean = {m3[cs]['mean']:.2f}")

    # ── Save raw numbers ─────────────────────────────────────────────────
    def _to_native(d: dict) -> dict:
        """Convert numpy types to native Python for JSON serialization."""
        return {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in d.items()}

    numbers = {
        "B0": _to_native(b0),
        "M1": _to_native(m1),
        "M2": {str(k): _to_native(v) for k, v in m2.items()},
        "M3": {str(k): _to_native(v) for k, v in m3.items()},
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(numbers, f, indent=2)
    print(f"\nRaw numbers written to {OUT_DIR / 'results.json'}")

    # ── Print summary table ──────────────────────────────────────────────
    print("\n" + "=" * 90)
    header = f"{'Task':<14}" + "".join(f"{TASK_LABELS.get(t,t):>14}" for t in TASKS) + f"{'Mean':>14}"
    print(header)
    print("-" * 90)

    def _print_row(label: str, d: dict[str, float]) -> None:
        vals = "".join(f"{d[t]:14.2f}" for t in TASKS) + f"{d['mean']:14.2f}"
        print(f"{label:<14}{vals}")

    _print_row("B0", b0)
    _print_row("M1 (LoRA)", m1)
    for cs in sorted(m2):
        _print_row(f"M2 cs{cs}", m2[cs])
    for cs in sorted(m3):
        _print_row(f"M3 cs{cs}", m3[cs])
    print("=" * 90)

    # ── Plot styling ─────────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
    })
    task_labels = [TASK_LABELS[t] for t in TASKS]
    cache_sizes = sorted(m3.keys())
    cs_colors = {1024: "#e74c3c", 2048: "#3498db", 3072: "#2ecc71"}

    # ── Plot 1: Sensitivity bar ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(TASKS))
    width = 0.25
    for i, cs in enumerate(cache_sizes):
        sens = [(m1[t] - m3[cs][t]) / m1[t] * 100 if m1[t] != 0 else 0.0
                for t in TASKS]
        bars = ax.bar(x + i * width, sens, width, label=f"cs={cs}",
                      color=cs_colors[cs], edgecolor="white")
        # annotate values
        for bar, val in zip(bars, sens):
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, y,
                    f"{val:.1f}%", ha="center", va=va, fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(task_labels)
    ax.set_ylabel("Eviction Sensitivity  (M1-M3)/M1  [%]")
    ax.set_title("Per-Task Eviction Sensitivity (positive = M3 worse)")
    ax.legend(title="Cache Size")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sensitivity_bar.png")
    plt.close(fig)
    print("Saved sensitivity_bar.png")

    # ── Plot 2: Best val F1 comparison ───────────────────────────────────
    conditions = ["B0", "M1"] + [f"M2 cs{cs}" for cs in cache_sizes] + [f"M3 cs{cs}" for cs in cache_sizes]
    cond_data = [b0, m1] + [m2[cs] for cs in cache_sizes] + [m3[cs] for cs in cache_sizes]
    cond_colors = (
        ["#95a5a6", "#f39c12"]
        + [plt.cm.Blues(0.4 + 0.2 * i) for i in range(len(cache_sizes))]
        + [plt.cm.Reds(0.4 + 0.2 * i) for i in range(len(cache_sizes))]
    )
    tasks_plus_mean = TASKS + ["mean"]
    labels_plus_mean = task_labels + ["Mean"]

    fig, ax = plt.subplots(figsize=(14, 6))
    n_cond = len(conditions)
    n_tasks = len(tasks_plus_mean)
    width = 0.8 / n_cond
    x = np.arange(n_tasks)

    for i, (cond, data, color) in enumerate(zip(conditions, cond_data, cond_colors)):
        vals = [data[t] for t in tasks_plus_mean]
        ax.bar(x + i * width, vals, width, label=cond, color=color, edgecolor="white")

    ax.set_xticks(x + width * n_cond / 2 - width / 2)
    ax.set_xticklabels(labels_plus_mean)
    ax.set_ylabel("Best Validation F1")
    ax.set_title("Best Validation F1 Comparison Across Conditions")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.set_ylim(0, max(max(d[t] for t in tasks_plus_mean) for d in cond_data) * 1.15)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "best_val_f1_comparison.png")
    plt.close(fig)
    print("Saved best_val_f1_comparison.png")

    # ── Plot 3: Recovery ratio ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(TASKS) + 1)  # tasks + mean
    width = 0.25

    for i, cs in enumerate(cache_sizes):
        rr = []
        for t in TASKS:
            denom = m1[t] - m2[cs][t]
            if abs(denom) < 1e-9:
                rr.append(float("nan"))
            else:
                rr.append((m3[cs][t] - m2[cs][t]) / denom)
        # mean recovery ratio
        denom_mean = m1["mean"] - m2[cs]["mean"]
        if abs(denom_mean) < 1e-9:
            rr.append(float("nan"))
        else:
            rr.append((m3[cs]["mean"] - m2[cs]["mean"]) / denom_mean)

        bars = ax.bar(x + i * width, rr, width, label=f"cs={cs}",
                      color=cs_colors[cs], edgecolor="white")
        for bar, val in zip(bars, rr):
            if np.isnan(val):
                continue
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, y,
                    f"{val:.2f}", ha="center", va=va, fontsize=8)

    ax.axhline(1.0, color="green", linewidth=1.0, linestyle="--", label="Full recovery")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(task_labels + ["Mean"])
    ax.set_ylabel("Recovery Ratio  (M3-M2)/(M1-M2)")
    ax.set_title("Recovery Ratio: How Much LoRA+NAMM Recovers vs LoRA-only")
    ax.legend(title="Cache Size")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recovery_ratio.png")
    plt.close(fig)
    print("Saved recovery_ratio.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
