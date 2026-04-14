"""Report 1 -- Per-Task Eviction Sensitivity (Mask-Fix Runs).

Loads pre-extracted val F1 data from maskfix_data.json and generates:
  1. best_val_f1_comparison.png -- grouped bar: B0, M1, M2, M3 per task
  2. sensitivity_bar.png        -- (M3 - M1) / M1 per task
  3. recovery_ratio.png         -- M3 / M1 per task

Run:
    PYTHONPATH=. .venv/bin/python analysis/report_1/scripts/generate_plots.py

To refresh maskfix_data.json from WandB, re-run the extraction script with
the following run IDs:
  - M1 (LoRA full-context): kz6vqo2o, x9a4smmf, qfoxxi2m (3 segments)
  - M2 (NAMM-only maskfix, cs1024): z5bo4n8k
  - M3 (LoRA + frozen NAMM maskfix, cs1024): h0bzg6on
  - B0 baseline: logged in kz6vqo2o (first M1 segment)
  Entity: SNLP_NAMM, Project: memory_evolution_hf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent.parent / "data" / "maskfix_data.json"
OUT_DIR = Path(__file__).parent.parent / "plots"

TASKS = ["qasper", "2wikimqa", "qasper_e", "hotpotqa_e", "2wikimqa_e"]
TASK_LABELS = {
    "qasper": "Qasper",
    "2wikimqa": "2WikiMQA",
    "qasper_e": "Qasper-E",
    "hotpotqa_e": "HotpotQA-E",
    "2wikimqa_e": "2WikiMQA-E",
}

# WandB run IDs (for reference / re-extraction)
WANDB_ENTITY = "SNLP_NAMM"
WANDB_PROJECT = "memory_evolution_hf"
M1_RUN_IDS = ["kz6vqo2o", "x9a4smmf", "qfoxxi2m"]
M2_MASKFIX_RUN_ID = "z5bo4n8k"
M3_MASKFIX_RUN_ID = "h0bzg6on"
B0_RUN_ID = "kz6vqo2o"


# ── Data loading ────────────────────────────────────────────────────────────


def load_data() -> dict:
    with open(DATA_PATH) as f:
        return json.load(f)


# ── Plot 1: Best val F1 comparison ─────────────────────────────────────────


def plot_best_val_f1_comparison(data: dict) -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    })

    b0 = data["B0"]
    m1 = data["M1"]
    m2 = data["M2_maskfix"]
    m3 = data["M3_maskfix"]

    conditions = ["B0", "M1", "M2", "M3"]
    cond_data = [b0, m1, m2, m3]
    cond_colors = ["#95a5a6", "#f39c12", "#3498db", "#e74c3c"]

    tasks_plus_mean = TASKS + ["mean"]
    labels_plus_mean = [TASK_LABELS[t] for t in TASKS] + ["Mean"]

    fig, ax = plt.subplots(figsize=(12, 6))
    n_cond = len(conditions)
    n_groups = len(tasks_plus_mean)
    width = 0.8 / n_cond
    x = np.arange(n_groups)

    for i, (cond, cd, color) in enumerate(zip(conditions, cond_data, cond_colors)):
        vals = [cd[t] for t in tasks_plus_mean]
        bars = ax.bar(x + i * width, vals, width, label=cond, color=color,
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7,
                    rotation=90)

    ax.set_xticks(x + width * n_cond / 2 - width / 2)
    ax.set_xticklabels(labels_plus_mean)
    ax.set_ylabel("Best Validation F1")
    ax.set_title("Best Validation F1 Comparison (Mask-Fix Runs)")
    ax.legend(loc="upper left", fontsize=9)
    max_val = max(
        cd[t] for cd in cond_data for t in tasks_plus_mean
    )
    ax.set_ylim(0, max_val * 1.2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "best_val_f1_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 2: Sensitivity bar ────────────────────────────────────────────────


def plot_sensitivity_bar(data: dict) -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
    })

    m1 = data["M1"]
    m3 = data["M3_maskfix"]
    task_labels = [TASK_LABELS[t] for t in TASKS]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(TASKS))
    width = 0.5

    sens = []
    for t in TASKS:
        if abs(m1[t]) < 1e-9:
            sens.append(0.0)
        else:
            sens.append((m3[t] - m1[t]) / m1[t] * 100)

    colors = ["#2ca02c" if s >= 0 else "#e74c3c" for s in sens]
    bars = ax.bar(x, sens, width, color=colors, edgecolor="white")
    for bar, val in zip(bars, sens):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f"{val:+.1f}%", ha="center", va=va, fontsize=9)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_ylabel("M3 Gain Over M1  (M3 - M1) / M1  [%]")
    ax.set_title("Per-Task M3 Gain (positive = M3 better than M1)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "sensitivity_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 3: Recovery ratio ─────────────────────────────────────────────────


def plot_recovery_ratio(data: dict) -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
    })

    m1 = data["M1"]
    m2 = data["M2_maskfix"]
    m3 = data["M3_maskfix"]
    task_labels = [TASK_LABELS[t] for t in TASKS] + ["Mean"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(TASKS) + 1)
    width = 0.5

    rr = []
    for t in TASKS:
        denom = m1[t] - m2[t]
        if abs(denom) < 1e-9:
            rr.append(float("nan"))
        else:
            rr.append((m3[t] - m2[t]) / denom)

    denom_mean = m1["mean"] - m2["mean"]
    if abs(denom_mean) < 1e-9:
        rr.append(float("nan"))
    else:
        rr.append((m3["mean"] - m2["mean"]) / denom_mean)

    bars = ax.bar(x, rr, width, color="#3498db", edgecolor="white")
    for bar, val in zip(bars, rr):
        if np.isnan(val):
            continue
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f"{val:.2f}", ha="center", va=va, fontsize=9)

    ax.axhline(1.0, color="green", linewidth=1.0, linestyle="--",
               label="Full recovery")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_ylabel("Recovery Ratio  (M3 - M2) / (M1 - M2)")
    ax.set_title("Recovery Ratio: How Much LoRA+NAMM Recovers vs LoRA-only")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "recovery_ratio.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Main ────────────────────────────────────────────────────────────────────


def extract_from_wandb() -> dict:
    """Re-extract maskfix_data.json from WandB."""
    import wandb

    api = wandb.Api()
    entity, project = WANDB_ENTITY, WANDB_PROJECT

    def best_per_task(run_ids, key_prefix, step_key, is_namm=False):
        """Get best per-task val F1 across run segments."""
        import pandas as pd
        frames = []
        avg_key = (f"{key_prefix}mean_f1" if is_namm
                   else f"{key_prefix}avg_f1")
        task_keys = [f"{key_prefix}{t}" for t in TASKS]
        all_keys = [step_key, avg_key] + task_keys
        for rid in run_ids:
            r = api.run(f"{entity}/{project}/{rid}")
            h = r.history(keys=all_keys, pandas=True, samples=10000)
            if not h.empty:
                h = h.dropna(subset=[avg_key])
                if frames:
                    prev_max = frames[-1][step_key].max()
                    h = h[h[step_key] > prev_max]
                frames.append(h)
        if not frames:
            return {}
        df = pd.concat(frames, ignore_index=True)
        idx = df[avg_key].idxmax()
        result = {t: float(df.loc[idx, f"{key_prefix}{t}"])
                  for t in TASKS}
        result["mean"] = float(df.loc[idx, avg_key])
        result["best_step"] = int(df.loc[idx, step_key])
        return result

    print("Extracting B0 baseline...")
    r = api.run(f"{entity}/{project}/{B0_RUN_ID}")
    keys = [f"lora/baseline_lb_{t}" for t in TASKS]
    h = r.history(keys=keys, pandas=True, samples=10000)
    h = h.dropna(subset=[keys[0]])
    b0 = {t: float(h.iloc[0][f"lora/baseline_lb_{t}"]) for t in TASKS}
    b0["mean"] = float(np.mean(list(b0.values())))

    print("Extracting M1...")
    m1 = best_per_task(M1_RUN_IDS, "lora/val_lb_", "lora/global_step")

    print("Extracting M2 maskfix...")
    m2 = best_per_task([M2_MASKFIX_RUN_ID], "val_lb/", "iter",
                       is_namm=True)

    print("Extracting M3 maskfix...")
    m3 = best_per_task([M3_MASKFIX_RUN_ID], "lora/val_lb_",
                       "lora/global_step")

    data = {
        "B0": b0,
        "M1": m1,
        "M2_maskfix": m2,
        "M3_maskfix": m3,
    }

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {DATA_PATH}")
    return data


def main() -> None:
    import sys
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--extract" in sys.argv:
        data = extract_from_wandb()
    else:
        data = load_data()
    print(f"Loaded data: {list(data.keys())}")

    plot_best_val_f1_comparison(data)
    plot_sensitivity_bar(data)
    plot_recovery_ratio(data)
    print(f"\nDone. All plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
