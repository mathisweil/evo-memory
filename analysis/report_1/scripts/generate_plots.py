"""Report 1 -- Per-Task Eviction Sensitivity (Mask-Fix Runs).

Loads pre-extracted val and test F1 data from maskfix_data.json and generates:
  1. best_f1_comparison.png    -- two-panel (val / test) grouped bar: B0, M1, M2, M3
  2. sensitivity_bar.png       -- M3 gain over M1, val vs test side by side
  3. recovery_ratio.png        -- recovery ratio, val vs test side by side

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

# -- Config ------------------------------------------------------------------

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

CONDITIONS = ["B0", "M1", "M2", "M3"]
COND_COLORS = ["#95a5a6", "#f39c12", "#3498db", "#e74c3c"]


# -- Data loading ------------------------------------------------------------


def load_data() -> dict:
    with open(DATA_PATH) as f:
        return json.load(f)


def _get_val_cond_data(data: dict) -> list[dict]:
    """Return [B0, M1, M2_maskfix, M3_maskfix] val dicts."""
    return [data["B0"], data["M1"], data["M2_maskfix"], data["M3_maskfix"]]


def _get_test_cond_data(data: dict) -> list[dict]:
    """Return [B0_test, M1_test, M2_test, M3_test] dicts."""
    return [data["B0_test"], data["M1_test"], data["M2_test"], data["M3_test"]]


# -- Plot 1: Best F1 comparison (val + test two-panel) ----------------------


def plot_best_f1_comparison(data: dict) -> None:
    val_data = _get_val_cond_data(data)
    test_data = _get_test_cond_data(data)

    tasks_plus_mean = TASKS + ["mean"]
    labels_plus_mean = [TASK_LABELS[t] for t in TASKS] + ["Mean"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, split_data, split_label in [
        (axes[0], val_data, "Validation"),
        (axes[1], test_data, "Test"),
    ]:
        n_cond = len(CONDITIONS)
        n_groups = len(tasks_plus_mean)
        width = 0.8 / n_cond
        x = np.arange(n_groups)

        for i, (cond, cd, color) in enumerate(
            zip(CONDITIONS, split_data, COND_COLORS)
        ):
            vals = [cd[t] for t in tasks_plus_mean]
            bars = ax.bar(
                x + i * width, vals, width, label=cond, color=color,
                edgecolor="white", linewidth=0.5,
            )
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8,
                )

        ax.set_xticks(x + width * n_cond / 2 - width / 2)
        ax.set_xticklabels(labels_plus_mean, fontsize=11)
        ax.set_ylabel("F1", fontsize=12)
        ax.set_title(f"Best {split_label} F1", fontsize=14)
        ax.legend(loc="upper left", fontsize=10)
        max_val = max(cd[t] for cd in split_data for t in tasks_plus_mean)
        ax.set_ylim(0, max_val * 1.3)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "best_f1_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# -- Plot 2: Sensitivity bar (val + test side by side) ----------------------


def _compute_sensitivity(m1: dict, m3: dict) -> list[float]:
    sens = []
    for t in TASKS:
        if abs(m1[t]) < 1e-9:
            sens.append(0.0)
        else:
            sens.append((m3[t] - m1[t]) / m1[t] * 100)
    return sens


def plot_sensitivity_bar(data: dict) -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
    })

    val_sens = _compute_sensitivity(data["M1"], data["M3_maskfix"])
    test_sens = _compute_sensitivity(data["M1_test"], data["M3_test"])

    task_labels = [TASK_LABELS[t] for t in TASKS]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(TASKS))
    width = 0.35

    # Val bars
    val_colors = ["#2ca02c" if s >= 0 else "#e74c3c" for s in val_sens]
    bars_val = ax.bar(
        x - width / 2, val_sens, width, color=val_colors,
        edgecolor="white", label="Val", alpha=0.85,
    )
    # Test bars
    test_colors = ["#2ca02c" if s >= 0 else "#e74c3c" for s in test_sens]
    bars_test = ax.bar(
        x + width / 2, test_sens, width, color=test_colors,
        edgecolor="white", alpha=0.85, hatch="//",
    )

    for bar, val in zip(bars_val, val_sens):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2, y,
            f"{val:+.1f}%", ha="center", va=va, fontsize=8,
        )
    for bar, val in zip(bars_test, test_sens):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2, y,
            f"{val:+.1f}%", ha="center", va=va, fontsize=8,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_ylabel("M3 Gain Over M1  (M3 - M1) / M1  [%]")
    ax.set_title("Per-Task M3 Gain (positive = M3 better than M1)")

    # Custom legend: solid = val, hatched = test
    import matplotlib.patches as mpatches
    val_patch = mpatches.Patch(facecolor="#888888", alpha=0.85, label="Val")
    test_patch = mpatches.Patch(
        facecolor="#888888", alpha=0.85, hatch="//", label="Test",
    )
    ax.legend(handles=[val_patch, test_patch], fontsize=9)

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "sensitivity_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# -- Plot 3: Recovery ratio (val + test side by side) ----------------------


def _compute_recovery(m1: dict, m2: dict, m3: dict) -> list[float]:
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
    return rr


def plot_recovery_ratio(data: dict) -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
    })

    val_rr = _compute_recovery(
        data["M1"], data["M2_maskfix"], data["M3_maskfix"],
    )
    test_rr = _compute_recovery(
        data["M1_test"], data["M2_test"], data["M3_test"],
    )

    task_labels = [TASK_LABELS[t] for t in TASKS] + ["Mean"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(TASKS) + 1)
    width = 0.35

    bars_val = ax.bar(
        x - width / 2, val_rr, width, color="#3498db",
        edgecolor="white", label="Val", alpha=0.85,
    )
    bars_test = ax.bar(
        x + width / 2, test_rr, width, color="#3498db",
        edgecolor="white", label="Test", alpha=0.85, hatch="//",
    )

    for bar, val in zip(bars_val, val_rr):
        if np.isnan(val):
            continue
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2, y,
            f"{val:.2f}", ha="center", va=va, fontsize=8,
        )
    for bar, val in zip(bars_test, test_rr):
        if np.isnan(val):
            continue
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2, y,
            f"{val:.2f}", ha="center", va=va, fontsize=8,
        )

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


# -- WandB extraction (unchanged) -------------------------------------------


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


# -- Main --------------------------------------------------------------------


def main() -> None:
    import sys
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--extract" in sys.argv:
        data = extract_from_wandb()
    else:
        data = load_data()
    print(f"Loaded data: {list(data.keys())}")

    plot_best_f1_comparison(data)
    plot_sensitivity_bar(data)
    plot_recovery_ratio(data)
    print(f"\nDone. All plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
