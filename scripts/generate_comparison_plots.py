"""Generate comparison plots across cache sizes and conditions.

Produces:
  results/M2/comparison_val_f1.png       — M2 val mean F1 across cache sizes + baseline
  results/M2/comparison_bar.png          — Bar chart of best val F1 per task per cache size + baseline
  results/M3/comparison_val_f1.png       — M3 val mean F1 across cache sizes + M1
  results/M3/comparison_bar.png          — Bar chart of best val F1 per task, M3 cache sizes vs M1
  results/M3/comparison_loss.png         — Loss curves across cache sizes
"""

import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

ENTITY = "SNLP_NAMM"
PROJECT = "memory_evolution_hf"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

TASKS = ["qasper", "2wikimqa", "qasper_e", "hotpotqa_e", "2wikimqa_e"]
TASK_LABELS = {"qasper": "Qasper", "2wikimqa": "2WikiMQA", "qasper_e": "Qasper-E",
               "hotpotqa_e": "HotpotQA-E", "2wikimqa_e": "2WikiMQA-E"}

CS_COLORS = {"1024": "#1f77b4", "2048": "#ff7f0e", "3072": "#2ca02c"}
M1_COLOR = "#d62728"
BASELINE_COLOR = "#7f7f7f"


def get_api():
    return wandb.Api()


def fetch_lora_history(api, run_ids):
    frames = []
    for rid in run_ids:
        r = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        keys = (
            ["lora/global_step", "lora/loss"]
            + [f"lora/val_lb_{t}" for t in TASKS]
            + ["lora/val_lb_avg_f1"]
            + [f"lora/train_lb_{t}" for t in TASKS]
            + ["lora/train_lb_avg_f1"]
        )
        h = r.history(keys=keys, pandas=True, samples=10000)
        if len(frames) > 0 and len(h) > 0:
            prev_max = frames[-1]["lora/global_step"].max()
            h = h[h["lora/global_step"] > prev_max]
        frames.append(h)
    return pd.concat(frames, ignore_index=True)


def fetch_namm_history(api, run_ids):
    """Fetch and concatenate history from one or more NAMM run segments."""
    keys = (
        ["iter"]
        + [f"val_lb/{t}" for t in TASKS]
        + ["val_tasks_aggregate"]
        + [f"train_lb/{t}" for t in TASKS]
        + ["train_tasks_aggregate"]
    )
    frames = []
    for rid in run_ids:
        r = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        h = r.history(keys=keys, pandas=True, samples=10000)
        if len(frames) > 0 and len(h) > 0:
            prev_max = frames[-1]["iter"].max()
            h = h[h["iter"] > prev_max]
        frames.append(h)
    return pd.concat(frames, ignore_index=True)


def get_baseline(api):
    """Get baseline F1 values from the start of the M1 run."""
    r = api.run(f"{ENTITY}/{PROJECT}/qfoxxi2m")
    h = r.history(
        keys=[f"lora/baseline_lb_{t}" for t in TASKS] + ["lora/baseline_lb_avg_f1"],
        pandas=True, samples=10000,
    )
    row = h.dropna(subset=["lora/baseline_lb_avg_f1"]).iloc[0]
    return {t: row[f"lora/baseline_lb_{t}"] for t in TASKS}, row["lora/baseline_lb_avg_f1"]


def get_best_namm_val(df):
    """Get best validation scores from a NAMM run history."""
    sub = df.dropna(subset=["val_tasks_aggregate"])
    if sub.empty:
        return {}, 0, 0
    # Compute mean F1 for each row
    sub = sub.copy()
    sub["mean_f1"] = sum(sub[f"val_lb/{t}"] for t in TASKS) / len(TASKS)
    best_idx = sub["mean_f1"].idxmax()
    best = sub.loc[best_idx]
    scores = {t: best[f"val_lb/{t}"] for t in TASKS}
    return scores, best["mean_f1"], best["iter"]


def get_best_lora_val(df):
    """Get best validation scores from a LoRA run history."""
    sub = df.dropna(subset=["lora/val_lb_avg_f1"])
    if sub.empty:
        return {}, 0, 0
    best_idx = sub["lora/val_lb_avg_f1"].idxmax()
    best = sub.loc[best_idx]
    scores = {t: best[f"lora/val_lb_{t}"] for t in TASKS}
    return scores, best["lora/val_lb_avg_f1"], best["lora/global_step"]


# ============================================================
# M2 comparison plots
# ============================================================

def plot_m2_comparison_curves(namm_data, baseline_mean, m1_mean, out_dir):
    """Overlay val mean F1 curves for M2 across cache sizes, with B0 and M1 baselines."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cs, df in sorted(namm_data.items()):
        sub = df.dropna(subset=["val_tasks_aggregate"])
        sub = sub.copy()
        sub["mean_f1"] = sum(sub[f"val_lb/{t}"] for t in TASKS) / len(TASKS)
        ax.plot(sub["iter"], sub["mean_f1"], label=f"M2 cache={cs}", color=CS_COLORS[cs], linewidth=2)

    ax.axhline(y=baseline_mean, color=BASELINE_COLOR, linestyle="--", linewidth=2, label="B0 (base model, full cache)")
    ax.axhline(y=m1_mean, color=M1_COLOR, linestyle="--", linewidth=2, label="M1 (LoRA, full cache)")
    ax.set_xlabel("CMA-ES Iteration", fontsize=12)
    ax.set_ylabel("Val Mean F1", fontsize=12)
    ax.set_title("M2 — Standalone NAMM: Val Mean F1 Across Cache Sizes", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_val_f1.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved M2/comparison_val_f1.png")


def plot_m2_comparison_bar(namm_data, baseline_scores, baseline_mean, m1_scores, m1_mean, out_dir):
    """Bar chart: best val F1 per task for each M2 cache size + B0 + M1."""
    fig, ax = plt.subplots(figsize=(14, 6))

    labels = list(TASK_LABELS.values()) + ["Mean"]
    cache_sizes = sorted(namm_data.keys())
    x = np.arange(len(labels))
    width = 0.15
    n_bars = len(cache_sizes) + 2  # +B0 +M1

    # B0 baseline
    baseline_vals = [baseline_scores[t] for t in TASKS] + [baseline_mean]
    offset = -(n_bars - 1) / 2 * width
    bars = ax.bar(x + offset, baseline_vals, width, label="B0 (base model)", color=BASELINE_COLOR, alpha=0.7)
    ax.bar_label(bars, fmt="%.1f", fontsize=6, padding=2)

    # M1
    m1_vals = [m1_scores.get(t, 0) for t in TASKS] + [m1_mean]
    offset = (-(n_bars - 1) / 2 + 1) * width
    bars = ax.bar(x + offset, m1_vals, width, label="M1 (LoRA, full cache)", color=M1_COLOR)
    ax.bar_label(bars, fmt="%.1f", fontsize=6, padding=2)

    # M2 per cache size
    for i, cs in enumerate(cache_sizes):
        scores, mean_f1, _ = get_best_namm_val(namm_data[cs])
        vals = [scores.get(t, 0) for t in TASKS] + [mean_f1]
        offset = (-(n_bars - 1) / 2 + i + 2) * width
        bars = ax.bar(x + offset, vals, width, label=f"M2 cache={cs}", color=CS_COLORS[cs])
        ax.bar_label(bars, fmt="%.1f", fontsize=6, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Best Val F1", fontsize=12)
    ax.set_title("M2 — Best Val F1 per Task (NAMM vs B0 vs M1)", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_bar.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved M2/comparison_bar.png")


# ============================================================
# M3 comparison plots
# ============================================================

def plot_m3_comparison_curves(m3_data, m1_df, out_dir):
    """Overlay val mean F1 curves for M3 cache sizes + M1."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # M1
    m1_sub = m1_df.dropna(subset=["lora/val_lb_avg_f1"])
    ax.plot(m1_sub["lora/global_step"], m1_sub["lora/val_lb_avg_f1"],
            label="M1 (no NAMM)", color=M1_COLOR, linewidth=2, alpha=0.8)

    # M3 per cache size
    for cs, df in sorted(m3_data.items()):
        sub = df.dropna(subset=["lora/val_lb_avg_f1"])
        ax.plot(sub["lora/global_step"], sub["lora/val_lb_avg_f1"],
                label=f"M3 cache={cs}", color=CS_COLORS[cs], linewidth=2)

    ax.set_xlabel("Global Step", fontsize=12)
    ax.set_ylabel("Val Mean F1", fontsize=12)
    ax.set_title("M3 vs M1 — Val Mean F1 Over Training", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_val_f1.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved M3/comparison_val_f1.png")


def plot_m3_comparison_bar(m3_data, m1_df, baseline_scores, baseline_mean, out_dir):
    """Bar chart: best val F1 per task for M3 cache sizes + M1 + baseline."""
    fig, ax = plt.subplots(figsize=(14, 6))

    labels = list(TASK_LABELS.values()) + ["Mean"]
    cache_sizes = sorted(m3_data.keys())
    x = np.arange(len(labels))
    n_bars = len(cache_sizes) + 2  # +baseline +M1
    width = 0.15

    # Baseline
    baseline_vals = [baseline_scores[t] for t in TASKS] + [baseline_mean]
    offset = -(n_bars - 1) / 2 * width
    bars = ax.bar(x + offset, baseline_vals, width, label="B0 baseline", color=BASELINE_COLOR, alpha=0.7)
    ax.bar_label(bars, fmt="%.1f", fontsize=6, padding=2)

    # M1
    m1_scores, m1_mean, _ = get_best_lora_val(m1_df)
    m1_vals = [m1_scores.get(t, 0) for t in TASKS] + [m1_mean]
    offset = (-(n_bars - 1) / 2 + 1) * width
    bars = ax.bar(x + offset, m1_vals, width, label="M1 (no NAMM)", color=M1_COLOR)
    ax.bar_label(bars, fmt="%.1f", fontsize=6, padding=2)

    # M3 per cache size
    for i, cs in enumerate(cache_sizes):
        scores, mean_f1, _ = get_best_lora_val(m3_data[cs])
        vals = [scores.get(t, 0) for t in TASKS] + [mean_f1]
        offset = (-(n_bars - 1) / 2 + i + 2) * width
        bars = ax.bar(x + offset, vals, width, label=f"M3 cache={cs}", color=CS_COLORS[cs])
        ax.bar_label(bars, fmt="%.1f", fontsize=6, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Best Val F1", fontsize=12)
    ax.set_title("M3 vs M1 vs Baseline — Best Val F1 per Task", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_bar.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved M3/comparison_bar.png")


def plot_m3_comparison_loss(m3_data, m1_df, out_dir):
    """Loss curves for M3 cache sizes + M1."""
    fig, ax = plt.subplots(figsize=(10, 5))

    m1_sub = m1_df.dropna(subset=["lora/loss"])
    ax.plot(m1_sub["lora/global_step"], m1_sub["lora/loss"],
            label="M1 (no NAMM)", color=M1_COLOR, linewidth=1.5, alpha=0.7)

    for cs, df in sorted(m3_data.items()):
        sub = df.dropna(subset=["lora/loss"])
        ax.plot(sub["lora/global_step"], sub["lora/loss"],
                label=f"M3 cache={cs}", color=CS_COLORS[cs], linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Global Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("M3 vs M1 — Training Loss", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_loss.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved M3/comparison_loss.png")


def main():
    api = get_api()

    # Fetch baseline
    print("Fetching baseline...")
    baseline_scores, baseline_mean = get_baseline(api)
    print(f"  Baseline mean F1: {baseline_mean:.2f}")

    # --- M2 ---
    print("\n=== M2 Comparison Plots ===")
    m2_out = os.path.join(RESULTS_DIR, "M2")
    os.makedirs(m2_out, exist_ok=True)

    m2_ids = {"1024": ["lenhmfb1"], "2048": ["y5fdw0f9", "ccflnsds"], "3072": ["quc95irz"]}
    m2_data = {}
    for cs, rids in m2_ids.items():
        print(f"  Fetching M2 cs{cs} ({rids})...")
        m2_data[cs] = fetch_namm_history(api, rids)

    # Fetch M1 (needed for both M2 and M3 comparisons)
    print("  Fetching M1...")
    m1_df = fetch_lora_history(api, ["kz6vqo2o", "x9a4smmf", "qfoxxi2m"])
    m1_scores, m1_mean, _ = get_best_lora_val(m1_df)
    print(f"  M1 best val mean F1: {m1_mean:.2f}")

    plot_m2_comparison_curves(m2_data, baseline_mean, m1_mean, m2_out)
    plot_m2_comparison_bar(m2_data, baseline_scores, baseline_mean, m1_scores, m1_mean, m2_out)

    # --- M3 ---
    print("\n=== M3 Comparison Plots ===")
    m3_out = os.path.join(RESULTS_DIR, "M3")
    os.makedirs(m3_out, exist_ok=True)

    m3_ids = {"1024": ["ovosogkj"], "2048": ["m4knrhmr"], "3072": ["4sgkswa6"]}
    m3_data = {}
    for cs, rids in m3_ids.items():
        print(f"  Fetching M3 cs{cs} ({rids})...")
        m3_data[cs] = fetch_lora_history(api, rids)

    plot_m3_comparison_curves(m3_data, m1_df, m3_out)
    plot_m3_comparison_bar(m3_data, m1_df, baseline_scores, baseline_mean, m3_out)
    plot_m3_comparison_loss(m3_data, m1_df, m3_out)

    print("\nDone.")


if __name__ == "__main__":
    main()
