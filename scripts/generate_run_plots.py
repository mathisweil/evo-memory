"""Generate training plots for all completed wandb runs.

Reads run history from wandb and produces per-run plot sets in results/<run_label>/:
  - val_f1.png       : validation F1 per task + mean
  - train_f1.png     : train F1 per task + mean
  - loss.png         : training loss curve
For M2 NAMM runs (no LoRA), produces:
  - val_f1.png       : validation F1 per task + aggregate
  - train_f1.png     : train F1 per task + aggregate
  - cache_stats.png  : dynamic cache size over iterations
"""

import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import wandb

ENTITY = "SNLP_NAMM"
PROJECT = "memory_evolution_hf"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

TASKS = ["qasper", "2wikimqa", "qasper_e", "hotpotqa_e", "2wikimqa_e"]

# --- Run definitions (from experiment_specification.md §6) ---

LORA_RUNS = {
    # M1 — LoRA only (3 segments, concatenated)
    "M1_lora_r8": {
        "ids": ["kz6vqo2o", "x9a4smmf", "qfoxxi2m"],
        "title": "M1 — LoRA r=8 (no NAMM, full context)",
    },
    # M3 — LoRA + frozen NAMM
    "M3_cs1024": {
        "ids": ["ovosogkj"],
        "title": "M3 — LoRA + frozen NAMM (cache=1024)",
    },
    "M3_cs2048": {
        "ids": ["m4knrhmr"],
        "title": "M3 — LoRA + frozen NAMM (cache=2048)",
    },
    "M3_cs3072": {
        "ids": ["4sgkswa6"],
        "title": "M3 — LoRA + frozen NAMM (cache=3072)",
    },
}

NAMM_RUNS = {
    "M2_cs1024": {
        "id": "lenhmfb1",
        "title": "M2 — Standalone NAMM (cache=1024)",
    },
    "M2_cs2048": {
        "id": "ccflnsds",
        "title": "M2 — Standalone NAMM (cache=2048)",
    },
    "M2_cs3072": {
        "id": "quc95irz",
        "title": "M2 — Standalone NAMM (cache=3072)",
    },
}

COLORS = {
    "qasper": "#1f77b4",
    "2wikimqa": "#ff7f0e",
    "qasper_e": "#2ca02c",
    "hotpotqa_e": "#d62728",
    "2wikimqa_e": "#9467bd",
    "mean": "#333333",
}


def fetch_lora_history(run_ids):
    """Fetch and concatenate history from one or more LoRA run segments."""
    api = wandb.Api()
    frames = []
    step_offset = 0
    for rid in run_ids:
        r = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        keys = (
            ["lora/global_step", "lora/epoch", "lora/loss", "lora/grad_norm", "lora/lr"]
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
    df = pd.concat(frames, ignore_index=True)
    return df


def fetch_namm_history(run_id):
    """Fetch history from a NAMM CMA-ES run."""
    api = wandb.Api()
    r = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    keys = (
        ["iter"]
        + [f"val_lb/{t}" for t in TASKS]
        + ["val_tasks_aggregate"]
        + [f"train_lb/{t}" for t in TASKS]
        + ["train_tasks_aggregate"]
        + ["mem_stats/layer_id_0/dynamic_cache_sizes", "mem_stats/layer_id_0/final_dynamic_cache_sizes"]
        + ["evo_stats/step_size"]
    )
    h = r.history(keys=keys, pandas=True, samples=10000)
    return h


def plot_lora_f1(df, split, title, out_path):
    """Plot per-task F1 for a LoRA run (val or train)."""
    prefix = f"lora/{split}_lb_"
    avg_col = f"lora/{split}_lb_avg_f1"

    sub = df.dropna(subset=[avg_col])
    if sub.empty:
        print(f"  Skipping {split} F1 — no data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = sub["lora/global_step"]

    for task in TASKS:
        col = f"{prefix}{task}"
        if col in sub.columns:
            vals = sub[col]
            ax.plot(x, vals, label=task, color=COLORS[task], alpha=0.7, linewidth=1)

    ax.plot(x, sub[avg_col], label="mean", color=COLORS["mean"], linewidth=2.5, linestyle="--")

    ax.set_xlabel("Global Step")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"{title}\n{split.capitalize()} F1 per Task")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_lora_loss(df, title, out_path):
    """Plot training loss for a LoRA run."""
    sub = df.dropna(subset=["lora/loss"])
    if sub.empty:
        print("  Skipping loss — no data")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sub["lora/global_step"], sub["lora/loss"], color="#1f77b4", linewidth=1, alpha=0.8)
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title}\nTraining Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_namm_f1(df, split, title, out_path):
    """Plot per-task F1 for a NAMM run (val or train)."""
    prefix = f"{split}_lb/"
    agg_col = f"{split}_tasks_aggregate"

    sub = df.dropna(subset=[agg_col])
    if sub.empty:
        print(f"  Skipping {split} F1 — no data")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]})

    x = sub["iter"]

    ax = axes[0]
    for task in TASKS:
        col = f"{prefix}{task}"
        if col in sub.columns:
            ax.plot(x, sub[col], label=task, color=COLORS[task], alpha=0.7, linewidth=1)

    ax.set_ylabel("F1 Score")
    ax.set_title(f"{title}\n{split.capitalize()} F1 per Task")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(x, sub[agg_col] * 10000, color=COLORS["mean"], linewidth=2)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Aggregate (x1e4)")
    ax2.set_title("Tasks Aggregate")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_namm_cache(df, title, out_path):
    """Plot dynamic cache size over iterations for a NAMM run."""
    dyn_col = "mem_stats/layer_id_0/dynamic_cache_sizes"
    fin_col = "mem_stats/layer_id_0/final_dynamic_cache_sizes"

    sub = df.dropna(subset=[dyn_col])
    if sub.empty:
        print("  Skipping cache stats — no data")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sub["iter"], sub[dyn_col], label="Avg dynamic cache", color="#1f77b4", linewidth=1.5)
    if fin_col in sub.columns:
        ax.plot(sub["iter"], sub[fin_col], label="Final cache size", color="#ff7f0e", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Tokens")
    ax.set_title(f"{title}\nKV Cache Size (Layer 0)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- LoRA runs (M1, M3) ---
    for label, info in LORA_RUNS.items():
        out_dir = os.path.join(RESULTS_DIR, label)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== {label}: {info['title']} ===")
        print(f"  Fetching history from {info['ids']}...")

        df = fetch_lora_history(info["ids"])
        print(f"  Got {len(df)} rows")

        plot_lora_f1(df, "val", info["title"], os.path.join(out_dir, "val_f1.png"))
        plot_lora_f1(df, "train", info["title"], os.path.join(out_dir, "train_f1.png"))
        plot_lora_loss(df, info["title"], os.path.join(out_dir, "loss.png"))

    # --- NAMM runs (M2) ---
    for label, info in NAMM_RUNS.items():
        out_dir = os.path.join(RESULTS_DIR, label)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== {label}: {info['title']} ===")
        print(f"  Fetching history from {info['id']}...")

        df = fetch_namm_history(info["id"])
        print(f"  Got {len(df)} rows")

        plot_namm_f1(df, "val", info["title"], os.path.join(out_dir, "val_f1.png"))
        plot_namm_f1(df, "train", info["title"], os.path.join(out_dir, "train_f1.png"))
        plot_namm_cache(df, info["title"], os.path.join(out_dir, "cache_stats.png"))

    print(f"\nAll plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
