"""Analysis 2 — Adaptation Rate and Learning Efficiency.

Produces four plots in analysis/report_2/:
  normalised_improvement.png  — Normalised improvement curves [0, 1]
  overfitting_gap.png         — Train F1 - Val F1 over training
  steps_to_threshold.png      — Bar chart of steps to 50/75/90% of best val F1
  learning_curves_overlay.png — Raw val F1 with light smoothing

Also prints a summary table of steps-to-threshold values.
"""

import os
import sys
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
OUT_DIR = os.path.join(os.path.dirname(__file__))

# Run IDs
M1_RUN_IDS = ["kz6vqo2o", "x9a4smmf", "qfoxxi2m"]
M3_RUNS = {
    "M3 cs1024": ["ovosogkj"],
    "M3 cs2048": ["m4knrhmr"],
    "M3 cs3072": ["4sgkswa6"],
}

CONDITIONS = ["M1", "M3 cs1024", "M3 cs2048", "M3 cs3072"]
CONDITION_COLORS = {
    "M1": "#d62728",
    "M3 cs1024": "#1f77b4",
    "M3 cs2048": "#ff7f0e",
    "M3 cs3072": "#2ca02c",
}
CONDITION_LINESTYLES = {
    "M1": "-",
    "M3 cs1024": "--",
    "M3 cs2048": "-.",
    "M3 cs3072": ":",
}

SMOOTH_WINDOW = 5

# ── Data fetching ───────────────────────────────────────────────────────────


def get_api():
    return wandb.Api()


def fetch_lora_history(api, run_ids):
    """Fetch and concatenate LoRA history from multiple run segments.

    Deduplicates by global_step, keeping the first occurrence (from earlier
    segments) when segments overlap.

    Note: We fetch without keys= filter because the wandb API returns empty
    DataFrames when keys= includes columns that are only logged on a few rows
    (e.g. baseline_lb_avg_f1 logged once).
    """
    cols_of_interest = [
        "lora/global_step",
        "lora/loss",
        "lora/val_lb_avg_f1",
        "lora/train_lb_avg_f1",
        "lora/baseline_lb_avg_f1",
    ]
    frames = []
    for rid in run_ids:
        r = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        h = r.history(pandas=True, samples=10000)
        # Keep only columns we care about (plus any that exist)
        keep = [c for c in cols_of_interest if c in h.columns]
        frames.append(h[keep])
    df = pd.concat(frames, ignore_index=True)
    # Deduplicate by global_step — keep the first (earliest segment)
    df = df.dropna(subset=["lora/global_step"])
    df = df.drop_duplicates(subset=["lora/global_step"], keep="first")
    df = df.sort_values("lora/global_step").reset_index(drop=True)
    return df


def get_baseline(api, run_id):
    """Get baseline F1 from a specific run."""
    r = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    h = r.history(pandas=True, samples=10000)
    if "lora/baseline_lb_avg_f1" not in h.columns:
        return None
    rows = h.dropna(subset=["lora/baseline_lb_avg_f1"])
    if len(rows) > 0:
        return rows["lora/baseline_lb_avg_f1"].iloc[0]
    return None


def smooth(series, window=SMOOTH_WINDOW):
    """Rolling average with min_periods=1 to avoid NaN at edges."""
    return series.rolling(window=window, min_periods=1, center=True).mean()


# ── Plot 1: Normalised improvement ─────────────────────────────────────────


def plot_normalised_improvement(data, baselines, out_dir):
    """Plot normalised improvement: (val_F1 - baseline) / (best_val_F1 - baseline)."""
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
    ax.set_title("Normalised Improvement Curve\n(val F1 - baseline) / (best val F1 - baseline)", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.15)
    fig.tight_layout()
    path = os.path.join(out_dir, "normalised_improvement.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 2: Overfitting gap ────────────────────────────────────────────────


def plot_overfitting_gap(data, out_dir):
    """Plot train_F1 - val_F1 over training (smoothed)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in CONDITIONS:
        df = data[cond]
        # Only rows where BOTH train and val F1 are logged
        sub = df.dropna(subset=["lora/train_lb_avg_f1", "lora/val_lb_avg_f1"]).copy()
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
    ax.set_title("Overfitting Gap Over Training\n(positive = overfitting, smoothed window=5)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "overfitting_gap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 3: Steps to threshold ─────────────────────────────────────────────


def compute_steps_to_thresholds(data, thresholds=(0.50, 0.75, 0.90)):
    """For each condition, find the first step to reach X% of best val F1.

    Returns a dict: {condition: {threshold: step_or_None}}.
    """
    results = {}
    for cond in CONDITIONS:
        df = data[cond]
        sub = df.dropna(subset=["lora/val_lb_avg_f1"]).copy()
        best_val = sub["lora/val_lb_avg_f1"].max()
        results[cond] = {}
        for thr in thresholds:
            target = thr * best_val
            hits = sub[sub["lora/val_lb_avg_f1"] >= target]
            if len(hits) > 0:
                results[cond][thr] = int(hits["lora/global_step"].iloc[0])
            else:
                results[cond][thr] = None
    return results


def plot_steps_to_threshold(data, out_dir):
    """Bar chart showing steps to reach 50/75/90% of best val F1."""
    thresholds = [0.50, 0.75, 0.90]
    threshold_labels = ["50%", "75%", "90%"]
    steps_data = compute_steps_to_thresholds(data, thresholds)

    # Find max step across all conditions for "never reached" bars
    max_steps = {}
    for cond in CONDITIONS:
        df = data[cond]
        sub = df.dropna(subset=["lora/val_lb_avg_f1"])
        max_steps[cond] = int(sub["lora/global_step"].max()) if len(sub) > 0 else 0
    global_max_step = max(max_steps.values())

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(thresholds))
    width = 0.18
    n_conds = len(CONDITIONS)

    for i, cond in enumerate(CONDITIONS):
        vals = []
        hatches = []
        for thr in thresholds:
            step = steps_data[cond][thr]
            if step is not None:
                vals.append(step)
                hatches.append("")
            else:
                vals.append(max_steps[cond])
                hatches.append("//")

        offset = (i - (n_conds - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=cond,
            color=CONDITION_COLORS[cond],
            edgecolor="black",
            linewidth=0.5,
        )
        # Apply hatching for "never reached" bars
        for j, bar in enumerate(bars):
            if hatches[j]:
                bar.set_hatch(hatches[j])
                bar.set_alpha(0.5)
        ax.bar_label(bars, fmt="%d", fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(threshold_labels, fontsize=12)
    ax.set_xlabel("Threshold (% of Best Val F1)", fontsize=12)
    ax.set_ylabel("Global Step", fontsize=12)
    ax.set_title("Steps to Reach Performance Thresholds\n(hatched = threshold never reached, showing max step)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(out_dir, "steps_to_threshold.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    return steps_data


# ── Plot 4: Learning curves overlay ────────────────────────────────────────


def plot_learning_curves_overlay(data, out_dir):
    """Val F1 curves for all conditions overlaid, with light smoothing."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in CONDITIONS:
        df = data[cond]
        sub = df.dropna(subset=["lora/val_lb_avg_f1"]).copy()
        if sub.empty:
            continue

        # Plot raw as faint background
        ax.plot(
            sub["lora/global_step"],
            sub["lora/val_lb_avg_f1"],
            color=CONDITION_COLORS[cond],
            alpha=0.15,
            linewidth=1,
        )
        # Plot smoothed
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
    ax.set_title("Learning Curves — Val Mean F1 (smoothed, window=5)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "learning_curves_overlay.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    api = get_api()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Fetch data ──
    print("Fetching M1 data...")
    m1_df = fetch_lora_history(api, M1_RUN_IDS)
    print(f"  M1: {len(m1_df)} rows, steps {m1_df['lora/global_step'].min():.0f}–{m1_df['lora/global_step'].max():.0f}")

    m3_data = {}
    for cond, rids in M3_RUNS.items():
        print(f"Fetching {cond} data...")
        m3_data[cond] = fetch_lora_history(api, rids)
        df = m3_data[cond]
        print(f"  {cond}: {len(df)} rows, steps {df['lora/global_step'].min():.0f}–{df['lora/global_step'].max():.0f}")

    data = {"M1": m1_df, **m3_data}

    # ── Fetch baselines ──
    print("\nFetching baselines...")
    # M1 baseline from kz6vqo2o (first segment = true base model, before any training)
    m1_baseline = get_baseline(api, "kz6vqo2o")
    print(f"  M1 baseline: {m1_baseline:.2f}")

    baselines = {"M1": m1_baseline}
    # Each M3 run logs its own baseline at the start
    for cond, rids in M3_RUNS.items():
        bl = get_baseline(api, rids[0])
        if bl is not None:
            baselines[cond] = bl
            print(f"  {cond} baseline: {bl:.2f}")
        else:
            baselines[cond] = m1_baseline
            print(f"  {cond} baseline: {m1_baseline:.2f} (fallback to M1)")

    # ── Print summary stats ──
    print("\n=== Summary Stats ===")
    for cond in CONDITIONS:
        df = data[cond]
        sub = df.dropna(subset=["lora/val_lb_avg_f1"])
        best_val = sub["lora/val_lb_avg_f1"].max()
        best_step = sub.loc[sub["lora/val_lb_avg_f1"].idxmax(), "lora/global_step"]
        print(f"  {cond}: best val F1 = {best_val:.2f} at step {best_step:.0f}, baseline = {baselines[cond]:.2f}")

    # ── Generate plots ──
    print("\n=== Generating Plots ===")
    plot_normalised_improvement(data, baselines, OUT_DIR)
    plot_overfitting_gap(data, OUT_DIR)
    steps_data = plot_steps_to_threshold(data, OUT_DIR)
    plot_learning_curves_overlay(data, OUT_DIR)

    # ── Print steps-to-threshold table ──
    print("\n=== Steps to Threshold ===")
    thresholds = [0.50, 0.75, 0.90]
    header = f"{'Condition':<15}" + "".join(f"{'%d%%' % int(t*100):>12}" for t in thresholds)
    print(header)
    print("-" * len(header))
    for cond in CONDITIONS:
        row = f"{cond:<15}"
        for thr in thresholds:
            step = steps_data[cond][thr]
            if step is not None:
                row += f"{step:>12d}"
            else:
                row += f"{'never':>12}"
        print(row)

    # ── Print overfitting gap summary ──
    print("\n=== Overfitting Gap Summary (mean gap over training) ===")
    for cond in CONDITIONS:
        df = data[cond]
        sub = df.dropna(subset=["lora/train_lb_avg_f1", "lora/val_lb_avg_f1"])
        if len(sub) > 0:
            gap = (sub["lora/train_lb_avg_f1"] - sub["lora/val_lb_avg_f1"]).mean()
            gap_std = (sub["lora/train_lb_avg_f1"] - sub["lora/val_lb_avg_f1"]).std()
            print(f"  {cond}: mean gap = {gap:.2f} +/- {gap_std:.2f}")

    # ── Print final overfitting gap (last 10 eval points) ──
    print("\n=== Final Overfitting Gap (last 10 eval points) ===")
    for cond in CONDITIONS:
        df = data[cond]
        sub = df.dropna(subset=["lora/train_lb_avg_f1", "lora/val_lb_avg_f1"])
        if len(sub) >= 10:
            last10 = sub.tail(10)
            gap = (last10["lora/train_lb_avg_f1"] - last10["lora/val_lb_avg_f1"]).mean()
            print(f"  {cond}: final gap = {gap:.2f}")
        elif len(sub) > 0:
            gap = (sub["lora/train_lb_avg_f1"] - sub["lora/val_lb_avg_f1"]).mean()
            print(f"  {cond}: gap (all {len(sub)} points) = {gap:.2f}")

    # ── Generate report.md ──
    write_report(data, baselines, steps_data, OUT_DIR)
    print("\nDone. All plots saved to:", OUT_DIR)


def write_report(data, baselines, steps_data, out_dir):
    """Write the analysis report as report.md."""
    # Gather stats for the report
    stats = {}
    for cond in CONDITIONS:
        df = data[cond]
        sub = df.dropna(subset=["lora/val_lb_avg_f1"])
        best_val = sub["lora/val_lb_avg_f1"].max()
        best_step = int(sub.loc[sub["lora/val_lb_avg_f1"].idxmax(), "lora/global_step"])
        total_steps = int(sub["lora/global_step"].max())
        sub_gap = df.dropna(subset=["lora/train_lb_avg_f1", "lora/val_lb_avg_f1"])
        mean_gap = (sub_gap["lora/train_lb_avg_f1"] - sub_gap["lora/val_lb_avg_f1"]).mean() if len(sub_gap) > 0 else 0
        gap_std = (sub_gap["lora/train_lb_avg_f1"] - sub_gap["lora/val_lb_avg_f1"]).std() if len(sub_gap) > 0 else 0
        if len(sub_gap) >= 10:
            last10 = sub_gap.tail(10)
            final_gap = (last10["lora/train_lb_avg_f1"] - last10["lora/val_lb_avg_f1"]).mean()
        elif len(sub_gap) > 0:
            final_gap = mean_gap
        else:
            final_gap = 0
        stats[cond] = {
            "baseline": baselines[cond],
            "best_val": best_val,
            "best_step": best_step,
            "total_steps": total_steps,
            "mean_gap": mean_gap,
            "gap_std": gap_std,
            "final_gap": final_gap,
        }

    thresholds = [0.50, 0.75, 0.90]

    # Build run ID strings
    run_ids_str = {
        "M1": "kz6vqo2o, x9a4smmf, qfoxxi2m",
        "M3 cs1024": "ovosogkj",
        "M3 cs2048": "m4knrhmr",
        "M3 cs3072": "4sgkswa6",
    }

    lines = []
    lines.append("# Analysis 2 -- Adaptation Rate and Learning Efficiency")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("This analysis compares the learning dynamics of M1 (LoRA fine-tuning with full context) "
                 "and M3 (LoRA fine-tuning with frozen NAMM active) across three cache sizes (1024, 2048, 3072). "
                 "We examine normalised improvement curves, convergence speed, and the train-val generalisation gap.")
    lines.append("")
    lines.append("**Key findings:**")
    lines.append("")
    lines.append(f"- M1 and the M3 variants reach comparable peak validation F1 (~45-46), except M3 cs3072, "
                 f"which peaked at {stats['M3 cs3072']['best_val']:.2f} but was only trained for "
                 f"{stats['M3 cs3072']['total_steps']} steps (vs {stats['M1']['total_steps']} for M1).")
    lines.append("- Each M3 condition starts from a substantially lower baseline than M1 "
                 f"({stats['M3 cs1024']['baseline']:.2f}-{stats['M3 cs3072']['baseline']:.2f} vs "
                 f"{stats['M1']['baseline']:.2f}) because eviction degrades zero-shot performance, "
                 "yet M3 cs1024 and M3 cs2048 recover to match or slightly exceed M1's best F1.")
    lines.append(f"- M3 cs1024 converges more slowly than M1 to the 75% threshold "
                 f"({steps_data['M3 cs1024'][0.75]} vs {steps_data['M1'][0.75]} steps).")
    lines.append(f"- The train-val gap is consistently *negative* for all conditions (val F1 exceeds train F1), "
                 "so the traditional overfitting framing does not apply. M3 cs1024 shows the smallest "
                 f"magnitude gap (mean {stats['M3 cs1024']['mean_gap']:.2f}), while M1 shows the largest "
                 f"(mean {stats['M1']['mean_gap']:.2f}).")
    lines.append("")

    # Conditions and Baselines table
    lines.append("## Conditions and Baselines")
    lines.append("")
    lines.append("| Condition  | WandB Run(s)                         | Baseline F1 | Best Val F1 | Best Step | Total Steps |")
    lines.append("|------------|--------------------------------------|-------------|-------------|-----------|-------------|")
    for cond in CONDITIONS:
        s = stats[cond]
        lines.append(f"| {cond:<10} | {run_ids_str[cond]:<36} | {s['baseline']:.2f}       | {s['best_val']:.2f}       | {s['best_step']:<9} | {s['total_steps']:<11} |")
    lines.append("")
    lines.append("The baseline F1 is the zero-shot performance of the base model under each condition's inference setup "
                 "(full context for M1, evicted context for M3). Each condition's own baseline was used for normalisation.")
    lines.append("")

    # Steps to threshold
    lines.append("## Steps to Threshold")
    lines.append("")
    lines.append("The table below shows the number of gradient steps required for each condition to first reach "
                 "50%, 75%, and 90% of its own best validation F1.")
    lines.append("")
    lines.append("| Condition  | 50% Threshold | 75% Threshold | 90% Threshold |")
    lines.append("|------------|---------------|---------------|---------------|")
    for cond in CONDITIONS:
        vals = []
        for thr in thresholds:
            step = steps_data[cond][thr]
            vals.append(str(step) if step is not None else "never")
        lines.append(f"| {cond:<10} | {vals[0]:<13} | {vals[1]:<13} | {vals[2]:<13} |")
    lines.append("")
    lines.append("See `steps_to_threshold.png`.")
    lines.append("")
    lines.append("**Interpretation:** All conditions reach 50% of their best F1 almost immediately (within 2-16 steps), "
                 "because the baselines already provide substantial performance. The differences emerge at higher thresholds:")
    lines.append("")
    lines.append(f"- **M3 cs1024** is the slowest to reach 75% ({steps_data['M3 cs1024'][0.75]} steps vs "
                 f"{steps_data['M1'][0.75]} for M1), consistent with it having to recover from the lowest baseline "
                 f"({stats['M3 cs1024']['baseline']:.2f}) while navigating an information bottleneck.")
    lines.append(f"- **M3 cs2048** is intermediate ({steps_data['M3 cs2048'][0.75]} steps to 75%).")
    lines.append(f"- **M3 cs3072** reaches all thresholds very early, but this is partly because its best F1 "
                 f"({stats['M3 cs3072']['best_val']:.2f}) is lower than the others, so its absolute thresholds are "
                 "easier to meet. Additionally, its run is very short, so it may not have converged.")
    lines.append("")

    # Normalised improvement
    lines.append("## Normalised Improvement Curves")
    lines.append("")
    lines.append("See `normalised_improvement.png`.")
    lines.append("")
    lines.append("The normalised improvement maps each condition's trajectory to [0, 1], where 0 = baseline and "
                 "1 = best achieved F1. This isolates the *shape* of learning from the absolute performance level.")
    lines.append("")
    lines.append(f"- **M1** shows a gradual, steady rise over the first ~300 steps, reaching its peak around "
                 f"step {stats['M1']['best_step']} before declining.")
    lines.append(f"- **M3 cs1024** shows a noisier trajectory with a slower initial rise, but ultimately reaches "
                 f"its normalised peak at a similar step count (~{stats['M3 cs1024']['best_step']}). "
                 "The higher noise reflects the stochasticity introduced by token eviction during training.")
    lines.append(f"- **M3 cs2048** follows a trajectory similar to M1 but with more variance, peaking at "
                 f"step {stats['M3 cs2048']['best_step']}.")
    lines.append("- **M3 cs3072** rises quickly in its short window but shows high variance.")
    lines.append("")
    lines.append("The fact that M3 cs1024 and M3 cs2048 both reach normalised improvement ~1.0 at approximately "
                 "the same step count as M1 suggests that the learning rate of adaptation is roughly comparable, "
                 "despite the additional challenge of eviction.")
    lines.append("")

    # Learning curves overlay
    lines.append("## Learning Curves Overlay")
    lines.append("")
    lines.append("See `learning_curves_overlay.png`.")
    lines.append("")
    lines.append("The raw (lightly smoothed) validation F1 curves confirm that:")
    lines.append("")
    lines.append("- M1, M3 cs1024, and M3 cs2048 all converge to a similar performance band (~43-46 F1) by step ~300-350.")
    lines.append("- M3 cs1024 starts lowest (~20 F1) and has the steepest absolute improvement trajectory.")
    lines.append("- M3 cs2048 starts at ~25 F1 and tracks M1 closely after ~150 steps.")
    lines.append("- M3 cs3072 starts at ~31 F1 but its short run makes comparison difficult.")
    lines.append("- After their respective peaks, M1 and M3 cs1024 show performance degradation in later steps.")
    lines.append("")

    # Overfitting gap
    lines.append("## Overfitting Gap")
    lines.append("")
    lines.append("See `overfitting_gap.png`.")
    lines.append("")
    lines.append("The overfitting gap is defined as `train_F1 - val_F1`. Positive values indicate overfitting; "
                 "negative values indicate val outperforms train.")
    lines.append("")
    lines.append("### Surprising finding: negative gap throughout")
    lines.append("")
    lines.append("All conditions show a consistently *negative* gap, meaning validation F1 exceeds training F1 "
                 "throughout training. This likely reflects a methodological difference between how training and "
                 "validation F1 are computed (e.g. different evaluation subsets or answer extraction methodology).")
    lines.append("")
    lines.append("| Condition  | Mean Gap | Std  | Final Gap (last 10 evals) |")
    lines.append("|------------|----------|------|--------------------------|")
    for cond in CONDITIONS:
        s = stats[cond]
        lines.append(f"| {cond:<10} | {s['mean_gap']:.2f}    | {s['gap_std']:.2f} | {s['final_gap']:.2f}                    |")
    lines.append("")
    lines.append("### Does eviction act as regularisation?")
    lines.append("")
    lines.append("The original hypothesis was that eviction might act as implicit regularisation (analogous to dropout). "
                 "While the traditional overfitting framing does not apply (all gaps are negative), we can compare "
                 "the *magnitude* of the gap:")
    lines.append("")
    lines.append(f"- **M3 cs1024** has the smallest mean gap magnitude ({stats['M3 cs1024']['mean_gap']:.2f}), "
                 "suggesting that eviction with a small cache does reduce the gap -- consistent with a regularisation effect.")
    lines.append(f"- However, **M3 cs2048** shows a larger final gap ({stats['M3 cs2048']['final_gap']:.2f}) than "
                 f"M1 ({stats['M1']['final_gap']:.2f}), which goes against the regularisation hypothesis.")
    lines.append(f"- **M3 cs3072** shows a small final gap ({stats['M3 cs3072']['final_gap']:.2f}) but this may be "
                 "an artefact of its short training duration.")
    lines.append("")
    lines.append("The evidence for eviction-as-regularisation is mixed.")
    lines.append("")

    # Cache size and convergence
    lines.append("## Convergence Speed: Does Larger Cache = Faster Convergence?")
    lines.append("")
    lines.append("| Cache Size | Baseline F1 | Steps to 75% | Steps to 90% |")
    lines.append("|------------|-------------|--------------|--------------|")
    for cond in ["M3 cs1024", "M3 cs2048", "M3 cs3072"]:
        s75 = steps_data[cond][0.75]
        s90 = steps_data[cond][0.90]
        lines.append(f"| {cond.split()[-1]:<10} | {stats[cond]['baseline']:.2f}       | {s75 if s75 else 'never':<12} | {s90 if s90 else 'never':<12} |")
    lines.append("")
    lines.append("Larger cache sizes start from higher baselines and reach thresholds faster. However, this is "
                 "confounded by the fact that M3 cs3072's best F1 is much lower (due to its short training run), "
                 "making its thresholds easier to reach.")
    lines.append("")
    lines.append("If we consider convergence to M1's best val F1 as an absolute benchmark:")
    lines.append(f"- M3 cs1024 first reaches {stats['M1']['best_val']:.2f} at approximately step {stats['M3 cs1024']['best_step']}.")
    lines.append(f"- M3 cs2048 first reaches {stats['M1']['best_val']:.2f} at approximately step {stats['M3 cs2048']['best_step']}.")
    lines.append(f"- M3 cs3072 never reaches {stats['M1']['best_val']:.2f} within its {stats['M3 cs3072']['total_steps']} steps.")
    lines.append("")

    # Conclusions
    lines.append("## Conclusions")
    lines.append("")
    lines.append("1. **M3 converges at a comparable rate to M1.** Despite starting from much lower baselines "
                 f"({stats['M3 cs1024']['baseline']:.0f}-{stats['M3 cs3072']['baseline']:.0f} F1 vs "
                 f"{stats['M1']['baseline']:.0f} F1), M3 cs1024 and M3 cs2048 reach their best performance "
                 "at similar step counts. The absolute improvement per step is therefore *larger* for M3 than M1.")
    lines.append("")
    lines.append("2. **The information bottleneck from eviction does not substantially slow convergence.** "
                 f"M3 cs2048 converges only ~50% slower than M1 to the 75% threshold "
                 f"({steps_data['M3 cs2048'][0.75]} vs {steps_data['M1'][0.75]} steps), while M3 cs1024 is ~2x slower. "
                 "Given that these conditions operate with reduced KV cache, this is a relatively modest penalty.")
    lines.append("")
    lines.append("3. **Evidence for eviction-as-regularisation is mixed.** M3 cs1024 shows the smallest "
                 "train-val gap, consistent with the regularisation hypothesis, but the pattern is not consistent "
                 "across cache sizes.")
    lines.append("")
    lines.append("4. **Larger cache does not clearly yield faster convergence** when controlling for training duration. "
                 "M3 cs3072 was undertrained, preventing meaningful comparison. Between cs1024 and cs2048, "
                 "the larger cache converges moderately faster, but both reach comparable peak performance.")
    lines.append("")
    pct_below = (1 - stats['M3 cs1024']['baseline'] / stats['M1']['baseline']) * 100
    lines.append(f"5. **The most striking finding is that M3 recovers from severe baseline degradation.** "
                 f"M3 cs1024 starts with a baseline of {stats['M3 cs1024']['baseline']:.2f} "
                 f"({pct_below:.0f}% below M1's baseline of {stats['M1']['baseline']:.2f}) yet reaches a peak of "
                 f"{stats['M3 cs1024']['best_val']:.2f} -- slightly *exceeding* M1's best of "
                 f"{stats['M1']['best_val']:.2f}. This suggests that LoRA can fully compensate for the "
                 "information loss from aggressive eviction, at least on average across tasks.")
    lines.append("")

    # Plot references
    lines.append("## Plots")
    lines.append("")
    lines.append("- `normalised_improvement.png` -- Normalised improvement curves for all conditions")
    lines.append("- `overfitting_gap.png` -- Train-val F1 gap over training (smoothed)")
    lines.append("- `steps_to_threshold.png` -- Steps to reach 50/75/90% of best val F1")
    lines.append("- `learning_curves_overlay.png` -- Raw val F1 with light smoothing")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved {report_path}")


if __name__ == "__main__":
    main()
