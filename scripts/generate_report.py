#!/usr/bin/env python3
"""Generate a report for an experiment.

Aggregates results across all methods/runs, generates comparison plots
and a summary report.json. Can be re-run at any time to update the report
as new results become available.

Usage:
    python scripts/generate_report.py experiment_1
    python scripts/generate_report.py 1          # shorthand
    python scripts/generate_report.py --all       # all active experiments
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")
MANIFEST_PATH = os.path.join(EXPERIMENTS_DIR, "manifest.json")


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"experiments": {}}


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def normalize_name(name):
    if name.isdigit():
        return f"experiment_{name}"
    return name


def collect_runs(experiment_dir):
    """Walk experiment_dir/method/run_name/ and collect results.

    Also discovers eval-only subdirectories (e.g. eval_namm_c1024/)
    within training run directories.
    """
    runs = []
    for method in sorted(os.listdir(experiment_dir)):
        method_dir = os.path.join(experiment_dir, method)
        if not os.path.isdir(method_dir) or method in ("plots",):
            continue
        for run_name in sorted(os.listdir(method_dir)):
            run_dir = os.path.join(method_dir, run_name)
            results_path = os.path.join(run_dir, "results.json")
            if not os.path.isfile(results_path):
                continue
            with open(results_path) as f:
                results = json.load(f)

            # Load examples if present
            examples_path = os.path.join(run_dir, "examples.json")
            examples = None
            if os.path.isfile(examples_path):
                with open(examples_path) as f:
                    examples = json.load(f)

            runs.append({
                "method": method,
                "run_name": run_name,
                "results": results,
                "examples": examples,
                "run_dir": run_dir,
            })

            # Scan for eval-only subdirectories (eval_*)
            for sub in sorted(os.listdir(run_dir)):
                sub_dir = os.path.join(run_dir, sub)
                if not os.path.isdir(sub_dir) or not sub.startswith("eval_"):
                    continue
                sub_results_path = os.path.join(sub_dir, "results.json")
                if not os.path.isfile(sub_results_path):
                    continue
                with open(sub_results_path) as f:
                    sub_results = json.load(f)
                runs.append({
                    "method": method,
                    "run_name": f"{run_name}/{sub}",
                    "results": sub_results,
                    "examples": None,
                    "run_dir": sub_dir,
                })
    return runs


def _extract_scores(results):
    """Extract final and baseline score dicts from a results dict."""
    if results.get("type") == "eval":
        return results.get("scores", {}), {}, {}
    full_eval = results.get("full_eval", {})
    baseline = results.get("baseline_eval", {})
    training = results.get("training", {})
    return full_eval.get("scores", {}), baseline.get("scores", {}), training


def _get_qasper_f1(scores):
    """Get qasper F1 from a scores dict, or None if not present."""
    return scores.get("lb/qasper")


def build_summary(runs):
    """Build a comparison table from all runs."""
    rows = []
    for run in runs:
        r = run["results"]
        is_eval_only = r.get("type") == "eval"
        config = r.get("config", {})
        final_scores, baseline_scores, training = _extract_scores(r)

        final_f1 = _get_qasper_f1(final_scores)
        baseline_f1 = _get_qasper_f1(baseline_scores)

        rows.append({
            "method": run["method"],
            "run_name": run["run_name"],
            "type": "eval" if is_eval_only else "training",
            "final_f1": final_f1,
            "baseline_f1": baseline_f1,
            "improvement": round(final_f1 - baseline_f1, 2)
                if final_f1 is not None and baseline_f1 is not None else None,
            "iterations": training.get("iterations"),
            "training_time_h": training.get("total_time_h"),
            "mini_batch_size": config.get("mini_batch_size"),
            "cache_size": config.get("cache_size"),
            "sigma": config.get("sigma"),
            "alpha": config.get("alpha"),
            "population_size": config.get("population_size"),
        })
    return rows


def generate_plots(experiment_dir, runs):
    """Generate comparison plots for the experiment."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    plots_dir = os.path.join(experiment_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Group runs by method
    methods = {}
    for run in runs:
        methods.setdefault(run["method"], []).append(run)

    # Plot 1: F1 comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    all_labels = []
    all_baseline = []
    all_final = []
    all_types = []
    colors_final = []
    method_colors = {"es_namm": "steelblue", "es_only": "seagreen",
                     "namm_only": "coral"}

    for method, method_runs in sorted(methods.items()):
        for run in method_runs:
            r = run["results"]
            is_eval_only = r.get("type") == "eval"
            final_scores, baseline_scores, _ = _extract_scores(r)

            f1 = _get_qasper_f1(final_scores)
            if f1 is not None:
                label = f"{method}/{run['run_name']}"
                all_labels.append(label)
                all_final.append(f1)
                all_baseline.append(_get_qasper_f1(baseline_scores) or 0)
                all_types.append("eval" if is_eval_only else "training")
                colors_final.append(method_colors.get(method, "gray"))

    if all_labels:
        y = np.arange(len(all_labels))
        bar_h = 0.35
        ax.barh(y + bar_h / 2, all_baseline, bar_h, color="lightgray",
                label="Baseline")
        ax.barh(y - bar_h / 2, all_final, bar_h, color=colors_final,
                label="Fine-tuned / Eval")
        for i, (bv, fv, t) in enumerate(
                zip(all_baseline, all_final, all_types)):
            suffix = " (eval)" if t == "eval" else ""
            ax.text(max(bv, fv) + 0.3, i, f"{fv:.1f}{suffix}", va="center",
                    fontsize=8, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(all_labels, fontsize=7)
        ax.set_xlabel("F1 Score (0-100)")
        ax.set_title("Qasper F1: Baseline vs Fine-tuned")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "f1_comparison.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {plots_dir}/f1_comparison.png")

    # Plot 2: Training curves (all runs on one plot, distinct styles)
    line_styles = ["-", "--", "-.", ":"]
    if runs:
        fig, ax = plt.subplots(figsize=(10, 6))
        method_run_count = {}
        for run in runs:
            reward = run["results"].get("training", {}).get(
                "reward_per_iteration", {})
            means = reward.get("mean", [])
            if not means:
                continue
            label = f"{run['method']}/{run['run_name']}"
            color = method_colors.get(run["method"], "gray")
            idx = method_run_count.get(run["method"], 0)
            method_run_count[run["method"]] = idx + 1
            ls = line_styles[idx % len(line_styles)]
            iters = list(range(1, len(means) + 1))
            ax.plot(iters, means, alpha=0.15, linewidth=0.5, color=color)
            # Smoothed curve
            window = max(1, len(means) // 10)
            if len(means) >= window:
                rolling = np.convolve(
                    means, np.ones(window) / window, mode="valid")
                ax.plot(range(window, len(means) + 1), rolling,
                        linewidth=2, label=label, color=color, linestyle=ls)
            else:
                ax.plot(iters, means, linewidth=2, label=label,
                        color=color, linestyle=ls)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward (F1, 0-1)")
        ax.set_title("Training Reward Curves (smoothed)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "training_curves.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {plots_dir}/training_curves.png")


def generate_report(experiment_name):
    experiment_dir = os.path.join(EXPERIMENTS_DIR, experiment_name)
    if not os.path.isdir(experiment_dir):
        print(f"ERROR: {experiment_dir} does not exist")
        return False

    print(f"Generating report for {experiment_name}...")

    runs = collect_runs(experiment_dir)
    if not runs:
        print(f"  No completed runs found in {experiment_dir}")
        return False

    print(f"  Found {len(runs)} runs")

    # Build summary table
    summary = build_summary(runs)

    # Print summary
    print(f"\n{'Method':<12} {'Run':<35} {'Type':<8} {'Baseline':>10} {'Final':>10} {'Delta':>8} {'Time':>8}")
    print("-" * 98)
    for row in summary:
        baseline = f"{row['baseline_f1']:.2f}" if row['baseline_f1'] is not None else "---"
        final = f"{row['final_f1']:.2f}" if row['final_f1'] is not None else "N/A"
        delta = f"{row['improvement']:+.2f}" if row['improvement'] is not None else "---"
        time_h = f"{row['training_time_h']:.1f}h" if row['training_time_h'] else "---"
        rtype = row.get("type", "training")
        print(f"{row['method']:<12} {row['run_name']:<35} {rtype:<8} {baseline:>10} {final:>10} {delta:>8} {time_h:>8}")

    # Generate plots
    generate_plots(experiment_dir, runs)

    # Save report.json
    report = {
        "experiment": experiment_name,
        "num_runs": len(runs),
        "generated_at": datetime.datetime.now().isoformat(),
        "summary": summary,
    }
    report_path = os.path.join(experiment_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate experiment report")
    parser.add_argument("experiment", nargs="?",
                        help="Experiment name or ID (e.g. experiment_1 or 1)")
    parser.add_argument("--all", action="store_true",
                        help="Generate reports for all active experiments")
    args = parser.parse_args()

    if args.all:
        manifest = load_manifest()
        for name, info in manifest["experiments"].items():
            if info["status"] == "active":
                generate_report(name)
                print()
    elif args.experiment:
        name = normalize_name(args.experiment)
        generate_report(name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
